# -*- coding: utf-8 -*-
"""循环编码器对应的模型外壳，含逐循环监督与一致性蒸馏。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.models.model import Model


class LoopedModel(Model):
    """循环编码器对应的模型外壳。

    原始 Model.forward 绕过 encoder.forward，直接索引
    encoder.transformer[i] 逐层调用。循环编码器只有一个共享块，
    该写法不兼容，因此这里改为调用编码器自身的 forward。
    embedding、encoder、target 三部分的组装方式与原来完全一致。

    在基线之上新增两个仅作用于预训练的机制，默认关闭、互不影响：

    1. 逐循环监督（loop_sup_weight > 0 开启）。中间各轮循环的隐藏态
       接入共享的 MLM/FM 任务头计算损失，取平均后按权重计入附加损失。
       任务头复用 target 模块的现有参数，不引入任何新参数，
       checkpoint 键名与基线完全对齐。
    2. 循环一致性蒸馏（loop_distill_weight > 0 开启）。中间各轮隐藏态
       向最后一轮隐藏态（阻断梯度）做 MSE 对齐，让浅循环的输出向
       深循环靠拢，防止各轮表征各自漂移。

    附加损失暂存在 self.loop_aux_loss，由 looped_trainer 中的训练器
    读取后加进总损失；target 返回的 loss_info 与基线格式完全一致，
    保证日志中 loss_mlm/loss_sp/loss_fm 的口径不变、可横向对比。
    评估模式下两个机制都不参与计算。
    """

    def __init__(self, args, embedding, encoder, target):
        super().__init__(args, embedding, encoder, target)
        self.loop_sup_weight = getattr(args, "loop_sup_weight", 0.0) or 0.0
        self.loop_distill_weight = getattr(args, "loop_distill_weight", 0.0) or 0.0
        self.loop_aux_loss = None

    def _final_norm(self):
        """layernorm_positioning 为 pre 时，编码器末尾还有一层 LayerNorm。

        中间轮隐藏态接任务头或做对齐前，需要过同一层归一化，
        保持与最终输出的分布一致。post 时返回 None 表示无需处理。
        """
        if getattr(self.encoder, "layernorm_positioning", "post") == "pre":
            return self.encoder.layer_norm
        return None

    def _token_supervision(self, hidden, tgt):
        """中间轮的 token 级监督损失，复用 target 的共享任务头。

        支持 bertpackets（MLM + 字段修改预测）与 mlm 两类 target，
        其余 target 返回 None 表示跳过逐循环监督。
        """
        if hasattr(self.target, "mlm1") and hasattr(self.target, "mlm2"):
            # bertpackets：tgt 为 (tgt_mlm, label, field_modified)。
            tgt_mlm, _, field_modified = tgt
            loss_mlm, _, _ = self.target.mlm1.mlm(hidden, tgt_mlm)
            loss_fm, _, _ = self.target.mlm2.mlm(hidden, field_modified)
            return loss_mlm + loss_fm
        if hasattr(self.target, "mlm"):
            loss_mlm, _, _ = self.target.mlm(hidden, tgt)
            return loss_mlm
        return None

    def forward(self, src, tgt, seg, proto=None):
        self.loop_aux_loss = None
        # 只有训练模式 + 循环编码器 + 权重非零时才需要中间轮隐藏态。
        need_loop_hiddens = (
            self.training
            and (self.loop_sup_weight > 0.0 or self.loop_distill_weight > 0.0)
            and hasattr(self.encoder, "loop_num")
        )
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg, proto=proto,
                              output_loop_hiddens=need_loop_hiddens)
        # 兼容三种返回形态：张量、（隐藏态, 隐藏态列表）、
        # 停机编码器的（隐藏态, act_stats, 隐藏态列表）。
        loop_hiddens = None
        if isinstance(output, (tuple, list)):
            hidden = output[0]
            for item in output[1:]:
                if isinstance(item, list):
                    loop_hiddens = item
        else:
            hidden = output
        loss_info = self.target(hidden, tgt)

        # 随机深度下本轮可能只循环 1 次，此时没有中间轮，跳过附加损失。
        if need_loop_hiddens and loop_hiddens and len(loop_hiddens) > 1:
            norm = self._final_norm()
            intermediates = loop_hiddens[:-1]
            if norm is not None:
                intermediates = [norm(h) for h in intermediates]
            aux = hidden.new_zeros(())
            n_inter = len(intermediates)

            if self.loop_sup_weight > 0.0:
                sup = None
                for h in intermediates:
                    loss_i = self._token_supervision(h, tgt)
                    if loss_i is None:
                        sup = None
                        break
                    sup = loss_i if sup is None else sup + loss_i
                if sup is not None:
                    aux = aux + self.loop_sup_weight * sup / n_inter        #逐轮监督消融。
                    # aux = aux        #逐轮监督消融。

            if self.loop_distill_weight > 0.0:
                # 最后一轮输出作教师，阻断梯度，只让浅层向深层靠拢。
                ref = hidden.detach()
                dis = hidden.new_zeros(())
                for h in intermediates:
                    dis = dis + F.mse_loss(h, ref)
                aux = aux + self.loop_distill_weight * dis / n_inter        #一致性蒸馏消融。
                # aux = aux        #一致性蒸馏消融。

            self.loop_aux_loss = aux
        return loss_info
