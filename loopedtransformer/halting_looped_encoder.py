# -*- coding: utf-8 -*-
"""借鉴 Universal Transformer 动态停机（ACT）的循环编码器。

在参数共享循环基线（LoopedTransformerEncoder）之上新增一个停机头：
一个 hidden_size -> 1 的线性层加 sigmoid，每轮循环给每个样本打一个
"把握度"分数 p。累计把握度超过阈值的样本提前停机，之后各轮不再更新
其隐藏态；达到 loop_num 仍未停机的样本由上限兜底。

停机判定完全复刻 Universal Transformer 官方实现（截图同款）：
- still_running：还没停机的样本掩码；
- new_halted：本轮加完 p 后首次越过阈值的样本；
- remainders：越过阈值部分（1 - 累计把握度），用于把刚停机样本的
  把握度恰好补到 1，同时作为本轮状态更新的权重；
- update_weights：未停机样本按 p 采纳新状态，刚停机样本按 remainders
  采纳，已停机样本权重为 0、保持原状态。

与基线的两点差异：
1. 停机是逐样本的：分类任务中简单样本一两轮即停，困难样本多循环，
   loop_num 只作为 max_steps 兜底。
2. forward 恒返回 (hidden, act_stats) 二元组，act_stats 内含
   n_updates（每个样本实际消耗的循环步数），供分类头把
   ponder penalty（步数均值 x 权重）加进损失，逼模型学会早停。
   评估模式下若整个 batch 全部停机则提前退出循环，节省推理算力。

注意：停机头参数不参与循环预训练，从预训练权重加载时随机初始化，
在动态停机微调阶段随分类损失一起端到端学习。
"""
import torch
import torch.nn as nn

from loopedtransformer.looped_encoder import LoopedTransformerEncoder


class HaltingLoopedTransformerEncoder(LoopedTransformerEncoder):
    """带 ACT 动态停机的循环编码器（逐样本早停）。"""

    def __init__(self, args):
        super().__init__(args)
        self.halting_threshold = getattr(args, "halting_threshold", 0.99)
        self.halting_pooling = getattr(args, "halting_pooling", None) \
            or getattr(args, "pooling", "first")
        self.halting_linear = nn.Linear(args.hidden_size, 1)

        self.halt_check_every = getattr(args, "halt_check_every", 0) or 1  # or 2

    def _pool_for_halting(self, hidden):
        """按 halting_pooling 把 [batch x seq x hidden] 压成 [batch x hidden]。"""
        if self.halting_pooling == "mean":
            return torch.mean(hidden, dim=1)
        if self.halting_pooling == "max":
            return torch.max(hidden, dim=1)[0]
        if self.halting_pooling == "last":
            return hidden[:, -1, :]
        return hidden[:, 0, :]

    def forward(self, emb, seg, input_ids=None, proto=None,
                output_attentions=False, output_loop_hiddens=False):
        if self.factorized_embedding_parameterization:
            emb = self.linear(emb)

        batch_size, seq_length, _ = emb.size()
        mask = (seg > 0). \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0

        hidden = emb
        if self.relative_position_embedding:
            position_bias = self.relative_pos_emb(hidden, hidden)
        else:
            position_bias = None

        loop_hiddens = []
        probss = []
        halting_probability = torch.zeros(batch_size, device=emb.device)
        remainders = torch.zeros(batch_size, device=emb.device)
        n_updates = torch.zeros(batch_size, device=emb.device)

        for i in range(self.loop_num):
            still_running = (halting_probability < 1.0).float()
            p = torch.sigmoid(self.halting_linear(
                self._pool_for_halting(hidden))).squeeze(-1)

            increment = p * still_running
            new_halted = (
                halting_probability + increment > self.halting_threshold
            ).float() * still_running
            still_running = (
                halting_probability + increment <= self.halting_threshold
            ).float() * still_running

            halting_probability = halting_probability + increment
            remainders = remainders + new_halted * (1.0 - halting_probability)
            halting_probability = halting_probability + new_halted * remainders
            n_updates = n_updates + still_running + new_halted

            update_weights = (
                increment + new_halted * remainders
            ).view(-1, 1, 1)

            transformed, probs = self.transformer(hidden, mask,
                                                  position_bias=position_bias)
            # hidden = transformed * update_weights + hidden * (1.0 - update_weights)
            # hidden = torch.addcmul(hidden, transformed - hidden, update_weights)
            hidden = torch.lerp(hidden, transformed, update_weights)

            if output_loop_hiddens:
                loop_hiddens.append(hidden)
            if output_attentions:
                probss.append(probs)

            if (not self.training
                    and i+1 >= 8    # or n
                    and (i + 1) % self.halt_check_every == 0
                    and still_running.sum().item() == 0.0):
                break

        if self.layernorm_positioning == "pre":
            hidden = self.layer_norm(hidden)

        act_stats = {
            "n_updates": n_updates,
            "halting_probability": halting_probability,
            "remainders": remainders,
        }
        outputs = [hidden, act_stats]
        if output_attentions:
            outputs.append(probss)
        if output_loop_hiddens:
            outputs.append(loop_hiddens)
        return tuple(outputs)
