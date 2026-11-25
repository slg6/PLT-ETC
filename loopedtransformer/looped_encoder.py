import random

import torch
import torch.nn as nn
from uer.encoders.transformer_encoder import TransformerEncoder


class LoopedTransformerEncoder(TransformerEncoder):
    """循环 Transformer 编码器（参数共享基线）。

    与原始 TransformerEncoder 的四点区别：
    1. 强制参数共享，整个编码器只有一个共享的 Transformer 块。
    2. 循环次数由 loop_num 控制，与配置中的 layers_num 解耦。
       layers_num 在本编码器中仅作兼容保留，不参与实际计算。
    3. 可选返回每次循环的隐藏态，供表征诊断（如 CKA、各向异性）使用。
    4. 支持随机深度预训练：设置 loop_num_min 后，训练模式下每次前向
       从 [loop_num_min, loop_num] 均匀采样实际循环次数，使任意深度的
       输出都被训练到；评估模式始终跑满 loop_num，保证评测可比。

    注意：原始 Model.forward 直接索引 encoder.transformer[i]，
    与单共享块结构不兼容，因此必须搭配 looped_model.LoopedModel 使用。
    """

    def __init__(self, args):
        assert not args.is_moe, "参数共享基线暂不支持 MoE。"
        assert args.mask == "fully_visible", "基线版本仅支持 fully_visible 掩码。"
        args.parameter_sharing = True
        super().__init__(args)
        self.loop_num = getattr(args, "loop_num", None) or args.layers_num
        self.loop_num_min = getattr(args, "loop_num_min", None)

    def forward(self, emb, seg, input_ids=None, proto=None,
                output_attentions=False, output_loop_hiddens=False):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            output_loop_hiddens: 为 True 时额外返回每次循环后的隐藏态列表。
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
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

        loop_num = self.loop_num
        if self.training and self.loop_num_min is not None \
                and self.loop_num_min < self.loop_num:
            loop_num = random.randint(self.loop_num_min, self.loop_num)

        loop_hiddens = []
        probss = []
        for _ in range(loop_num):
            hidden, probs = self.transformer(hidden, mask, position_bias=position_bias)
            if output_loop_hiddens:
                loop_hiddens.append(hidden)
            if output_attentions:
                probss.append(probs)

        if self.layernorm_positioning == "pre":
            hidden = self.layer_norm(hidden)

        outputs = [hidden]
        if output_attentions:
            outputs.append(probss)
        if output_loop_hiddens:
            outputs.append(loop_hiddens)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
