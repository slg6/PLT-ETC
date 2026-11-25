import random

import torch
import torch.nn as nn
from uer.encoders.transformer_encoder import TransformerEncoder


class LoopedTransformerEncoder(TransformerEncoder):

    def __init__(self, args):
        assert not args.is_moe, "参数共享基线暂不支持 MoE。"
        assert args.mask == "fully_visible", "基线版本仅支持 fully_visible 掩码。"
        args.parameter_sharing = True
        super().__init__(args)
        self.loop_num = getattr(args, "loop_num", None) or args.layers_num
        self.loop_num_min = getattr(args, "loop_num_min", None)

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
