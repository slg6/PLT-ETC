# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from loopedtransformer.looped_encoder import LoopedTransformerEncoder


class HaltingLoopedTransformerEncoder(LoopedTransformerEncoder):

    def __init__(self, args):
        super().__init__(args)
        self.halting_threshold = getattr(args, "halting_threshold", 0.99)
        self.halting_pooling = getattr(args, "halting_pooling", None) \
            or getattr(args, "pooling", "first")
        self.halting_linear = nn.Linear(args.hidden_size, 1)

        self.halt_check_every = getattr(args, "halt_check_every", 0) or 1  # or 2

    def _pool_for_halting(self, hidden):
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
