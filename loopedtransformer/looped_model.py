# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from uer.models.model import Model


class LoopedModel(Model):

    def __init__(self, args, embedding, encoder, target):
        super().__init__(args, embedding, encoder, target)
        self.loop_sup_weight = getattr(args, "loop_sup_weight", 0.0) or 0.0
        self.loop_distill_weight = getattr(args, "loop_distill_weight", 0.0) or 0.0
        self.loop_aux_loss = None

    def _final_norm(self):
        if getattr(self.encoder, "layernorm_positioning", "post") == "pre":
            return self.encoder.layer_norm
        return None

    def _token_supervision(self, hidden, tgt):
        if hasattr(self.target, "mlm1") and hasattr(self.target, "mlm2"):
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
        need_loop_hiddens = (
            self.training
            and (self.loop_sup_weight > 0.0 or self.loop_distill_weight > 0.0)
            and hasattr(self.encoder, "loop_num")
        )
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg, proto=proto,
                              output_loop_hiddens=need_loop_hiddens)
        loop_hiddens = None
        if isinstance(output, (tuple, list)):
            hidden = output[0]
            for item in output[1:]:
                if isinstance(item, list):
                    loop_hiddens = item
        else:
            hidden = output
        loss_info = self.target(hidden, tgt)

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
                    aux = aux + self.loop_sup_weight * sup / n_inter

            if self.loop_distill_weight > 0.0:
                ref = hidden.detach()
                dis = hidden.new_zeros(())
                for h in intermediates:
                    dis = dis + F.mse_loss(h, ref)
                aux = aux + self.loop_distill_weight * dis / n_inter

            self.loop_aux_loss = aux
        return loss_info
