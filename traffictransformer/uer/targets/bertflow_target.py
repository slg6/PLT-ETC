import torch
import torch.nn as nn
from uer.targets import *


class BertFlowTarget(MlmTarget):

    def __init__(self, args, vocab_size):
        super(BertFlowTarget, self).__init__(args, vocab_size)
        # MSP.
        self.msp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.msp_linear_2 = nn.Linear(args.hidden_size, 5)
    def forward(self, memory_bank, tgt):

        # Masked language model (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_msp = tgt[0], tgt[1]
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)

        # Mixed sentence prediction (MSP).
        output_msp = torch.tanh(self.msp_linear_1(memory_bank[:, 0, :]))
        output_msp = self.msp_linear_2(output_msp)
        loss_msp = self.criterion(self.softmax(output_msp), tgt_msp)
        correct_msp = self.softmax(output_msp).argmax(dim=-1).eq(tgt_msp).sum()

        return loss_mlm, loss_msp, correct_mlm, correct_msp, denominator
