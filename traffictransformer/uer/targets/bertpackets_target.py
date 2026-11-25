import torch
import torch.nn as nn
from uer.targets import *


# class BertPacketsTarget(MlmTarget):
#     """
#     """

#     def __init__(self, args, vocab_size):
#         super(BertPacketsTarget, self).__init__(args, vocab_size)
#         # MSP.
#         self.msp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
#         self.msp_linear_2 = nn.Linear(args.hidden_size, 4)
#         # self.pos_linear = nn.Linear(args.hidden_size, 10)
#     def forward(self, memory_bank, tgt):
#         """


#         """

#         # Masked language model (MLM).
#         assert type(tgt) == tuple       #tgt_mlm, label, field_modified

#         # Mixed sentence prediction (MSP).
#         output_msp = torch.tanh(self.msp_linear_1(memory_bank[:, 0, :]))
#         output_msp = self.msp_linear_2(output_msp)
#         correct_msp = self.softmax(output_msp).argmax(dim=-1).eq(label).sum()

#         return loss_mlm, loss_msp, loss_fm,correct_mlm, correct_msp, correct_fm, denominator_mlm, denominator_fm

class BertPacketsTarget(nn.Module):

    def __init__(self, args, vocab_size):
        # super(BertPacketsTarget, self).__init__(args, vocab_size)
        super(BertPacketsTarget, self).__init__()
        # MSP.
        self.msp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.msp_linear_2 = nn.Linear(args.hidden_size, 4)
        self.mlm1 = MlmTarget(args, vocab_size)
        self.mlm2 = MlmTarget(args, vocab_size)
        self.criterion = nn.NLLLoss()
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.pos_linear = nn.Linear(args.hidden_size, 10)
    def forward(self, memory_bank, tgt):

        # Masked language model (MLM).
        assert type(tgt) == tuple       #tgt_mlm, label, field_modified
        tgt_mlm, label, field_modified = tgt[0], tgt[1], tgt[2]
        loss_mlm, correct_mlm, denominator_mlm = self.mlm1(memory_bank, tgt_mlm)

        loss_fm, correct_fm, denominator_fm = self.mlm2(memory_bank, field_modified)
        # Mixed sentence prediction (MSP).
        output_msp = torch.tanh(self.msp_linear_1(memory_bank[:, 0, :]))
        output_msp = self.msp_linear_2(output_msp)
        loss_msp = self.criterion(self.softmax(output_msp), label)
        correct_msp = self.softmax(output_msp).argmax(dim=-1).eq(label).sum()

        return loss_mlm, loss_msp, loss_fm,correct_mlm, correct_msp, correct_fm, denominator_mlm, denominator_fm
