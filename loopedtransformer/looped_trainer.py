# -*- coding: utf-8 -*-
from uer.trainer import BertPacketsTrainer


class LoopedBertPacketsTrainer(BertPacketsTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.total_loss_loop_aux = 0.0

    def forward_propagation(self, batch, model):
        loss = super().forward_propagation(batch, model)
        raw_model = model.module if hasattr(model, "module") else model
        aux = getattr(raw_model, "loop_aux_loss", None)
        if aux is not None:
            loss = loss + aux / self.accumulation_steps
            self.total_loss += aux.item()
            self.total_loss_loop_aux += aux.item()
        return loss

    def report_and_reset_stats(self):
        import logging
        import time
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                            handlers=[
                                logging.FileHandler(self.log_file),
                                logging.StreamHandler()
                            ])
        logger = logging.getLogger(__name__)
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        log_message = (
            f"| {self.current_step:8d}/{self.total_steps:8d} steps"
            f"| {time.time() - self.start_time:3.3f} s"
            f"| {done_tokens / (time.time() - self.start_time):8.2f} tokens/s"
            f"| loss {self.total_loss / self.report_steps:7.2f}"
            f"| loss_mlm: {self.total_loss_mlm / self.report_steps:3.3f}"
            f"| loss_sp: {self.total_loss_sp / self.report_steps:3.3f}"
            f"| loss_fm: {self.total_loss_fm / self.report_steps:3.3f}"
            f"| loop_aux: {self.total_loss_loop_aux / self.report_steps:3.3f}"
            f"| acc_mlm: {self.total_correct_mlm / self.total_denominator:3.3f}"
            f"| acc_sp: {self.total_correct_sp / self.total_instances:3.3f}"
            f"| acc_fm: {self.total_correct_fm / self.total_denominator_fm:3.3f}"
        )
        logger.info(log_message)

        self.total_loss, self.total_loss_mlm, self.total_loss_sp, self.total_loss_fm = 0.0, 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator, self.total_denominator_fm = 0.0, 0.0, 0.0
        self.total_correct_sp, self.total_correct_fm, self.total_instances = 0.0, 0.0, 0.0
        self.total_loss_loop_aux = 0.0
