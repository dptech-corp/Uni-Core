# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss

@register_loss("masked_lm")
class MaskedLMLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        masked_tokens = torch.where(
            masked_tokens.any(),
            masked_tokens,
            masked_tokens.new([True]),
        )
        logits = model(**sample["net_input"], masked_tokens=masked_tokens)
        target = sample['target']
        if masked_tokens is not None:
            target = target[masked_tokens]
        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            target,
            ignore_index=self.padding_idx,
            reduction='sum',
        )
        logging_output = {
            "loss": loss.data,
            "bsz": sample["target"].size(0),
            "sample_size": sample_size,
            "seq_len": sample["target"].size(1) * sample["target"].size(0),
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "seq_len", seq_len / bsz, 1, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
