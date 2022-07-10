# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from unicore.optim.lr_scheduler import UnicoreLRScheduler, register_lr_scheduler

@register_lr_scheduler("exponential_decay")
class ExponentialDecayLRSchedule(UnicoreLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer, total_train_steps):
        super().__init__(args, optimizer, total_train_steps)
        self.warmup_updates = args.warmup_updates
        self.lr = args.lr[0]
        if self.warmup_updates > 0:
            self.warmup_factor = 1.0 / self.warmup_updates
        else:
            self.warmup_factor = 1.0
        self.decay_ratio = args.decay_ratio
        self.decay_steps = args.decay_steps
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        self.stair_decay = getattr(args, "stair_decay", False)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup-updates', default=1000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--decay-ratio', default=0.95, type=float)
        parser.add_argument('--decay-steps', default=500, type=int)
        parser.add_argument('--stair-decay', action="store_true")

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            self.warmup_factor = num_updates / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        else:
            if self.stair_decay:
                step = num_updates
                lr = self.lr * float(self.decay_ratio ** (int(step // self.decay_steps)))
            else:
                step = num_updates - self.warmup_updates
                lr = self.lr * float(self.decay_ratio ** (float(step / self.decay_steps)))
        self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()
