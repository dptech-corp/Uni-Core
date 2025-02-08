# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Collection
from typing import List

from unicore.optim.lr_scheduler import UnicoreLRScheduler, register_lr_scheduler


@register_lr_scheduler("cosine")
class CosineLRSchedule(UnicoreLRScheduler):
    """Assign LR based on a cyclical schedule that follows the cosine function.

    See https://arxiv.org/pdf/1608.03983.pdf for details.

    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    max learning rate (``--lr``).

    During warmup::

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup::

      lr = args.min_lr + 0.5*(args.lr - args.min_lr)*(1 + cos(t_curr / t_i))

    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """

    def __init__(self, args, unicore_optimizer, total_train_steps):
        super().__init__(args, unicore_optimizer, total_train_steps)
        if isinstance(args.lr, Collection) and len(args.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with cosine."
                f" Consider --lr-scheduler=fixed instead. ({args.lr})"
            )

        self.max_lr = args.lr[0] if isinstance(args.lr, Collection) else args.lr
        assert (
            self.max_lr > args.min_lr
        ), f"max_lr (={args.lr}) must be more than min_lr (={args.min_lr})"

        assert total_train_steps is not None
        if self.args.warmup_ratio > 0:
            self.warmup_updates = int(self.args.warmup_ratio * total_train_steps)
        else:
            self.warmup_updates = args.warmup_updates

        warmup_end_lr = self.max_lr
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = args.min_lr

        self.t_mult = args.t_mult
        self.period = args.lr_period_updates

        if self.period <= 0:
            self.period = total_train_steps - self.warmup_updates

        if self.warmup_updates > 0:
            # linearly warmup for the first args.warmup_updates
            self.lr_step = (warmup_end_lr - args.warmup_init_lr) / self.warmup_updates
        else:
            self.lr_step = 1

        self.lr_shrink = args.lr_shrink

        # initial learning rate
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-ratio', default=-1.0, type=float, metavar='N',
                            help='warmup the learning rate linearly for the first N-percent updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--min-lr', type=float, metavar='LR',
                            help='min learning rate')
        parser.add_argument('--max-lr', type=float, metavar='LR',
                            help='max learning rate, must be more than args.lr')
        parser.add_argument('--t-mult', default=1, type=float, metavar='LR',
                            help='factor to grow the length of each period')
        parser.add_argument('--lr-period-updates', default=-1, type=float, metavar='LR',
                            help='initial number of updates per period')
        parser.add_argument('--lr-shrink', default=0.1, type=float, metavar='LS',
                            help='shrink factor for annealing')
        # fmt: on

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates * self.lr_step
        else:
            curr_updates = num_updates - self.warmup_updates
            if self.t_mult != 1:
                i = math.floor(
                    math.log(
                        1 - curr_updates / self.period * (1 - self.t_mult), self.t_mult
                    )
                )
                t_i = self.t_mult**i * self.period
                t_curr = (
                    curr_updates
                    - (1 - self.t_mult**i) / (1 - self.t_mult) * self.period
                )
                r = float(t_curr) / t_i
            else:
                # force i to zero in one-cycle
                i = 0
                t_i = self.period
                t_curr = curr_updates
                r = float(t_curr) / t_i
                r = min(1.0, r)

            lr_shrink = self.lr_shrink**i
            min_lr = self.args.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink

            self.lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * r))

        self.optimizer.set_lr(self.lr)
        return self.lr
