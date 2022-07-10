# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace

from unicore.optim import UnicoreOptimizer


class UnicoreLRScheduler(object):
    def __init__(self, args, optimizer, total_train_steps):
        super().__init__()
        if optimizer is not None and not isinstance(optimizer, UnicoreOptimizer):
            raise ValueError("optimizer must be an instance of UnicoreOptimizer")
        self.args = args
        self.optimizer = optimizer
        self.total_train_steps = total_train_steps
        self.best = None

    @classmethod
    def add_args(cls, parser):
        """Add arguments to the parser for this LR scheduler."""
        pass

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {"best": self.best}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict["best"]

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        pass

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            if self.best is None:
                self.best = val_loss
            else:
                self.best = min(self.best, val_loss)

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.get_lr()

