# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from collections.abc import Collection
from typing import List

import torch
import torch.optim
from unicore.optim import UnicoreOptimizer, register_optimizer
from unicore.optim.fused_adam import get_fused_adam_class


logger = logging.getLogger(__name__)


@register_optimizer("adam")
class UnicoreAdam(UnicoreOptimizer):
    """Adam optimizer for unicore.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, args, params):
        super().__init__(args)
        fused_adam_cls = get_fused_adam_class()
        # priority is governed by speed: custom fused > fused > foreach
        use_fused_adam = (
            not getattr(args, "use_old_adam", False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 7
        )
        if use_fused_adam:
            logger.info("using FusedAdam")
            self._optimizer = fused_adam_cls(params, **self.optimizer_config)
        else:
            self._optimizer = torch.optim.AdamW(params, fused=args.use_fused_optimizer, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adam-betas', default='(0.9, 0.999)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.args.lr[0]
            if isinstance(self.args.lr, Collection)
            else self.args.lr,
            "betas": eval(self.args.adam_betas),
            "eps": self.args.adam_eps,
            "weight_decay": self.args.weight_decay,
        }
