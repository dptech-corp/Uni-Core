# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various unicore models.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseUnicoreModel(nn.Module):
    """Base class for unicore models."""

    def __init__(self):
        super().__init__()

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError("Model must implement the build_model method")

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_args = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. 
        """
        return super().load_state_dict(state_dict, strict)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""

        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)
