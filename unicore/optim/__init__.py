# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from unicore import registry
from unicore.optim.unicore_optimizer import (  # noqa
    UnicoreOptimizer,
)
from unicore.optim.fp16_optimizer import FP16Optimizer

__all__ = [
    "UnicoreOptimizer",
    "FP16Optimizer",
]

(
    _build_optimizer,
    register_optimizer,
    OPTIMIZER_REGISTRY
) = registry.setup_registry("--optimizer", base_class=UnicoreOptimizer, default='adam')


def build_optimizer(args, params, *extra_args, **extra_kwargs):
    if all(isinstance(p, dict) for p in params):
        params = [t for p in params for t in p.values()]
    params = list(filter(lambda p: p.requires_grad, params))
    return _build_optimizer(args, params, *extra_args, **extra_kwargs)


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("unicore.optim." + file_name)
