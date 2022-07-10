# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from unicore import registry
from unicore.losses.unicore_loss import (  # noqa
    UnicoreLoss,
)


(
    build_loss_,
    register_loss,
    CRITERION_REGISTRY,
) = registry.setup_registry(
    "--loss", base_class=UnicoreLoss, default="cross_entropy"
)


def build_loss(args, task):
    return build_loss_(args, task)


# automatically import any Python files in the losses/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("unicore.losses." + file_name)
