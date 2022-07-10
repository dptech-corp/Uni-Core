# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from unicore.X import Y`
from unicore.distributed import utils as distributed_utils
from unicore.logging import meters, metrics, progress_bar  # noqa

sys.modules["unicore.distributed_utils"] = distributed_utils
sys.modules["unicore.meters"] = meters
sys.modules["unicore.metrics"] = metrics
sys.modules["unicore.progress_bar"] = progress_bar

import unicore.losses  # noqa
import unicore.distributed  # noqa
import unicore.models  # noqa
import unicore.modules  # noqa
import unicore.optim  # noqa
import unicore.optim.lr_scheduler  # noqa
import unicore.tasks  # noqa

