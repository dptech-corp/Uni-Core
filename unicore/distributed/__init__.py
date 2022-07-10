# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .module_proxy_wrapper import ModuleProxyWrapper
from .legacy_distributed_data_parallel import LegacyDistributedDataParallel

__all__ = [
    "ModuleProxyWrapper",
]
