# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from unicore.distributed import (
    ModuleProxyWrapper, LegacyDistributedDataParallel
)


logger = logging.getLogger(__name__)

def DistributedUnicoreModel(args, model, process_group, device):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): unicore args
        model (BaseUnicoreModel): model to wrap
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
        device: device to move model to
    """
    assert isinstance(model, nn.Module)
    if args.ddp_backend in {"c10d", "pytorch_ddp"}:
        wrapped_model = DistributedDataParallel(
            module=model.to(device),
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            bucket_cap_mb=args.bucket_cap_mb,
            process_group=process_group,
            find_unused_parameters=args.find_unused_parameters,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {'apex'}:
        import apex
        wrapped_model = apex.parallel.DistributedDataParallel(
            module=model.to(device)
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {"no_c10d", "legacy_ddp"}:
        wrapped_model = LegacyDistributedDataParallel(
            module=model.to(device),
            buffer_size=2 ** 28,
            process_group=process_group,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    else:
        raise ValueError("Unknown --ddp-backend: " + args.ddp_backend)

    return wrapped_model
