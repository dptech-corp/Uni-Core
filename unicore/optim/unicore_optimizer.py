# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from unicore import utils

class UnicoreOptimizer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._grad_buffer = None
        self._need_sync_grad_buf = False

    @classmethod
    def add_args(cls, parser):
        """Add optimizer-specific arguments to the parser."""
        pass

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, "_optimizer"):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Reset optimizer instance."""
        if not hasattr(self, "_optimizer"):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        self._optimizer = optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.param_groups[0]["lr"]

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward()

    def all_reduce_grads(self, module):
        """Manually all-reduce gradients (if required)."""
        self.__sync_grad_from_buf__()
        if hasattr(module, "all_reduce_grads"):
            module.all_reduce_grads()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def per_sample_clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        if max_norm <= 0.0:
            return 0.0
        if self._grad_buffer is None:
            self._grad_buffer = [torch.zeros_like(g) for g in self.params]
        gnorm = utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self._grad_buffer[i] += p.grad
            p.grad = None
        self._need_sync_grad_buf = True
        return gnorm
    
    def __sync_grad_from_buf__(self):
        if self._need_sync_grad_buf:
            assert self._grad_buffer is not None
            for i, p in enumerate(self.params):
                p.grad = self._grad_buffer[i]
            self._need_sync_grad_buf = False

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        self.__sync_grad_from_buf__()
        return utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)

    def step(self, closure=None, scale=1.0, groups=None):
        """Performs a single optimization step."""
        self.__sync_grad_from_buf__()
        if self.supports_step_with_scale:
            if self.supports_groups:
                self.optimizer.step(closure, scale=scale, groups=groups)
            else:
                self.optimizer.step(closure, scale=scale)
        else:
            if scale != 1.0:
                self.multiply_grads(1.0 / scale)
            if self.supports_groups:
                self.optimizer.step(closure, groups=groups)
            else:
                self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()
        self._need_sync_grad_buf = False
        if self._grad_buffer is not None:
            for t in self._grad_buffer:
                t.zero_()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, "supports_memory_efficient_fp16"):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    @property
    def supports_step_with_scale(self):
        if hasattr(self.optimizer, "supports_step_with_scale"):
            return self.optimizer.supports_step_with_scale
        return False

    @property
    def supports_groups(self):
        if hasattr(self.optimizer, "supports_groups"):
            return self.optimizer.supports_groups
        return False

    @property
    def supports_flat_params(self):
        """
        Whether the optimizer supports collapsing of the model
        parameters/gradients into a single contiguous Tensor.
        """
        if hasattr(self.optimizer, "supports_flat_params"):
            return self.optimizer.supports_flat_params
        return False


