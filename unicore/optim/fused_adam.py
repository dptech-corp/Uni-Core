# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


def get_fused_adam_class():
    try:
        global unicore_fused_adam
        import importlib
        unicore_fused_adam = importlib.import_module("unicore_fused_adam")
        return FusedAdam
    except ImportError:
        pass
    return None


class FusedAdam(torch.optim.Optimizer):
    """
    Implements Adam algorithm. Currently GPU-only. Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Compared to the original version in Apex, the unicore version casts grads
    and params to FP32 internally to support ``--memory-efficient-fp16``.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!
        eps_inside_sqrt (boolean, optional): in the "update parameters" step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params,
                 lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0., amsgrad=False):
        global unicore_fused_adam
        import importlib
        unicore_fused_adam = importlib.import_module("unicore_fused_adam")

        if amsgrad:
            raise RuntimeError("FusedAdam does not support the AMSGrad variant.")
        defaults = {
            "lr": lr,
            "bias_correction": bias_correction,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    @property
    def supports_step_with_scale(self):
        return True

    def step(self, closure=None, scale=1.):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # compute combined scale factor for this group
            combined_scale = scale
            bias_correction = 1 if group.get("bias_correction", 1) else 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "FusedAdam does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.float)
                else:
                    state["exp_avg"] = state["exp_avg"].to(dtype=torch.float)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(dtype=torch.float)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                with torch.cuda.device(p.device):
                    unicore_fused_adam.adam(p.data,
                                         exp_avg,
                                         exp_avg_sq,
                                         grad,
                                         group["lr"],
                                         beta1,
                                         beta2,
                                         group["eps"],
                                         combined_scale,
                                         state["step"],
                                         bias_correction,
                                         group["weight_decay"])

        return loss


