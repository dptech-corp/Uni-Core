# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F

try:
    import unicore_fused_layernorm
    import unicore_fused_layernorm_backward_gamma_beta
    HAS_LAYER_NORM = True
except:
    print("fused_layer_norm is not installed corrected")
    HAS_LAYER_NORM = False

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
    HAS_LAYER_NORM = False

class FusedLayerNormFastFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, weight, bias, normalized_shape, eps):
    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    input = input.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    output, mean, invvar = unicore_fused_layernorm.forward(
        input, ctx.normalized_shape, weight, bias, ctx.eps)
    ctx.save_for_backward(input, weight, bias, mean, invvar)
    return output
  @staticmethod
  def backward(ctx, grad_output):
    input_, weight_, bias_, mean, invvar = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    grad_input = unicore_fused_layernorm.backward(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        weight_, bias_, ctx.eps)
    grad_weight, grad_bias = unicore_fused_layernorm_backward_gamma_beta.backward(
        grad_output.contiguous(), mean, invvar,
        input_, ctx.normalized_shape,
        weight_, bias_, ctx.eps)
    return grad_input, grad_weight, grad_bias, None, None


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input):
        if not input.is_cuda or not HAS_LAYER_NORM:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        return FusedLayerNormFastFunction.apply(
            input, self.weight, self.bias, self.normalized_shape, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine=True'.format(**self.__dict__)
