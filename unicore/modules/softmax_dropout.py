# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

try:
    import unicore_fused_softmax_dropout
    HAS_SOFTMAX = True
except:
    print("fused_softmax is not installed corrected")
    HAS_SOFTMAX = False

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
    HAS_SOFTMAX = False

class SoftmaxDropoutFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, is_training, inputs, mask, bias, dropout_prob):
        (
            dropout_results,
            dropout_mask,
            softmax_results,
        ) = unicore_fused_softmax_dropout.forward(
            is_training, inputs, mask, bias, dropout_prob, None
        )
        if is_training:
            ctx.dropout_prob = dropout_prob
            ctx.save_for_backward(softmax_results, dropout_mask)
            ctx.has_bias = bias is not None and bias.requires_grad
            if ctx.has_bias:
                ctx.bias_batch_dim = bias.shape[0]
        return dropout_results

    @staticmethod
    def backward(ctx, grad_output):
        softmax_results, dropout_mask = ctx.saved_tensors
        dropout_prob = ctx.dropout_prob
        grad_output = grad_output.contiguous()
        grad_input = unicore_fused_softmax_dropout.backward(
            grad_output, softmax_results, dropout_mask, dropout_prob
        )
        if ctx.has_bias:
            grad_bias = grad_input.view(
                -1, ctx.bias_batch_dim, grad_input.shape[-2], grad_input.shape[-1]
            ).sum(dim=0)
        else:
            grad_bias = None
        return None, grad_input, None, grad_bias, None


def _check_mask(mask, input):
    """Validate that mask is compatible with the CUDA fused kernel.

    Uses explicit shape checks instead of try/except (expert 8.1): the old
    version accessed ``mask.shape[-3]`` inside a bare ``except`` which could
    raise ``IndexError`` on low-rank inputs (rank < 3) and crash instead of
    falling back. Now returns False cleanly for any incompatible shape.
    """
    if mask.dtype != input.dtype:
        return False
    if mask.ndim != input.ndim:
        return False
    if mask.ndim < 3:
        return False
    if mask.shape[-3] != 1 and mask.shape[-3] != input.shape[-3]:
        return False
    if mask.shape[-3] == 1:
        if mask.shape[-2] != 1:
            return False
    else:
        if mask.shape[-2] != 1 and mask.shape[-2] != input.shape[-2]:
            return False
    return True


def _check_bias(bias, input):
    """Validate that bias is compatible with the CUDA fused kernel.

    Uses explicit shape checks instead of try/except (expert 8.1).
    """
    if bias.dtype != input.dtype:
        return False
    if bias.ndim != input.ndim:
        return False
    if bias.ndim < 2:
        return False
    if bias.shape[-1] != input.shape[-1]:
        return False
    if bias.shape[-2] != input.shape[-2]:
        return False
    len_shape = input.ndim
    if len_shape > 3:
        if bias.shape[-3] != input.shape[-3]:
            return False
        offset = 3
    else:
        offset = 2
    prev_non_one = True
    for i in range(len_shape - offset - 1, -1, -1):
        if prev_non_one:
            if bias.shape[i] != input.shape[i] and bias.shape[i] != 1:
                return False
        else:
            if bias.shape[i] != 1:
                return False
        prev_non_one = bias.shape[i] != 1
    return True


def _prepare_input(input, inplace):
    """Make input contiguous and (if not inplace) clone it.

    This is the *single* place input preprocessing happens — extracted to
    preserve upstream's single-preprocessing semantics. The public
    ``softmax_dropout`` calls this once; the reference helper and all
    backend fast paths receive already-prepared input and must NOT repeat
    contiguous/clone.
    """
    x = input.contiguous()
    if not inplace:
        x = x.clone()
    return x


def _softmax_dropout_reference_prepared(
    x, dropout_prob, is_training=True, mask=None, bias=None, inplace=True,
):
    """Generic PyTorch softmax + dropout + additive mask/bias.

    ``x`` must already be prepared by ``_prepare_input`` (contiguous, and
    cloned if ``inplace=False``). This function does NOT repeat that work.

    Since ``x`` is already a safe copy (cloned by ``_prepare_input`` when
    ``inplace=False``), mask/bias are applied in-place via ``add_`` to
    avoid extra tensor allocations. This matches upstream's allocation
    semantics: the caller's input is never modified, and no intermediate
    tensors are created for mask/bias addition.

    This is the exact computation that upstream's ``else`` branch performs:
    ``input += mask; input += bias; F.dropout(F.softmax(input, -1), ...)``.
    """
    if mask is not None:
        x.add_(mask)
    if bias is not None:
        x.add_(bias)
    return F.dropout(F.softmax(x, dim=-1), p=dropout_prob, training=is_training)


def _softmax_dropout_reference(
    input,
    dropout_prob,
    is_training=True,
    mask=None,
    bias=None,
    inplace=True,
):
    """Generic PyTorch softmax + dropout + additive mask/bias.

    Convenience wrapper: prepares input then delegates to
    ``_softmax_dropout_reference_prepared``. Kept for backwards
    compatibility with existing tests that call the reference helper
    directly with a raw (un-prepared) tensor.
    """
    x = _prepare_input(input, inplace)
    return _softmax_dropout_reference_prepared(
        x, dropout_prob, is_training, mask, bias, inplace
    )


def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None, inplace=True):
    """softmax dropout, and mask, bias are optional.
    Args:
        input (torch.Tensor): input tensor
        dropout_prob (float): dropout probability
        is_training (bool, optional): is in training or not. Defaults to True.
        mask (torch.Tensor, optional): the mask tensor, use as input + mask . Defaults to None.
        bias (torch.Tensor, optional): the bias tensor, use as input + bias . Defaults to None.

    Returns:
        torch.Tensor: the result after softmax
    """
    input = _prepare_input(input, inplace)
    if input.is_cuda and HAS_SOFTMAX:
        input_size = input.size()
        if mask is not None:
            if _check_mask(mask, input):
                mask = mask.contiguous().view(-1, mask.shape[-2], mask.shape[-1])
            else:
                input += mask
                mask = None
        if bias is not None:
            if _check_bias(bias, input):
                bias = bias.contiguous().view(-1, input_size[-2], input_size[-1])
            else:
                input += bias
                bias = None
        input = input.view(-1, input_size[-2], input_size[-1])
        if dropout_prob <= 0.0 or input_size[-1] <= 1024:
            return SoftmaxDropoutFast.apply(
                is_training, input, mask, bias, dropout_prob
            ).view(*input_size)
        else:
            return F.dropout(SoftmaxDropoutFast.apply(
                is_training, input, mask, bias, 0.0
            ).view(*input_size), p=dropout_prob, training=is_training)
    else:
        return _softmax_dropout_reference_prepared(
            input, dropout_prob, is_training, mask, bias, inplace
        )
