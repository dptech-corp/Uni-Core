# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unicore_fused_softmax_dropout
import torch.nn.functional as F


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
    assert mask.dtype == input.dtype, "mask and input must have the same dtype"
    assert len(mask.shape) == len(input.shape), "wrong length of mask.shape"
    assert (
        mask.shape[-3] == 1 or mask.shape[-3] == input.shape[-3]
    ), "mask.shape[-3] must be 1 or input.shape[-3]"
    if mask.shape[-3] == 1:
        assert mask.shape[-2] == 1, "when mask.shape[-3] == 1, mask.shape[-2] must be 1"
    else:
        assert (
            mask.shape[-2] == 1 or mask.shape[-2] == input.shape[-2]
        ), "mask.shape[-2] must be 1 or input.shape[-2]"


def _check_bias(bias, input):
    assert bias.dtype == input.dtype, "bias and input must have the same dtype"
    assert len(bias.shape) == len(input.shape), "wrong length of bias.shape"
    assert bias.shape[-1] == input.shape[-1], "bias.shape[-1] must be input.shape[-1]"
    assert bias.shape[-2] == input.shape[-2], "bias.shape[-2] must be input.shape[-2]"
    len_shape = len(input.shape)
    if len_shape > 3:
        # head dim should be the same
        assert (
            bias.shape[-3] == input.shape[-3]
        ), "bias.shape[-3] must be input.shape[-3]"
        offset = 3
    else:
        offset = 2
    prev_non_one = True
    for i in range(len_shape - offset - 1, -1, -1):
        if prev_non_one:
            assert (
                bias.shape[i] == input.shape[i] or bias.shape[i] == 1
            ), "bias.shape[{}] must be input.shape[{}] or 1".format(i, i)
        else:
            assert bias.shape[i] == 1, "bias.shape[{}] must be 1".format(i)
        prev_non_one = bias.shape[i] != 1


def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None):
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
    input = input.contiguous()
    input_size = input.size()
    if mask is not None:
        _check_mask(mask, input)
        mask = mask.contiguous().view(-1, mask.shape[-2], mask.shape[-1])
    if bias is not None:
        _check_bias(bias, input)
        bias = bias.contiguous().view(-1, input_size[-2], input_size[-1])
    input = input.view(-1, input_size[-2], input_size[-1])
    if input.is_cuda and input.shape[-1] <= 2048:
        return SoftmaxDropoutFast.apply(
            is_training, input, mask, bias, dropout_prob
        ).view(*input_size)
    else:
        if mask is None:
            input += mask
        if bias is not None:
            input += bias
        return F.dropout(
            F.softmax(input, dim=-1), p=dropout_prob, training=is_training
        ).view(*input_size)
