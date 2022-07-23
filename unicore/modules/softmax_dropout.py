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


def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None):
    input = input.contiguous()
    input_size = input.size()
    if mask is not None:
        assert mask.dtype == input.dtype
        assert (
            mask.shape[-2] == 1 or mask.shape[-2] == input_size[-2]
        ), "row dim should be 1 or the same as input."
        assert (
            mask.shape[-3] == 1 or mask.shape[-3] == input_size[-3]
        ), "head dim should be 1 or the same as input."
        mask = mask.contiguous().view(-1, mask.shape[-2], mask.shape[-1])
    input = input.view(-1, input_size[-2], input_size[-1])
    if bias is not None:
        assert bias.dtype == input.dtype
        bias = bias.contiguous().view(-1, input_size[-2], input_size[-1])
        assert (
            input.shape[0] % bias.shape[0] == 0
        ), "bias batch size must be divisible by input batch size"
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
