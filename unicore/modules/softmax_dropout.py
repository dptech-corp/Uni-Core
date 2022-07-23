# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unicore_fused_softmax_dropout
import torch.nn.functional as F

class SoftmaxDropoutFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, is_training, inputs, bias, dropout_prob, num_groups, num_heads):
        # don't use ctx.save_for_backward to save dropout_prob
        # allocating space for a tensor is time-consuming
        dropout_results, dropout_mask, softmax_results = unicore_fused_softmax_dropout.forward(is_training,
            inputs, bias, dropout_prob, num_groups, num_heads, None)
        if is_training:
            ctx.dropout_prob = dropout_prob
            ctx.save_for_backward(softmax_results, dropout_mask)
            ctx.has_bias = bias is not None
        return dropout_results
    @staticmethod
    def backward(ctx, grad_output):
        softmax_results, dropout_mask = ctx.saved_tensors
        dropout_prob = ctx.dropout_prob
        grad_output = grad_output.contiguous()
        grad_input = unicore_fused_softmax_dropout.backward(grad_output, softmax_results,
            dropout_mask, dropout_prob)
        if ctx.has_bias:
            grad_bias = grad_input.sum(dim=1, keepdims=True)
        else:
            grad_bias = None
        return None, grad_input, grad_bias, None, None, None

def softmax_dropout(input, dropout_prob, bias=None, is_training=True):
    input = input.contiguous()
    input_size = input.size()
    num_heads = input_size[-3]
    num_groups = input_size[-4]
    input = input.view(-1, input_size[-2], input_size[-1])
    if bias is not None:
        bias = bias.contiguous().view(-1, input_size[-2], input_size[-1])
    if input.is_cuda and input.shape[-1] <= 2048:
        return SoftmaxDropoutFast.apply(is_training, input, bias, dropout_prob, num_heads, num_groups).view(*input_size)
    else:
        return F.dropout(F.softmax(input + bias, dim=-1), p=dropout_prob, training=is_training).view(*input_size)


a = torch.randn(1, 64, 8, 128, 256).cuda()
bias = torch.randn(1, 1, 8, 128, 256).cuda()

t1 = torch.softmax(a + bias, dim=-1)
t2 = softmax_dropout(a, 0, bias=None, is_training=True)

diff = (t1 - t2).abs().max()