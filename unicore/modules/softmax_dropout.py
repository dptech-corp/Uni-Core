# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import unicore_fused_softmax_dropout
import torch.nn.functional as F

class SoftmaxDropoutFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, is_training, inputs, dropout_prob):
        # don't use ctx.save_for_backward to save dropout_prob
        # allocating space for a tensor is time-consuming
        dropout_results, dropout_mask, softmax_results = unicore_fused_softmax_dropout.forward(is_training,
            inputs, dropout_prob, None)
        if is_training:
            ctx.dropout_prob = dropout_prob
            ctx.save_for_backward(softmax_results, dropout_mask)
        return dropout_results

    @staticmethod
    def backward(ctx, grad_output):
        softmax_results, dropout_mask = ctx.saved_tensors
        dropout_prob = ctx.dropout_prob
        grad_output = grad_output.contiguous()
        grad_input = unicore_fused_softmax_dropout.backward(grad_output, softmax_results,
            dropout_mask, dropout_prob)
        return None, grad_input, None

def softmax_dropout(input, dropout_prob, is_training=True):
    input = input.contiguous()
    input_size = input.size()
    input = input.view(-1, input_size[-2], input_size[-1])
    if input.is_cuda and input.shape[-1] <= 2048:
        return SoftmaxDropoutFast.apply(is_training, input, dropout_prob).view(*input_size)
    else:
        return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training).view(*input_size)
