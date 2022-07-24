#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>

#include "type_shim.h"
#include "softmax_fast.h"

std::vector<c10::optional<torch::Tensor>> fwd_cuda(
    bool is_training,
    torch::Tensor &input,
    const c10::optional<torch::Tensor> &attn_mask,
    const c10::optional<torch::Tensor> &bias,
    float dropout_prob,
    c10::optional<at::Generator> gen_)
{
    const int64_t attn_batches = input.size(0);
    const int q_seq_len = input.size(1);
    const int k_seq_len = input.size(2);
    void *bias_ptr = nullptr;
    int64_t bias_batches = 0;
    if (bias)
    {
        bias_ptr = reinterpret_cast<void *>(bias->data_ptr());
        bias_batches = bias->size(0);
    }
    void *attn_mask_prt = nullptr;
    int64_t mask_inner_skip = 0;
    if (attn_mask)
    {
        attn_mask_prt = reinterpret_cast<void *>(attn_mask->data_ptr());
        mask_inner_skip = static_cast<int64_t>(attn_batches / attn_mask->size(0) * q_seq_len / attn_mask->size(1));
    }
    auto act_options = input.options().requires_grad(false);
    auto mask_options = act_options.dtype(softmax_mask_dtype(k_seq_len));

    // Softmax Intermediate Result Ptr (used by Matmul1 -> Softmax)
    void *input_ptr = reinterpret_cast<void *>(input.data_ptr());
    void *softmax_results_ptr = reinterpret_cast<void *>(input.data_ptr());

    // Padded Softmax
    bool softmax_success = false;
    auto scalar_type = input.scalar_type();
    if (is_training && dropout_prob > 0.0f)
    {
        torch::Tensor dropout_results = torch::empty({static_cast<int64_t>(attn_batches), q_seq_len, k_seq_len}, act_options);
        torch::Tensor dropout_mask = torch::empty(
            {softmax_mask_size(static_cast<int64_t>(attn_batches * q_seq_len), k_seq_len)}, mask_options);
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        std::pair<uint64_t, uint64_t> rng_engine_inputs;
        {
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_engine_inputs(softmax_rng_delta_offset(k_seq_len));
        }
        uint64_t seed = std::get<0>(rng_engine_inputs);
        uint64_t offset = std::get<1>(rng_engine_inputs);
        if (bias)
        {
            if (attn_mask)
            {
                DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_forward",
                                                 softmax_success = dispatch_softmax_forward<scalar_t_0, scalar_t_0, float, true, true, true>(
                                                     reinterpret_cast<scalar_t_0 *>(dropout_results.data_ptr()),
                                                     reinterpret_cast<scalar_t_0 *>(softmax_results_ptr),
                                                     reinterpret_cast<const scalar_t_0 *>(input_ptr),
                                                     reinterpret_cast<const scalar_t_0 *>(attn_mask_prt),
                                                     reinterpret_cast<const scalar_t_0 *>(bias_ptr),
                                                     reinterpret_cast<void *>(dropout_mask.data_ptr()),
                                                     1.0f - dropout_prob,
                                                     k_seq_len,
                                                     attn_batches * q_seq_len,
                                                     mask_inner_skip,
                                                     bias_batches * q_seq_len,
                                                     seed, offset);)
            }
            else
            {
                DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_forward",
                                                 softmax_success = dispatch_softmax_forward<scalar_t_0, scalar_t_0, float, true, true, false>(
                                                     reinterpret_cast<scalar_t_0 *>(dropout_results.data_ptr()),
                                                     reinterpret_cast<scalar_t_0 *>(softmax_results_ptr),
                                                     reinterpret_cast<const scalar_t_0 *>(input_ptr),
                                                     nullptr,
                                                     reinterpret_cast<const scalar_t_0 *>(bias_ptr),
                                                     reinterpret_cast<void *>(dropout_mask.data_ptr()),
                                                     1.0f - dropout_prob,
                                                     k_seq_len,
                                                     attn_batches * q_seq_len,
                                                     mask_inner_skip,
                                                     bias_batches * q_seq_len,
                                                     seed, offset);)
            }
        }
        else
        {
            if (attn_mask)
            {
                DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_forward",
                                                 softmax_success = dispatch_softmax_forward<scalar_t_0, scalar_t_0, float, true, false, true>(
                                                     reinterpret_cast<scalar_t_0 *>(dropout_results.data_ptr()),
                                                     reinterpret_cast<scalar_t_0 *>(softmax_results_ptr),
                                                     reinterpret_cast<const scalar_t_0 *>(input_ptr),
                                                     reinterpret_cast<const scalar_t_0 *>(attn_mask_prt),
                                                     nullptr,
                                                     reinterpret_cast<void *>(dropout_mask.data_ptr()),
                                                     1.0f - dropout_prob,
                                                     k_seq_len,
                                                     attn_batches * q_seq_len,
                                                     mask_inner_skip,
                                                     bias_batches * q_seq_len,
                                                     seed, offset);)
            }
            else
            {
                DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_forward",
                                                 softmax_success = dispatch_softmax_forward<scalar_t_0, scalar_t_0, float, true, false, false>(
                                                     reinterpret_cast<scalar_t_0 *>(dropout_results.data_ptr()),
                                                     reinterpret_cast<scalar_t_0 *>(softmax_results_ptr),
                                                     reinterpret_cast<const scalar_t_0 *>(input_ptr),
                                                     nullptr,
                                                     nullptr,
                                                     reinterpret_cast<void *>(dropout_mask.data_ptr()),
                                                     1.0f - dropout_prob,
                                                     k_seq_len,
                                                     attn_batches * q_seq_len,
                                                     mask_inner_skip,
                                                     bias_batches * q_seq_len,
                                                     seed, offset);)
            }
        }

        if (softmax_success)
        {
            return {dropout_results, dropout_mask, input};
        }
        else
        {
            return {c10::optional<torch::Tensor>(), c10::optional<torch::Tensor>(), c10::optional<torch::Tensor>()};
        }
    }
    else
    {
        if (bias)
        {
            if (attn_mask)
            {
                DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_forward",
                                                 softmax_success = dispatch_softmax_forward<scalar_t_0, scalar_t_0, float, false, true, true>(
                                                     reinterpret_cast<scalar_t_0 *>(softmax_results_ptr),
                                                     nullptr,
                                                     reinterpret_cast<const scalar_t_0 *>(input_ptr),
                                                     reinterpret_cast<const scalar_t_0 *>(attn_mask_prt),
                                                     reinterpret_cast<const scalar_t_0 *>(bias_ptr),
                                                     nullptr,
                                                     1.0,
                                                     k_seq_len,
                                                     attn_batches * q_seq_len,
                                                     mask_inner_skip,
                                                     bias_batches * q_seq_len,
                                                     0, 0);)
            }
            else
            {
                DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_forward",
                                                 softmax_success = dispatch_softmax_forward<scalar_t_0, scalar_t_0, float, false, true, false>(
                                                     reinterpret_cast<scalar_t_0 *>(softmax_results_ptr),
                                                     nullptr,
                                                     reinterpret_cast<const scalar_t_0 *>(input_ptr),
                                                     nullptr,
                                                     reinterpret_cast<const scalar_t_0 *>(bias_ptr),
                                                     nullptr,
                                                     1.0,
                                                     k_seq_len,
                                                     attn_batches * q_seq_len,
                                                     mask_inner_skip,
                                                     bias_batches * q_seq_len,
                                                     0, 0);)
            }
        }
        else
        {
            if (attn_mask)
            {
                DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_forward",
                                                 softmax_success = dispatch_softmax_forward<scalar_t_0, scalar_t_0, float, false, false, true>(
                                                     reinterpret_cast<scalar_t_0 *>(softmax_results_ptr),
                                                     nullptr,
                                                     reinterpret_cast<const scalar_t_0 *>(input_ptr),
                                                     reinterpret_cast<const scalar_t_0 *>(attn_mask_prt),
                                                     nullptr,
                                                     nullptr,
                                                     1.0,
                                                     k_seq_len,
                                                     attn_batches * q_seq_len,
                                                     mask_inner_skip,
                                                     bias_batches * q_seq_len,
                                                     0, 0);)
            }
            else
            {
                DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_forward",
                                                 softmax_success = dispatch_softmax_forward<scalar_t_0, scalar_t_0, float, false, false, false>(
                                                     reinterpret_cast<scalar_t_0 *>(softmax_results_ptr),
                                                     nullptr,
                                                     reinterpret_cast<const scalar_t_0 *>(input_ptr),
                                                     nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     1.0,
                                                     k_seq_len,
                                                     attn_batches * q_seq_len,
                                                     mask_inner_skip,
                                                     bias_batches * q_seq_len,
                                                     0, 0);)
            }
        }
        if (softmax_success)
        {
            return {input, c10::optional<torch::Tensor>(), input};
        }
        else
        {
            return {c10::optional<torch::Tensor>(), c10::optional<torch::Tensor>(), c10::optional<torch::Tensor>()};
        }
    }
}

torch::Tensor bwd_cuda(
    torch::Tensor &output_grads,
    const torch::Tensor &softmax_results,
    const c10::optional<torch::Tensor> &dropout_mask,
    float dropout_prob)
{
    const int64_t attn_batches = output_grads.size(0);
    const int q_seq_len = output_grads.size(1);
    const int k_seq_len = output_grads.size(2);

    auto scalar_type = output_grads.scalar_type();

    if (dropout_mask)
    {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_backward",
                                         dispatch_softmax_backward<scalar_t_0, scalar_t_0, float, false, true>(
                                             reinterpret_cast<scalar_t_0 *>(output_grads.data_ptr()),
                                             reinterpret_cast<const scalar_t_0 *>(output_grads.data_ptr()),
                                             reinterpret_cast<const scalar_t_0 *>(softmax_results.data_ptr()),
                                             reinterpret_cast<const void *>(dropout_mask->data_ptr()),
                                             1.0f - dropout_prob,
                                             k_seq_len,
                                             attn_batches * q_seq_len);)
    }
    else
    {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(scalar_type, 0, "softmax_backward",
                                         dispatch_softmax_backward<scalar_t_0, scalar_t_0, float, false, false>(
                                             reinterpret_cast<scalar_t_0 *>(output_grads.data_ptr()),
                                             reinterpret_cast<scalar_t_0 *>(output_grads.data_ptr()),
                                             reinterpret_cast<const scalar_t_0 *>(softmax_results.data_ptr()),
                                             nullptr,
                                             1.0f,
                                             k_seq_len,
                                             attn_batches * q_seq_len);)
    }
    // backward pass is completely in-place
    return output_grads;
}
