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

#include "softmax_fast.h"

std::vector<c10::optional<torch::Tensor>> fwd_cuda(
    bool is_training,
    const torch::Tensor &input, 
    float dropout_prob,
    c10::optional<at::Generator> gen_
) {
    const int attn_batches   = input.size(0);
    const int q_seq_len      = input.size(1);
    const int k_seq_len      = input.size(2);

    auto act_options  = input.options().requires_grad(false);
    auto mask_options = act_options.dtype(softmax_mask_dtype(k_seq_len));

    torch::Tensor softmax_results = torch::empty({attn_batches, q_seq_len, k_seq_len}, act_options);

    // Softmax Intermediate Result Ptr (used by Matmul1 -> Softmax)
    void *input_ptr = reinterpret_cast<void *>(input.data_ptr());
    void *softmax_results_ptr = reinterpret_cast<void *>(softmax_results.data_ptr());

    // Padded Softmax
    bool softmax_success = false;
    auto scalar_type = input.scalar_type();
    if (is_training && dropout_prob > 0.0f) {
        torch::Tensor dropout_results   = torch::empty({attn_batches, q_seq_len, k_seq_len}, act_options);
        torch::Tensor dropout_mask      = torch::empty(
            {softmax_mask_size(attn_batches * q_seq_len, k_seq_len)}, mask_options
        );
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
        if (scalar_type == at::ScalarType::BFloat16){
            softmax_success = dispatch_softmax_forward<nv_bfloat16, nv_bfloat16, float, true>(
                reinterpret_cast<nv_bfloat16 *>(dropout_results.data_ptr()),
                reinterpret_cast<nv_bfloat16 *>(softmax_results_ptr),
                reinterpret_cast<const nv_bfloat16 *>(input_ptr),
                reinterpret_cast<void *>(dropout_mask.data_ptr()),
                1.0f - dropout_prob,
                k_seq_len,
                attn_batches*q_seq_len, seed, offset);
        } else if (scalar_type == at::ScalarType::Half){
            softmax_success = dispatch_softmax_forward<half, half, float, true>(
                reinterpret_cast<half *>(dropout_results.data_ptr()),
                reinterpret_cast<half *>(softmax_results_ptr),
                reinterpret_cast<const half *>(input_ptr),
                reinterpret_cast<void *>(dropout_mask.data_ptr()),
                1.0f - dropout_prob,
                k_seq_len,
                attn_batches*q_seq_len, seed, offset);
        } else if (scalar_type == at::ScalarType::Float){
            softmax_success = dispatch_softmax_forward<float, float, float, true>(
                reinterpret_cast<float *>(dropout_results.data_ptr()),
                reinterpret_cast<float *>(softmax_results_ptr),
                reinterpret_cast<const float *>(input_ptr),
                reinterpret_cast<void *>(dropout_mask.data_ptr()),
                1.0f - dropout_prob,
                k_seq_len,
                attn_batches*q_seq_len, seed, offset);
        } else {
            softmax_success = false;
        }
        if (softmax_success) {
            return {dropout_results, dropout_mask, softmax_results};
        } else {
            return {c10::optional<torch::Tensor>(), c10::optional<torch::Tensor>(), c10::optional<torch::Tensor>()};
        }
    } else {
        if (scalar_type == at::ScalarType::BFloat16){
            softmax_success = dispatch_softmax_forward<nv_bfloat16, nv_bfloat16, float, false>(
                reinterpret_cast<nv_bfloat16 *>(softmax_results_ptr),
                nullptr,
                reinterpret_cast<const nv_bfloat16 *>(input_ptr),
                nullptr,
                1.0,
                k_seq_len,
                attn_batches*q_seq_len, 0, 0);
        } else if (scalar_type == at::ScalarType::Half){
            softmax_success = dispatch_softmax_forward<half, half, float, false>(
                reinterpret_cast<half *>(softmax_results_ptr),
                nullptr,
                reinterpret_cast<const half *>(input_ptr),
                nullptr,
                1.0,
                k_seq_len,
                attn_batches*q_seq_len, 0, 0);
        } else if (scalar_type == at::ScalarType::Float){
            softmax_success = dispatch_softmax_forward<float, float, float, false>(
                reinterpret_cast<float *>(softmax_results_ptr),
                nullptr,
                reinterpret_cast<const float *>(input_ptr),
                nullptr,
                1.0,
                k_seq_len,
                attn_batches*q_seq_len, 0, 0);
        } else {
            softmax_success = false;
        }
        if (softmax_success) {
            return {softmax_results, c10::optional<torch::Tensor>(), softmax_results};
        } else {
            return {c10::optional<torch::Tensor>(), c10::optional<torch::Tensor>(), c10::optional<torch::Tensor>()};
        }
    }
}

torch::Tensor bwd_cuda(
    torch::Tensor &output_grads, 
    const torch::Tensor &softmax_results, 
    const c10::optional<torch::Tensor> &dropout_mask,
    float dropout_prob
) 
{
    const int attn_batches   = output_grads.size(0);
    const int q_seq_len      = output_grads.size(1);
    const int k_seq_len      = output_grads.size(2);

    auto scalar_type = output_grads.scalar_type();

    // Apply Dropout Mask and Scale by Dropout Probability 
    // Softmax Grad
    if (dropout_mask) {
        if (scalar_type == at::ScalarType::BFloat16){
            dispatch_softmax_backward<nv_bfloat16, nv_bfloat16, float, false, true>(
                reinterpret_cast<nv_bfloat16 *>(output_grads.data_ptr()), 
                reinterpret_cast<const nv_bfloat16 *>(output_grads.data_ptr()), 
                reinterpret_cast<const nv_bfloat16 *>(softmax_results.data_ptr()),
                reinterpret_cast<const void *>(dropout_mask->data_ptr()),
                1.0f - dropout_prob,
                k_seq_len,
                attn_batches*q_seq_len);
        } else if (scalar_type == at::ScalarType::Half){
            dispatch_softmax_backward<half, half, float, false, true>(
                reinterpret_cast<half *>(output_grads.data_ptr()), 
                reinterpret_cast<const half *>(output_grads.data_ptr()), 
                reinterpret_cast<const half *>(softmax_results.data_ptr()),
                reinterpret_cast<const void *>(dropout_mask->data_ptr()),
                1.0f - dropout_prob,
                k_seq_len,
                attn_batches*q_seq_len);
        } else if (scalar_type == at::ScalarType::Float){
            dispatch_softmax_backward<float, float, float, false, true>(
                reinterpret_cast<float *>(output_grads.data_ptr()), 
                reinterpret_cast<const float *>(output_grads.data_ptr()), 
                reinterpret_cast<const float *>(softmax_results.data_ptr()),
                reinterpret_cast<const void *>(dropout_mask->data_ptr()),
                1.0f - dropout_prob,
                k_seq_len,
                attn_batches*q_seq_len);
        }
    } else {
        if (scalar_type == at::ScalarType::BFloat16){
            dispatch_softmax_backward<nv_bfloat16, nv_bfloat16, float, false, false>(
                reinterpret_cast<nv_bfloat16 *>(output_grads.data_ptr()), 
                reinterpret_cast<nv_bfloat16 *>(output_grads.data_ptr()), 
                reinterpret_cast<const nv_bfloat16 *>(softmax_results.data_ptr()),
                nullptr,
                1.0f,
                k_seq_len,
                attn_batches*q_seq_len);
        } else if (scalar_type == at::ScalarType::Half){
            dispatch_softmax_backward<half, half, float, false, false>(
                reinterpret_cast<half *>(output_grads.data_ptr()), 
                reinterpret_cast<half *>(output_grads.data_ptr()), 
                reinterpret_cast<const half *>(softmax_results.data_ptr()),
                nullptr,
                1.0f,
                k_seq_len,
                attn_batches*q_seq_len);
        } else if (scalar_type == at::ScalarType::Float){
            dispatch_softmax_backward<float, float, float, false, false>(
                reinterpret_cast<float *>(output_grads.data_ptr()), 
                reinterpret_cast<float *>(output_grads.data_ptr()), 
                reinterpret_cast<const float *>(softmax_results.data_ptr()),
                nullptr,
                1.0f,
                k_seq_len,
                attn_batches*q_seq_len);
        }
    }

    //backward pass is completely in-place
    return output_grads;
}
