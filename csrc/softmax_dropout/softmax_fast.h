#pragma once
#include <iostream>
#include <type_traits>
#include <limits>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include "util.h"

template <int N>
using IntegerBits = typename std::conditional<N <= 8, uint8_t,
                                              typename std::conditional<N <= 16, uint16_t,
                                                                        typename std::conditional<N <= 32, uint32_t,
                                                                                                  typename std::conditional<N <= 64, uint64_t, void>::type>::type>::type>::type;

template <int LogElements>
struct SoftmaxParameters
{
    static_assert(LogElements <= 11, "");
    static constexpr int Elements = 1 << LogElements;
    static constexpr int WarpBatch = Elements <= 128 ? 2 : 1;
    static constexpr int WarpIterations = Elements <= 32 ? 1 : Elements / 32;
    using MaskType = IntegerBits<WarpIterations>;
    static constexpr int WarpSize = Elements <= 32 ? Elements : 32;
    static constexpr int MaskStride = WarpSize;
};

inline int log2_ceil(int value)
{
    int log2_value = 0;
    while ((1 << log2_value) < value)
        ++log2_value;
    return log2_value;
}

inline at::ScalarType softmax_mask_dtype(int elements)
{
    if (elements > 1024)
    {
        return torch::kInt64;
    }
    else if (elements > 512)
    {
        return torch::kInt32;
    }
    else if (elements > 256)
    {
        return torch::kInt16;
    }
    return torch::kInt8;
}

inline int softmax_mask_size(int batch_size, int elements)
{
    int log2_elements = log2_ceil(elements);
    int e = 1 << log2_elements;
    int warp_size = e < 32 ? e : 32;
    return batch_size * warp_size;
}

inline int softmax_rng_delta_offset(int elements)
{
    int log2_elements = log2_ceil(elements);
    int e = 1 << log2_elements;
    int warp_iterations = e <= 32 ? 1 : e / 32;
    int warp_batch = e <= 128 ? 2 : 1;
    return warp_iterations * warp_batch;
}


inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                int *num_blocks) {
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    *num_blocks =
        std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
    return cudaSuccess;
}

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

template <template <typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
    typedef cub::BlockReduce<T, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
    if (threadIdx.x == 0) {
        result_broadcast = result;
    }
    __syncthreads();
    return result_broadcast;
}

// modified from https://github.com/Oneflow-Inc/oneflow/blob/5d74efa4d07adfd0acbc8e0074778687f1006b86/oneflow/core/cuda/softmax.cuh#L480-L529
// Copyright 2020 The OneFlow Authors. All rights reserved.
template <typename input_t, typename output_t, typename acc_t, int block_size, bool NeedBias, bool NeedAttnMask>
__global__ void softmax_block_forward(const input_t *input, output_t *output, const input_t *attn_mask, const input_t *bias,
        int64_t rows, int cols, int64_t attn_inner_skip_batch, int64_t bias_batch_size) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto *buf = reinterpret_cast<acc_t *>(shared_buf);
    const int tid = threadIdx.x;
    auto element_count = cols;
    int64_t bias_mod_size = bias_batch_size * cols;
    int64_t attn_mask_div_size = element_count;
    if IF_CONSTEXPR (NeedAttnMask)
    {
        attn_mask_div_size = attn_inner_skip_batch * element_count;
    }
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        acc_t thread_max = -std::numeric_limits<acc_t>::infinity();
        int64_t idx_offset = row * cols;
        const input_t* input_ptr = input + idx_offset;
        output_t* output_ptr = output + idx_offset;
        const input_t* attn_mask_ptr = nullptr;
        if IF_CONSTEXPR (NeedAttnMask){
            attn_mask_ptr = attn_mask + static_cast<int64_t>(idx_offset / attn_mask_div_size) * element_count ;
        }
        const input_t* bias_ptr = nullptr;
        if IF_CONSTEXPR (NeedBias) {
            bias_ptr = bias + idx_offset % bias_mod_size;
        }
        // TODO: enable pack as oneflow 
        for (int col = tid; col < cols; col += block_size) {
            buf[col] = static_cast<acc_t>(input_ptr[col]);
            if IF_CONSTEXPR (NeedAttnMask)
            {
                buf[col] += attn_mask_ptr[col];
            }
            if IF_CONSTEXPR (NeedBias)
            {
                buf[col] += bias_ptr[col];
            }
            thread_max = max(thread_max, buf[col]);
        }

        const acc_t row_max = BlockAllReduce<MaxOp, acc_t, block_size>(thread_max);

        acc_t thread_sum = 0;
        for (int col = tid; col < cols; col += block_size) {
            buf[col] = std::exp(buf[col] - row_max);
            thread_sum += buf[col];
        }

        const acc_t row_sum = BlockAllReduce<SumOp, acc_t, block_size>(thread_sum);
        for (int col = tid; col < cols; col += block_size) {
            output_ptr[col] = static_cast<output_t>(buf[col] / row_sum);
        }
    }
}

template<typename input_t, typename output_t, typename acc_t, int block_size>
__global__ void softmax_block_backward(output_t* store, const input_t* dy, const input_t* y,
                                    const int64_t rows, const int64_t cols) {
  extern __shared__ __align__(sizeof(double)) unsigned char grad_shared_buf[];
  auto* dy_buf = reinterpret_cast<acc_t*>(grad_shared_buf);
  auto* y_buf = reinterpret_cast<input_t*>(dy_buf + cols);
  const int tid = threadIdx.x;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    acc_t thread_sum = 0;
    auto dy_ptr = dy + row * cols;
    auto y_ptr = y + row * cols;
    auto store_ptr = store + row * cols;
    for (int col = tid; col < cols; col += block_size) {
        y_buf[col] = y_ptr[col];
        dy_buf[col] = dy_ptr[col] * (acc_t)y_ptr[col];
    }
    for (int col = tid; col < cols; col += block_size) {
        thread_sum += dy_buf[col];
    }
    const acc_t row_sum = BlockAllReduce<SumOp, acc_t, block_size>(thread_sum);
    for (int col = tid; col < cols; col += block_size) {
        store_ptr[col] = static_cast<output_t>(dy_buf[col] - y_buf[col] * row_sum);
    }
  }
}

template <
    typename input_t, typename output_t, typename acc_t,
    typename Parameters, bool NeedMask, bool NeedBias, bool NeedAttnMask>
__global__ void softmax_warp_forward(input_t *dst, input_t *dst_orig, const output_t *src, const input_t *attn_mask, const input_t *bias,
                                     typename Parameters::MaskType *mask, acc_t p, int64_t batch_size, int64_t attn_inner_skip_batch, int64_t bias_batch_size, int element_count, uint64_t seed, uint64_t rand_offset)
{
    using MaskType = typename Parameters::MaskType;
    curandStatePhilox4_32_10_t state;
    int64_t first_batch = (static_cast<int64_t>(blockDim.y) * static_cast<int64_t>(blockIdx.x) + threadIdx.y) * Parameters::WarpBatch;
    // there might be multiple batches per warp. compute the index within the batch
    int64_t local_idx = threadIdx.x;
    const int64_t thread_offset = first_batch * element_count + local_idx;
    if IF_CONSTEXPR (NeedMask)
    {
        curand_init(seed, thread_offset, rand_offset, &state);
    }

    // batch_size might not be a multiple of Parameters::WarpBatch. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > Parameters::WarpBatch)
        local_batches = Parameters::WarpBatch;

    src += thread_offset;
    dst += thread_offset;
    if IF_CONSTEXPR (NeedMask)
    {
        dst_orig += thread_offset;
        mask += first_batch * Parameters::MaskStride;
    }

    int64_t bias_mod_size = bias_batch_size * element_count;

    int64_t attn_mask_div_size = element_count;
    if IF_CONSTEXPR (NeedAttnMask)
    {
        attn_mask_div_size = attn_inner_skip_batch * element_count;
    }

    // load data from global memory
    input_t elements_input[Parameters::WarpBatch][Parameters::WarpIterations];
#pragma unroll
    for (int i = 0; i < Parameters::WarpBatch; ++i)
    {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        auto src_ptr = src + i * element_count;
#pragma unroll
        for (int it = 0; it < Parameters::WarpIterations; ++it)
        {
            int element_index = local_idx + it * Parameters::WarpSize;
            elements_input[i][it] = -std::numeric_limits<float>::infinity();

            if (element_index < batch_element_count)
            {
                elements_input[i][it] = src_ptr[it * Parameters::WarpSize];
            }
        }
    }

    // convert input_t to acc_t
    acc_t elements[Parameters::WarpBatch][Parameters::WarpIterations];
#pragma unroll
    for (int i = 0; i < Parameters::WarpBatch; ++i)
    {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        int64_t idx_offset = (first_batch + i) * element_count;
        const input_t* attn_mask_ptr = nullptr;
        if IF_CONSTEXPR (NeedAttnMask){
            attn_mask_ptr = attn_mask + static_cast<int64_t>(idx_offset / attn_mask_div_size) * element_count  + local_idx;
        }
        const input_t* bias_ptr = nullptr;
        if IF_CONSTEXPR (NeedBias){
            bias_ptr = bias + idx_offset % bias_mod_size + local_idx;
        }
#pragma unroll
        for (int it = 0; it < Parameters::WarpIterations; ++it)
        {
            elements[i][it] = elements_input[i][it];
            int element_index = local_idx + it * Parameters::WarpSize;
            if (element_index < batch_element_count)
            {
                if IF_CONSTEXPR (NeedAttnMask)
                {
                    elements[i][it] += attn_mask_ptr[it * Parameters::WarpSize];
                }
                if IF_CONSTEXPR (NeedBias)
                {
                    elements[i][it] += bias_ptr[it * Parameters::WarpSize];
                }
            }
        }
    }

    // compute local max_value

    // take the max_value of the first element to avoid one max call
    acc_t max_value[Parameters::WarpBatch];
#pragma unroll
    for (int i = 0; i < Parameters::WarpBatch; ++i)
    {
        max_value[i] = elements[i][0];
    }

#pragma unroll
    for (int it = 1; it < Parameters::WarpIterations; ++it)
    {
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }

// reduction max_value
#pragma unroll
    for (int offset = Parameters::WarpSize / 2; offset > 0; offset /= 2)
    {
        float val[Parameters::WarpBatch];
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            val[i] = SHFL_XOR(max_value[i], offset, Parameters::WarpSize);
        }
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            max_value[i] = max_value[i] > val[i] ? max_value[i] : val[i];
        }
    }

    // compute local sum
    acc_t sum[Parameters::WarpBatch]{0.0f};

#pragma unroll
    for (int i = 0; i < Parameters::WarpBatch; ++i)
    {
#pragma unroll
        for (int it = 0; it < Parameters::WarpIterations; ++it)
        {
            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
            sum[i] += elements[i][it];
        }
    }

// reduction sum
#pragma unroll
    for (int offset = Parameters::WarpSize / 2; offset > 0; offset /= 2)
    {
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            sum[i] += SHFL_XOR(sum[i], offset, Parameters::WarpSize);
        }
    }

    // store result
    if IF_CONSTEXPR (NeedMask)
    {
        const acc_t pinv = 1.0 / p;
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            if (i >= local_batches)
                break;
            MaskType m = 0;
            if IF_CONSTEXPR (Parameters::WarpIterations == 1)
            {
                float rand = curand_uniform(&state);
                m = rand < p;
            }
            else if IF_CONSTEXPR (Parameters::WarpIterations == 2)
            {
                m = curand_uniform(&state) < p;
                m |= (curand_uniform(&state) < p) << 1;
            }
            else
            {
#pragma unroll
                for (int j = 0; j < DIV_CELL(Parameters::WarpIterations, 4); ++j)
                {
                    float4 rand4 = curand_uniform4(&state);
                    m |= (((MaskType)(rand4.x < p)) << (j * 4)) | (((MaskType)(rand4.y < p)) << (j * 4 + 1)) | (((MaskType)(rand4.z < p)) << (j * 4 + 2)) | (((MaskType)(rand4.w < p)) << (j * 4 + 3));
                }
            }
            mask[i * Parameters::MaskStride + local_idx] = m;
            auto dst_ptr = dst + i * element_count;
            auto dst_orig_ptr = dst_orig + i * element_count;
#pragma unroll
            for (int it = 0; it < Parameters::WarpIterations; ++it)
            {
                int element_index = local_idx + it * Parameters::WarpSize;
                if (element_index < element_count)
                {
                    const output_t d = elements[i][it] / sum[i];
                    dst_ptr[it * Parameters::WarpSize] = (acc_t)d * ((acc_t)((m >> it) & 1) * pinv);
                    dst_orig_ptr[it * Parameters::WarpSize] = d;
                }
                else
                {
                    break;
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            auto dst_ptr = dst + i * element_count;
            if (i >= local_batches)
                break;
#pragma unroll
            for (int it = 0; it < Parameters::WarpIterations; ++it)
            {
                int element_index = local_idx + it * Parameters::WarpSize;
                if (element_index < element_count)
                {
                    dst_ptr[it * Parameters::WarpSize] = elements[i][it] / sum[i];
                }
                else
                {
                    break;
                }
            }
        }
    }
}

#define LAUNCH_FORWARD_KERNEL(l)                                                                           \
    softmax_warp_forward<input_t, output_t, acc_t, SoftmaxParameters<l>, NeedMask, NeedBias, NeedAttnMask> \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(                                        \
            dst, dst_orig, src, attn_mask, bias, (typename SoftmaxParameters<l>::MaskType *)mask, p,       \
            batch_count, attn_inner_skip_batch, bias_batch_count, softmax_elements, seed, offset);         \
    return true;

template <typename input_t, typename output_t, typename acc_t, bool NeedMask, bool NeedBias, bool NeedAttnMask>
bool dispatch_softmax_forward(output_t *dst, output_t *dst_orig, const input_t *src, const input_t *attn_mask, const input_t *bias, void *mask, acc_t p,
                              int softmax_elements, int64_t batch_count, int64_t attn_inner_skip_batch, int64_t bias_batch_count, uint64_t seed, uint64_t offset)
{
    if (softmax_elements == 0)
    {
        return false;
    }
    else
    {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the Parameters::WarpSize constexpr value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        // This value must match the Parameters::WarpBatch constexpr value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements)
        {
            case 0:
                LAUNCH_FORWARD_KERNEL(0)
            case 1:
                LAUNCH_FORWARD_KERNEL(1)
            case 2:
                LAUNCH_FORWARD_KERNEL(2)
            case 3:
                LAUNCH_FORWARD_KERNEL(3)
            case 4:
                LAUNCH_FORWARD_KERNEL(4)
            case 5:
                LAUNCH_FORWARD_KERNEL(5)
            case 6:
                LAUNCH_FORWARD_KERNEL(6)
            case 7:
                LAUNCH_FORWARD_KERNEL(7)
            case 8:
                LAUNCH_FORWARD_KERNEL(8)
            case 9:
                LAUNCH_FORWARD_KERNEL(9)
            case 10:
                LAUNCH_FORWARD_KERNEL(10)
            default:
            {
                int grid_dim;
                constexpr int block_size = 128;
                constexpr int waves = 32;
                auto cols = softmax_elements;
                auto rows = batch_count;
                GetNumBlocks(block_size, rows, waves, &grid_dim);
                dim3 block(block_size);
                const size_t smem = cols * sizeof(acc_t);
                softmax_block_forward<input_t, output_t, acc_t,  block_size, NeedBias, NeedAttnMask><<<grid_dim, block, smem>>>(
                    src, dst, attn_mask, bias, rows, cols, attn_inner_skip_batch, bias_batch_count);
                return true;
            }
        }
    }
    return false;
}

template <
    typename input_t, typename output_t, typename acc_t, typename Parameters,
    bool IsLogSoftmax, bool NeedMask>
__global__ void softmax_warp_backward(output_t *gradInput, const input_t *grad, const input_t *output,
                                      const typename Parameters::MaskType *mask, acc_t p, int64_t batch_size, int element_count)
{
    using MaskType = typename Parameters::MaskType;
    int64_t first_batch = (static_cast<int64_t>(blockDim.y) * static_cast<int64_t>(blockIdx.x) + threadIdx.y) * Parameters::WarpBatch;

    // batch_size might not be a multiple of Parameters::WarpBatch. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > Parameters::WarpBatch)
        local_batches = Parameters::WarpBatch;

    // there might be multiple batches per warp. compute the index within the batch
    int64_t local_idx = threadIdx.x;

    // the first element to process by the current thread
    int64_t thread_offset = first_batch * element_count + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;
    if IF_CONSTEXPR (NeedMask)
    {
        mask += first_batch * Parameters::MaskStride;
    }

    // The nested loops over Parameters::WarpBatch and then Parameters::WarpIterations can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t grad_reg[Parameters::WarpBatch][Parameters::WarpIterations];
    input_t output_reg[Parameters::WarpBatch][Parameters::WarpIterations];
    if IF_CONSTEXPR (NeedMask)
    {
        MaskType mask_reg[Parameters::WarpBatch];
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            if (i >= local_batches)
                break;
            mask_reg[i] = mask[i * Parameters::MaskStride + local_idx];
        }

        const acc_t pinv = 1.0 / p;

#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            int batch_element_count = (i >= local_batches) ? 0 : element_count;
            MaskType m = mask_reg[i];
            auto output_ptr = output + i * element_count;
            auto grad_ptr = grad + i * element_count;
#pragma unroll
            for (int it = 0; it < Parameters::WarpIterations; ++it)
            {
                int element_index = local_idx + it * Parameters::WarpSize;
                if (element_index < batch_element_count)
                {
                    grad_reg[i][it] =
                        (acc_t)((m >> it) & 1) *
                                  (acc_t)grad_ptr[it * Parameters::WarpSize] *
                                  pinv *
                        output_ptr[it * Parameters::WarpSize];
                    output_reg[i][it] = output_ptr[it * Parameters::WarpSize];
                }
                else
                {
                    grad_reg[i][it] = acc_t(0);
                    output_reg[i][it] = input_t(0);
                }
            }
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            int batch_element_count = (i >= local_batches) ? 0 : element_count;
            auto output_ptr = output + i * element_count;
            auto grad_ptr = grad + i * element_count;
#pragma unroll
            for (int it = 0; it < Parameters::WarpIterations; ++it)
            {
                int element_index = local_idx + it * Parameters::WarpSize;
                if (element_index < batch_element_count)
                {
                    output_reg[i][it] = output_ptr[it * Parameters::WarpSize];
                    grad_reg[i][it] = grad_ptr[it * Parameters::WarpSize] *
                                      (acc_t)output_ptr[it * Parameters::WarpSize];
                }
                else
                {
                    grad_reg[i][it] = acc_t(0);
                    output_reg[i][it] = output_t(0);
                }
            }
        }
    }

    acc_t sum[Parameters::WarpBatch];
#pragma unroll
    for (int i = 0; i < Parameters::WarpBatch; ++i)
    {
        sum[i] = grad_reg[i][0];
#pragma unroll
        for (int it = 1; it < Parameters::WarpIterations; ++it)
        {
            sum[i] += grad_reg[i][it];
        }
    }

#pragma unroll
    for (int offset = Parameters::WarpSize / 2; offset > 0; offset /= 2)
    {
#pragma unroll
        for (int i = 0; i < Parameters::WarpBatch; ++i)
        {
            sum[i] += SHFL_XOR(sum[i], offset, Parameters::WarpSize);
        }
    }

// store result
#pragma unroll
    for (int i = 0; i < Parameters::WarpBatch; ++i)
    {
        if (i >= local_batches)
            break;
        auto gradInput_ptr = gradInput + i * element_count;
#pragma unroll
        for (int it = 0; it < Parameters::WarpIterations; ++it)
        {
            int element_index = local_idx + it * Parameters::WarpSize;
            if (element_index < element_count)
            {
                // compute gradients
                if IF_CONSTEXPR (IsLogSoftmax)
                {
                    gradInput_ptr[it * Parameters::WarpSize] =
                        (grad_reg[i][it] - std::exp((acc_t)output_reg[i][it]) * sum[i]);
                }
                else
                {
                    gradInput_ptr[it * Parameters::WarpSize] =
                        (grad_reg[i][it] - output_reg[i][it] * sum[i]);
                }
            }
        }
    }
}

#define LAUNCH_BACKWARD_KERNEL(l)                                                                 \
    softmax_warp_backward<input_t, output_t, acc_t, SoftmaxParameters<l>, IsLogSoftmax, NeedMask> \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(                               \
            grad_input, grad, output, (const typename SoftmaxParameters<l>::MaskType *)mask, p,   \
            batch_count, softmax_elements);                                                       \
    break;

template <typename input_t, typename output_t, typename acc_t, bool IsLogSoftmax, bool NeedMask>
void dispatch_softmax_backward(output_t *grad_input, const input_t *grad, const input_t *output,
                               const void *mask, acc_t p, int softmax_elements, int64_t batch_count)
{
    if (softmax_elements == 0)
    {
        return;
    }
    else
    {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < 32) ? next_power_of_two : 32;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements)
        {
            case 0:
                LAUNCH_BACKWARD_KERNEL(0)
            case 1:
                LAUNCH_BACKWARD_KERNEL(1)
            case 2:
                LAUNCH_BACKWARD_KERNEL(2)
            case 3:
                LAUNCH_BACKWARD_KERNEL(3)
            case 4:
                LAUNCH_BACKWARD_KERNEL(4)
            case 5:
                LAUNCH_BACKWARD_KERNEL(5)
            case 6:
                LAUNCH_BACKWARD_KERNEL(6)
            case 7:
                LAUNCH_BACKWARD_KERNEL(7)
            case 8:
                LAUNCH_BACKWARD_KERNEL(8)
            case 9:
                LAUNCH_BACKWARD_KERNEL(9)
            case 10:
                LAUNCH_BACKWARD_KERNEL(10)
            default:
            {
                int grid_dim;
                constexpr int block_size = 128;
                constexpr int waves = 32;
                auto cols = softmax_elements;
                auto rows = batch_count;
                GetNumBlocks(block_size, rows, waves, &grid_dim);
                dim3 block(block_size);
                const size_t smem = cols * sizeof(acc_t) + cols * sizeof(input_t) ;
                softmax_block_backward<input_t, output_t, acc_t,  block_size><<<grid_dim, block, smem>>>(
                    grad_input, grad, output, rows, cols);
            }
        }
    }
}
