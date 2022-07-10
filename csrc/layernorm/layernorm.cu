#include <iostream>
#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "util.h"

template <int Dim_, int VecSize_, int BatchesPerBlock_, int WarpsForOneBatchPerBlock_>
struct LNParameters {
    static constexpr int Dim = Dim_;
    static constexpr int VecSize = VecSize_;
    static constexpr int WarpSize = 32;
    static constexpr int BatchesPerBlock = BatchesPerBlock_;
    static constexpr int WarpStride = WarpSize * VecSize;
    static constexpr int WarpsForOneBatchPerBlock = WarpsForOneBatchPerBlock_;
    static constexpr int Iterations = Dim / WarpStride / WarpsForOneBatchPerBlock;
    static constexpr int BatchStride = Dim / WarpsForOneBatchPerBlock;
    static constexpr int ThreadsPerBlock = BatchesPerBlock * WarpSize * WarpsForOneBatchPerBlock;
    static_assert(Dim == WarpsForOneBatchPerBlock * WarpStride * Iterations, "");
    static_assert(Dim == BatchStride * WarpsForOneBatchPerBlock, "");
};

template <typename IndexType, typename input_t, typename output_t, typename acc_t, typename Parameters>
__global__ void layernorm_forward(output_t *dst, const input_t *src, const input_t *gamma, const input_t *beta,
    acc_t *mean, acc_t *invvar, IndexType bsz, acc_t epsilon) {
    static_assert(Parameters::WarpsForOneBatchPerBlock == 1, "");
    IndexType batch = blockIdx.x * Parameters::BatchesPerBlock + threadIdx.y;
    if (batch < bsz) {
        src += batch * Parameters::Dim + threadIdx.x * Parameters::VecSize;
        dst += batch * Parameters::Dim + threadIdx.x * Parameters::VecSize;
        gamma += threadIdx.x * Parameters::VecSize;
        beta += threadIdx.x * Parameters::VecSize;
        using VecInType = VecType<input_t, Parameters::VecSize>;
        VecInType elements[Parameters::Iterations];
        VecInType gamma_reg[Parameters::Iterations];
        VecInType beta_reg[Parameters::Iterations];
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations; ++i) {
            elements[i] = *(VecInType *)(src + i * Parameters::WarpStride);
            gamma_reg[i] = *(VecInType *)(gamma + i * Parameters::WarpStride);
            beta_reg[i] = *(VecInType *)(beta + i * Parameters::WarpStride);
        }
        input_t *elements_l = (input_t *)elements;
        input_t *gamma_l = (input_t *)gamma_reg;
        input_t *beta_l = (input_t *)beta_reg;
        
        acc_t sum = 0.0;
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations * Parameters::VecSize; ++i) {
            sum += (acc_t)elements_l[i];
        }
        #pragma unroll
        for (int offset = Parameters::WarpSize / 2; offset > 0; offset /= 2) {
            sum += SHFL_XOR(sum, offset, Parameters::WarpSize);
        }
        
        acc_t mu = sum / Parameters::Dim;
        acc_t var = 0.0;
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations * Parameters::VecSize; ++i) {
            acc_t diff = (acc_t)elements_l[i] - mu;
            var += diff * diff;
        }
        #pragma unroll
        for (int offset = Parameters::WarpSize / 2; offset > 0; offset /= 2) {
            var += SHFL_XOR(var, offset, Parameters::WarpSize);
        }
        const acc_t rsigma = rsqrtf(var / Parameters::Dim + epsilon);
        
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations * Parameters::VecSize; ++i) {
            elements_l[i] = (input_t)(((acc_t)elements_l[i] - mu) * rsigma) * gamma_l[i] + beta_l[i];
        }
        
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations; ++i) {
            *(VecInType *)(dst + i * Parameters::WarpStride) = elements[i];
        }
        
        if (threadIdx.x == 0) {
            mean[batch] = mu;
            invvar[batch] = rsigma;
        }
    }
}

template <typename IndexType, typename input_t, typename output_t, typename acc_t, typename Parameters>
__global__ void layernorm_backward_x(output_t *dst, const input_t *input, const input_t *grad, const input_t *gamma,
    const acc_t *mean, const acc_t *invvar, IndexType bsz) {
    IndexType batch = blockIdx.x * Parameters::BatchesPerBlock + threadIdx.y;
    if (batch < bsz) {
        input += batch * Parameters::Dim + threadIdx.x * Parameters::VecSize;
        dst += batch * Parameters::Dim + threadIdx.x * Parameters::VecSize;
        grad += batch * Parameters::Dim + threadIdx.x * Parameters::VecSize;
        gamma += threadIdx.x * Parameters::VecSize;
        using VecInType = VecType<input_t, Parameters::VecSize>;
        VecInType elements[Parameters::Iterations];
        VecInType grad_reg[Parameters::Iterations];
        VecInType gamma_reg[Parameters::Iterations];
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations; ++i) {
            elements[i] = *(VecInType *)(input + i * Parameters::WarpStride);
            grad_reg[i] = *(VecInType *)(grad + i * Parameters::WarpStride);
            gamma_reg[i] = *(VecInType *)(gamma + i * Parameters::WarpStride);
        }
        input_t *elements_l = (input_t *)elements;
        input_t *grad_l = (input_t *)grad_reg;
        input_t *gamma_l = (input_t *)gamma_reg;
        const acc_t mu = mean[batch];
        const acc_t var = invvar[batch];
        
        acc_t sum1 = 0.0, sum2 = 0.0;
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations * Parameters::VecSize; ++i) {
            elements_l[i] = elements_l[i] - (input_t)mu;
            sum1 += (acc_t)(elements_l[i] * grad_l[i] * gamma_l[i]);
            sum2 += (acc_t)(grad_l[i] * gamma_l[i]);
        }
        
        #pragma unroll
        for (int offset = Parameters::WarpSize / 2; offset > 0; offset /= 2) {
            sum1 += SHFL_XOR(sum1, offset, Parameters::WarpSize);
        }
        
        #pragma unroll
        for (int offset = Parameters::WarpSize / 2; offset > 0; offset /= 2) {
            sum2 += SHFL_XOR(sum2, offset, Parameters::WarpSize);
        }
        
        sum1 *= var * var * var / Parameters::Dim;
        sum2 *= var / Parameters::Dim;
        
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations * Parameters::VecSize; ++i) {
            elements_l[i] = grad_l[i] * gamma_l[i] * (input_t)var - (input_t)sum1 * elements_l[i] - (input_t)sum2;
        }
        
        #pragma unroll
        for (int i = 0; i < Parameters::Iterations; ++i) {
            *(VecInType *)(dst + i * Parameters::WarpStride) = elements[i];
        }
    }
}

#define LAUNCH_FORWARD_KERNEL(len, vec, batches, type) \
{ \
    dim3 threads(32, batches); \
    int blocks = DIV_CELL(n1, batches); \
    layernorm_forward<size_t, type, type, float, LNParameters<len, vec, batches, 1>> \
    <<<blocks, threads, 0, stream>>> \
    ((type *)output->data_ptr(), (type *)input->data_ptr(), (type *)gamma->data_ptr(), \
        (type *)beta->data_ptr(), (float *)mean->data_ptr(), (float *)invvar->data_ptr(), n1, epsilon); \
    break; \
}

#define LAUNCH_BACKWARD_KERNEL(len, vec, batches, type) \
{ \
    dim3 threads(32, batches); \
    int blocks = DIV_CELL(n1, batches); \
    layernorm_backward_x<size_t, type, type, float, LNParameters<len, vec, batches, 1>> \
    <<<blocks, threads, 0, stream>>> \
    ((type *)grad_input->data_ptr(), (type *)input->data_ptr(), (type *)dout->data_ptr(), \
        (type *)gamma->data_ptr(), (float *)mean->data_ptr(), (float *)invvar->data_ptr(), n1); \
    break; \
}

void cuda_layer_norm(
    at::Tensor* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon)
{
    using namespace at;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto type = input->scalar_type();
    
    if (type == at::ScalarType::BFloat16) {
        switch (n2) {
        case 64: LAUNCH_FORWARD_KERNEL(64, 2, 4, nv_bfloat16)
        case 128: LAUNCH_FORWARD_KERNEL(128, 2, 4, nv_bfloat16)
        case 256: LAUNCH_FORWARD_KERNEL(256, 2, 4, nv_bfloat16)
        case 384: LAUNCH_FORWARD_KERNEL(384, 2, 4, nv_bfloat16)
        case 512: LAUNCH_FORWARD_KERNEL(512, 2, 4, nv_bfloat16)
        case 768: LAUNCH_FORWARD_KERNEL(768, 2, 4, nv_bfloat16)
        case 1024: LAUNCH_FORWARD_KERNEL(1024, 2, 4, nv_bfloat16)
        case 1280: LAUNCH_FORWARD_KERNEL(1280, 2, 4, nv_bfloat16)
        case 1536: LAUNCH_FORWARD_KERNEL(1536, 2, 4, nv_bfloat16)
        case 1792: LAUNCH_FORWARD_KERNEL(1792, 2, 4, nv_bfloat16)
        case 2048: LAUNCH_FORWARD_KERNEL(2048, 2, 4, nv_bfloat16)
        }
    } else if (type == at::ScalarType::Half) {
        switch (n2) {
        case 64: LAUNCH_FORWARD_KERNEL(64, 2, 4, half)
        case 128: LAUNCH_FORWARD_KERNEL(128, 2, 4, half)
        case 256: LAUNCH_FORWARD_KERNEL(256, 2, 4, half)
        case 384: LAUNCH_FORWARD_KERNEL(384, 2, 4, half)
        case 512: LAUNCH_FORWARD_KERNEL(512, 2, 4, half)
        case 768: LAUNCH_FORWARD_KERNEL(768, 2, 4, half)
        case 1024: LAUNCH_FORWARD_KERNEL(1024, 2, 4, half)
        case 1280: LAUNCH_FORWARD_KERNEL(1280, 2, 4, half)
        case 1536: LAUNCH_FORWARD_KERNEL(1536, 2, 4, half)
        case 1792: LAUNCH_FORWARD_KERNEL(1792, 2, 4, half)
        case 2048: LAUNCH_FORWARD_KERNEL(2048, 2, 4, half)
        }
    } else if (type == at::ScalarType::Float) {
        switch (n2) {
        case 64: LAUNCH_FORWARD_KERNEL(64, 1, 4, float)
        case 128: LAUNCH_FORWARD_KERNEL(128, 1, 4, float)
        case 256: LAUNCH_FORWARD_KERNEL(256, 1, 4, float)
        case 384: LAUNCH_FORWARD_KERNEL(384, 1, 4, float)
        case 512: LAUNCH_FORWARD_KERNEL(512, 1, 4, float)
        case 768: LAUNCH_FORWARD_KERNEL(768, 1, 4, float)
        case 1024: LAUNCH_FORWARD_KERNEL(1024, 1, 4, float)
        case 1280: LAUNCH_FORWARD_KERNEL(1280, 1, 4, float)
        case 1536: LAUNCH_FORWARD_KERNEL(1536, 1, 4, float)
        case 1792: LAUNCH_FORWARD_KERNEL(1792, 1, 4, float)
        case 2048: LAUNCH_FORWARD_KERNEL(2048, 1, 4, float)
        }
    }
}

void cuda_layer_norm_gradient(
    at::Tensor* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon,
    at::Tensor* grad_input)
{   
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto type = input->scalar_type();
    
    if (type == at::ScalarType::BFloat16) {
        switch (n2) {
        case 64: LAUNCH_BACKWARD_KERNEL(64, 2, 4, nv_bfloat16)
        case 128: LAUNCH_BACKWARD_KERNEL(128, 2, 4, nv_bfloat16)
        case 256: LAUNCH_BACKWARD_KERNEL(256, 2, 4, nv_bfloat16)
        case 384: LAUNCH_BACKWARD_KERNEL(384, 2, 4, nv_bfloat16)
        case 512: LAUNCH_BACKWARD_KERNEL(512, 2, 4, nv_bfloat16)
        case 768: LAUNCH_BACKWARD_KERNEL(768, 2, 4, nv_bfloat16)
        case 1024: LAUNCH_BACKWARD_KERNEL(1024, 2, 4, nv_bfloat16)
        case 1280: LAUNCH_BACKWARD_KERNEL(1280, 2, 4, nv_bfloat16)
        case 1536: LAUNCH_BACKWARD_KERNEL(1536, 2, 4, nv_bfloat16)
        case 1792: LAUNCH_BACKWARD_KERNEL(1792, 2, 4, nv_bfloat16)
        case 2048: LAUNCH_BACKWARD_KERNEL(2048, 2, 4, nv_bfloat16)
        }
    } else if (type == at::ScalarType::Half) {
        switch (n2) {
        case 64: LAUNCH_BACKWARD_KERNEL(64, 2, 4, half)
        case 128: LAUNCH_BACKWARD_KERNEL(128, 2, 4, half)
        case 256: LAUNCH_BACKWARD_KERNEL(256, 2, 4, half)
        case 384: LAUNCH_BACKWARD_KERNEL(384, 2, 4, half)
        case 512: LAUNCH_BACKWARD_KERNEL(512, 2, 4, half)
        case 768: LAUNCH_BACKWARD_KERNEL(768, 2, 4, half)
        case 1024: LAUNCH_BACKWARD_KERNEL(1024, 2, 4, half)
        case 1280: LAUNCH_BACKWARD_KERNEL(1280, 2, 4, half)
        case 1536: LAUNCH_BACKWARD_KERNEL(1536, 2, 4, half)
        case 1792: LAUNCH_BACKWARD_KERNEL(1792, 2, 4, half)
        case 2048: LAUNCH_BACKWARD_KERNEL(2048, 2, 4, half)
        }
    } else if (type == at::ScalarType::Float) {
        switch (n2) {
        case 64: LAUNCH_BACKWARD_KERNEL(64, 1, 4, float)
        case 128: LAUNCH_BACKWARD_KERNEL(128, 1, 4, float)
        case 256: LAUNCH_BACKWARD_KERNEL(256, 1, 4, float)
        case 384: LAUNCH_BACKWARD_KERNEL(384, 1, 4, float)
        case 512: LAUNCH_BACKWARD_KERNEL(512, 1, 4, float)
        case 768: LAUNCH_BACKWARD_KERNEL(768, 1, 4, float)
        case 1024: LAUNCH_BACKWARD_KERNEL(1024, 1, 4, float)
        case 1280: LAUNCH_BACKWARD_KERNEL(1280, 1, 4, float)
        case 1536: LAUNCH_BACKWARD_KERNEL(1536, 1, 4, float)
        case 1792: LAUNCH_BACKWARD_KERNEL(1792, 1, 4, float)
        case 2048: LAUNCH_BACKWARD_KERNEL(2048, 1, 4, float)
        }
    }
}
