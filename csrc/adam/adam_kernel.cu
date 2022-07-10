#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "ATen/TensorUtils.h"
#include "ATen/AccumulateType.h"
#include <ATen/cuda/Exceptions.h>

#include "type_shim.h"

template <typename T, typename GRAD_T>
__global__ void adam_cuda_kernel(
    GRAD_T* __restrict__ p,
    T* __restrict__ m,
    T* __restrict__ v,
    const GRAD_T * __restrict__ g,
    const float b1,
    const float b2,
    const float eps,
    const float grad_scale,
    const float step_size,
    const size_t tsize,
    const float decay_size)
{
    //Assuming 2D grids and 2D blocks
    const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
    const int i = (blockId * threadsPerBlock + threadIdInBlock);
    const int totThreads = gridDim.x*gridDim.y*threadsPerBlock;

    for (int j = i; j < tsize; j+=totThreads) {
        // weight decay
        T cur_p = (T)p[j] * decay_size;
        T scaled_grad = static_cast<T>(g[j]) / grad_scale;
        m[j] = b1*m[j] + (1-b1)*scaled_grad;
        v[j] = b2*v[j] + (1-b2)*scaled_grad*scaled_grad;
        const float update = m[j] / (sqrtf(v[j]) + eps);
        p[j] = cur_p - (step_size*update);
    }
}

void fused_adam_cuda(
    at::Tensor & p,
    at::Tensor & m,
    at::Tensor & v,
    at::Tensor & g,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float grad_scale,
    int step,
    int bias_correction,
    float decay)
{
    //Get tensor size
    int tsize = p.numel();
    //Determine #threads and #blocks
    const int threadsPerBlock = 512;
    const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p), "parameter tensor is too large to be indexed with int32");
    //Constants
    float step_size = lr;
    if (bias_correction == 1) {
        const double bias_correction1 = 1.0 - std::pow(static_cast<double>(beta1), step);
        const double bias_correction2 = 1.0 - std::pow(static_cast<double>(beta2), step);
        step_size = static_cast<float>(lr * std::sqrt(bias_correction2) / bias_correction1);
    }
    float decay_size = 1.0;
    if (decay != 0.0) {
        decay_size = 1.0 - step_size * decay;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (g.scalar_type() == at::ScalarType::Half || g.scalar_type() == at::ScalarType::BFloat16) {
        AT_ASSERTM(p.scalar_type() == g.scalar_type(), "expected parameter to be the same type as grad");
        using namespace at; // prevents "toString is undefined" errors
        DISPATCH_FLOAT_AND_HALF_AND_BF16(g.scalar_type(), 0, "adam_cuda_kernel",
            using accscalar_t = at::acc_type<scalar_t_0, true>;
            adam_cuda_kernel<accscalar_t, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                    p.data_ptr<scalar_t_0>(),
                    m.data_ptr<accscalar_t>(),
                    v.data_ptr<accscalar_t>(),
                    g.data_ptr<scalar_t_0>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    tsize,
                    decay_size);
            );
    } else {
        using namespace at;
        DISPATCH_DOUBLE_AND_FLOAT(g.scalar_type(), 0, "adam_cuda_kernel",
            adam_cuda_kernel<scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                    p.data_ptr<scalar_t_0>(),
                    m.data_ptr<scalar_t_0>(),
                    v.data_ptr<scalar_t_0>(),
                    g.data_ptr<scalar_t_0>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    tsize,
                    decay_size);
        );
    }
    AT_CUDA_CHECK(cudaGetLastError());
}
