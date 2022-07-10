#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
#include <iostream>

union float_int_32
{
    uint32_t i;
    float f;
};

__global__ void fp32_to_bf16(
    const float* input,
    nv_bfloat16* output,
    const int tsize,
    uint64_t seed,
    uint64_t offset) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < tsize) {
        float_int_32 d;
        d.f = input[i];
        curandStatePhilox4_32_10_t state;
        curand_init(seed, i, offset, &state);
        d.i += curand(&state) & 0x0000ffff;
        output[i] = __float2bfloat16_rz(d.f);
    }
}

void fused_fp32_to_bf16_sr_cuda(
    at::Tensor & input,
    at::Tensor & output)
{
    int tsize = input.numel();
    const int threadsPerBlock = 512;
    const int blocks = (tsize + threadsPerBlock - 1) / threadsPerBlock;
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(input), "parameter tensor is too large to be indexed with int32");
    AT_ASSERTM(input.scalar_type() == at::ScalarType::Float, "expected input to be float32 tensor");
    AT_ASSERTM(output.scalar_type() == at::ScalarType::BFloat16, "expected output to be bfloat16 tensor");
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    std::pair<uint64_t, uint64_t> rng_engine_inputs;
    {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen.mutex());
        rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(1);
    }
    uint64_t seed = std::get<0>(rng_engine_inputs);
    uint64_t offset = std::get<1>(rng_engine_inputs);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fp32_to_bf16<<<blocks, threadsPerBlock, 0, stream>>>(
        (const float*)input.data_ptr(),
        (nv_bfloat16*)output.data_ptr(),
        tsize,
        seed,
        offset);
    AT_CUDA_CHECK(cudaGetLastError());
}

