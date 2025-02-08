#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "type_shim.h"

namespace {

template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float>
{
    __device__ float *getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory<double>
{
    __device__ double *getPointer() {
        extern __shared__ double s_double[];
        return s_double;
    }
};
} // end namespace


template<typename T, typename U> __device__
void cuLoadWriteStridedInputs_RMS(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf,
    const T* input,
    const T* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ invrms 
    )
{
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        U curr_invrms = invrms[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            const int i2 = i2_off + k;
            const int load_idx = i1 * n2 + i2;
            const int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                U curr_input = static_cast<U>(input[load_idx]);
                U curr_dout  = static_cast<U>(dout[load_idx]);
                // RMSNorm：normalized input = x * invrms
                warp_buf[write_idx] = curr_dout * curr_input * curr_invrms;
            } else {
                warp_buf[write_idx] = U(0);
            }
        }
    } else {
        for (int k = 0; k < blockDim.y; ++k) {
            const int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            warp_buf[write_idx] = U(0);
        }
    }
}

template<typename T, typename U> __device__
void cuLoadAddStridedInputs_RMS(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf,
    const T* input,
    const T* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ invrms
    )
{
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        U curr_invrms = invrms[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            const int i2 = i2_off + k;
            const int load_idx = i1 * n2 + i2;
            const int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                U curr_input = static_cast<U>(input[load_idx]);
                U curr_dout  = static_cast<U>(dout[load_idx]);
                warp_buf[write_idx] += curr_dout * curr_input * curr_invrms;
            }
        }
    }
}


template<typename T, typename U> __global__
void cuComputePartGradGamma_RMS(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const int n1,
    const int n2,
    const U* __restrict__ invrms,  
    U* part_grad_gamma)
{
    const int numsegs_n1 = (n1 + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x + 1;
    const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
    const int thr_load_row_off = (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U* warp_buf = buf; 


    cuLoadWriteStridedInputs_RMS<T, U>(i1_beg, thr_load_row_off, thr_load_col_off,
                                        i2_off, row_stride, warp_buf, input, dout, i1_end, n2, invrms);

    for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end; i1_block += blockDim.y * blockDim.y) {
        cuLoadAddStridedInputs_RMS<T, U>(i1_block, thr_load_row_off, thr_load_col_off,
                                         i2_off, row_stride, warp_buf, input, dout, i1_end, n2, invrms);
    }
    __syncthreads();

    U acc = U(0);
    for (int k = 0; k < blockDim.y; ++k) {
        const int row = threadIdx.y + k * blockDim.y;
        const int idx = row * row_stride + threadIdx.x;
        acc += warp_buf[idx];
    }
    warp_buf[threadIdx.y * row_stride + threadIdx.x] = acc;
    __syncthreads();

    for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        if (threadIdx.y < offset) {
            int idx = threadIdx.y * row_stride + threadIdx.x;
            warp_buf[idx] += warp_buf[(threadIdx.y + offset) * row_stride + threadIdx.x];
        }
        __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
        part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf[threadIdx.x];
    }
}

template<typename T, typename U> __global__
void cuComputeGradGamma_RMS(
    const U* part_grad_gamma,
    const int part_size,
    const int n1,
    const int n2,
    T* grad_gamma)
{
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        int num_warp_reductions = part_size / blockDim.y;
        U sum_gamma = U(0);
        const U* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
        for (int warp_offset = 0; warp_offset < num_warp_reductions; ++warp_offset) {
            sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
        }
        for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
            if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_gamma;
            }
            __syncthreads();
            if (threadIdx.y < offset) {
                int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_gamma += buf[read_idx];
            }
            __syncthreads();
        }
        if (threadIdx.y == 0) {
            grad_gamma[i2] = sum_gamma;
        }
    }
}

template<typename T, typename U> 
void HostRMSNormGradient(
    const T* dout,
    const U* invrms,
    at::Tensor* input,
    int n1,
    int n2,
    const T* gamma, 
    double epsilon,
    T* grad_gamma)
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL) {
      const int part_size = 16;
      const dim3 threads2(32, 4, 1);
      const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
      // 计算共享内存大小：只需一份缓存
      const int nshared2 = sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
      at::Tensor part_grad_gamma = at::empty({part_size, n2}, input->options().dtype(
          (input->scalar_type() == at::ScalarType::Half || 
           input->scalar_type() == at::ScalarType::BFloat16) 
          ? at::ScalarType::Float : input->scalar_type()));
      
      cuComputePartGradGamma_RMS<<<blocks2, threads2, nshared2, stream>>>(
          dout,
          input->data_ptr<T>(),
          n1, n2,
          invrms,
          part_grad_gamma.data_ptr<U>());
      
      const dim3 threads3(32, 8, 1);
      const dim3 blocks3((n2 + threads3.x - 1) / threads3.x, 1, 1);
      const int nshared3 = threads3.x * threads3.y * sizeof(U);
      cuComputeGradGamma_RMS<<<blocks3, threads3, nshared3, stream>>>(
          part_grad_gamma.data_ptr<U>(),
          part_size,
          n1, n2,
          grad_gamma);
    }
}


void cuda_rms_norm_gradient(
    at::Tensor* dout,
    at::Tensor* invrms,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntArrayRef normalized_shape,
    at::Tensor* gamma, 
    double epsilon,
    at::Tensor* grad_gamma)
{
    using namespace at;
    DISPATCH_DOUBLE_FLOAT_AND_HALF_AND_BF16(input->scalar_type(), 0, "cuComputeGradInput",
        using accscalar_t = at::acc_type<scalar_t_0, true>;
        HostRMSNormGradient<scalar_t_0, accscalar_t>(
            dout->data_ptr<scalar_t_0>(),
            invrms->data_ptr<accscalar_t>(), 
            input,
            n1, n2,
            gamma->data_ptr<scalar_t_0>(),
            epsilon,
            grad_gamma->data_ptr<scalar_t_0>());
    )
}