#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template<typename x_t>
struct L2NormFunctor
{
  __device__ __forceinline__ void operator()(
    int chunk_size,
    TensorListMetadata<1>& tl,
    float* output)
  {

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t* x = (x_t*)tl.addresses[0][tensor_loc];
    x += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    __shared__ float s_vals[512];

    float vals[ILP];
    x_t r_x[ILP];
    for(int i = 0; i < ILP; i++)
    {
      vals[i] = 0.0f;
      r_x[i] = (x_t)0.0f;
    }

    if(n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(x))
    {
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(r_x, x, 0 , i_start);
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          float next = static_cast<float>(r_x[ii]);
          vals[ii] += next*next;
        }
      }
    }
    else
    {
      for(int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x*ILP)
      {
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            float next = static_cast<float>(x[i]);
            vals[ii] += next*next;
          }
        }
      }
    }

    float val = 0.f;
    for(int i = 0; i < ILP; i++)
        val += vals[i];

    float res = reduce_block_into_lanes(s_vals, val);

    if(threadIdx.x == 0)
    {
      output[blockIdx.x] += res;
    }
  }
};



__global__ void cleanup(
  float* output,
  float* ret)
{
  __shared__ float vals[512];

  if(blockIdx.x == 0)
  {
    float val = 0;
    if(threadIdx.x < 320)
      val = output[threadIdx.x];

    float final = reduce_block_into_lanes(vals, val);

    if(threadIdx.x == 0)
      *ret = sqrt(final);
  }
}


at::Tensor multi_tensor_l2norm_cuda(
  int chunk_size,
  std::vector<std::vector<at::Tensor>> tensor_lists)
{
  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  auto output = at::zeros({320}, float_options);

  switch (tensor_lists[0][0].scalar_type()){
    case at::ScalarType::Float: { 
      multi_tensor_apply<1>(
        BLOCK_SIZE,
        chunk_size,
        tensor_lists,
        L2NormFunctor<float>(),
        output.data_ptr<float>()
      );
      break; 
    }
    case at::ScalarType::Half: { 
      multi_tensor_apply<1>(
        BLOCK_SIZE,
        chunk_size,
        tensor_lists,
        L2NormFunctor<half>(),
        output.data_ptr<float>()
      );
      break; 
    }
    case at::ScalarType::BFloat16: { 
      multi_tensor_apply<1>(
        BLOCK_SIZE,
        chunk_size,
        tensor_lists,
        L2NormFunctor<nv_bfloat16>(),
        output.data_ptr<float>()
      );
      break; 
    }
  }

  AT_CUDA_CHECK(cudaGetLastError());

  auto ret = at::empty({1}, output.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  auto stream = at::cuda::getCurrentCUDAStream();
  cleanup<<<1, 512, 0, stream>>>(
    output.data_ptr<float>(),
    ret.data_ptr<float>());

  return ret;
}
