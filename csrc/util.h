#pragma once
#define DIV_CELL(a, b) (((a) + (b) - 1) / (b))
#if __cplusplus >= 201703L
    #define IF_CONSTEXPR constexpr
#else
    #define IF_CONSTEXPR
#endif

template <typename T>
__device__ __forceinline__ T SHFL_XOR(T value, int laneMask, int width, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T, int N>
struct VecTypeImpl;

#define DEFINE_VEC_TYPE(t, n, tn) \
template <> \
struct VecTypeImpl<t, n> { \
    using type = tn; \
};

DEFINE_VEC_TYPE(half, 1, half)
DEFINE_VEC_TYPE(__nv_bfloat16, 1, __nv_bfloat16)
DEFINE_VEC_TYPE(float, 1, float)
DEFINE_VEC_TYPE(half, 2, half2)
DEFINE_VEC_TYPE(__nv_bfloat16, 2, __nv_bfloat162)
DEFINE_VEC_TYPE(float, 2, float2)
DEFINE_VEC_TYPE(half, 4, uint64_t)
DEFINE_VEC_TYPE(__nv_bfloat16, 4, uint64_t)
DEFINE_VEC_TYPE(float, 4, float4)

template <typename T, int N>
using VecType = typename VecTypeImpl<T, N>::type;