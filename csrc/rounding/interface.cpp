#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void fused_fp32_to_bf16_sr_cuda(at::Tensor & input, at::Tensor & output);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void fused_fp32_to_bf16_sr(at::Tensor & input, at::Tensor & output) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    int64_t num_elem = input.numel();
    AT_ASSERTM(output.numel() == num_elem, "number of elements in input ond output tensors should be equal");
    fused_fp32_to_bf16_sr_cuda(input, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp32_to_bf16_sr", &fused_fp32_to_bf16_sr, "fused fp32 to bf16 random rounding");
}