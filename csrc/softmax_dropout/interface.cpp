#include <torch/extension.h>
#include <ATen/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <vector>

std::vector<c10::optional<torch::Tensor>> fwd_cuda(
    bool is_training,
    torch::Tensor &input,
    const c10::optional<torch::Tensor> &attn_mask,
    const c10::optional<torch::Tensor> &bias,
    float dropout_prob,
    c10::optional<at::Generator> gen_);

torch::Tensor bwd_cuda(
    torch::Tensor &output_grads,
    const torch::Tensor &softmax_results,
    const c10::optional<torch::Tensor> &dropout_mask,
    float dropout_prob);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

std::vector<c10::optional<torch::Tensor>> fwd(
    bool is_training,
    torch::Tensor &input,
    const c10::optional<torch::Tensor> &attn_mask,
    const c10::optional<torch::Tensor> &bias,
    float dropout_prob,
    c10::optional<at::Generator> gen_)
{
    CHECK_INPUT(input);
    if (attn_mask)
    {
        CHECK_INPUT(attn_mask.value());
        AT_ASSERTM(attn_mask->dim() == 3, "expected 3D tensor");
    }
    if (bias)
    {
        CHECK_INPUT(bias.value());
        AT_ASSERTM(bias->dim() == 3, "expected 3D tensor");
        AT_ASSERTM(input.size(0) % bias->size(0) == 0, "wrong first dim of bias.");
        AT_ASSERTM(bias->size(1) == input.size(1) && bias->size(2) == input.size(2), "the last two dims of bias and input should be the same.");
    }
    AT_ASSERTM(input.dim() == 3, "expected 3D tensor");
    AT_ASSERTM(input.scalar_type() == at::ScalarType::Half ||
                   input.scalar_type() == at::ScalarType::BFloat16 ||
                   input.scalar_type() == at::ScalarType::Float,
               "Only HALF/BFloat16/Float is supported");
    return fwd_cuda(is_training, input, attn_mask, bias, dropout_prob, gen_);
}

torch::Tensor bwd(
    torch::Tensor &output_grads,
    const torch::Tensor &softmax_results,
    const c10::optional<torch::Tensor> &dropout_mask,
    float dropout_prob)
{
    CHECK_INPUT(output_grads);
    CHECK_INPUT(softmax_results);
    if (dropout_mask)
    {
        CHECK_INPUT(dropout_mask.value());
    }
    AT_ASSERTM(output_grads.dim() == 3, "expected 3D tensor");
    AT_ASSERTM(softmax_results.dim() == 3, "expected 3D tensor");
    AT_ASSERTM(!dropout_mask || dropout_mask->dim() == 1, "expected 1D tensor");

    AT_ASSERTM(output_grads.scalar_type() == at::ScalarType::Half ||
                   output_grads.scalar_type() == at::ScalarType::BFloat16 ||
                   output_grads.scalar_type() == at::ScalarType::Float,
               "Only HALF/BFloat16/Float is supported");
    AT_ASSERTM(softmax_results.scalar_type() == at::ScalarType::Half ||
                   softmax_results.scalar_type() == at::ScalarType::BFloat16 ||
                   softmax_results.scalar_type() == at::ScalarType::Float,
               "Only HALF/BFloat16/Float is supported");
    AT_ASSERTM(output_grads.scalar_type() == softmax_results.scalar_type(), "the types mismatch");
    return bwd_cuda(output_grads, softmax_results, dropout_mask, dropout_prob);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &fwd, "softmax dropout -- Forward.");
    m.def("backward", &bwd, "softmax dropout -- Backward.");
}
