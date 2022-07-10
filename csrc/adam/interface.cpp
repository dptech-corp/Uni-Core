#include <torch/extension.h>

void fused_adam_cuda(at::Tensor & p, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int bias_correction, float decay);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void adam(at::Tensor & p, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int bias_correction, float decay) {
    CHECK_INPUT(p);
    CHECK_INPUT(m);
    CHECK_INPUT(v);
    CHECK_INPUT(g);
    int64_t num_elem = p.numel();
    AT_ASSERTM(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
    AT_ASSERTM(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
    AT_ASSERTM(g.numel() == num_elem, "number of elements in g and p tensors should be equal");
    fused_adam_cuda(p, m, v, g, lr, beta1, beta2, eps, grad_scale, step, bias_correction, decay);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam", &adam, "Adam optimized CUDA implementation.");
}