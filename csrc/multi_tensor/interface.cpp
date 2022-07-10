#include <torch/extension.h>


at::Tensor multi_tensor_l2norm_cuda(
  int chunk_size,
  std::vector<std::vector<at::Tensor>> tensor_lists);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
}