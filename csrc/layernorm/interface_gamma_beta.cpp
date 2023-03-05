#include <torch/extension.h>
#include <vector>
#include <cassert>

namespace {
void compute_n1_n2(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    int& n1,
    int& n2)
{
    int idiff = input.ndimension() - normalized_shape.size();
    n2 = 1;
    for (int i = 0;  i < (int)normalized_shape.size();  ++i) {
	    assert( input.sizes()[i+idiff] == normalized_shape[i] );
	    n2 *= normalized_shape[i];
    }
    n1 = 1;
    for (int i = 0;  i < idiff;  ++i) {
	    n1 *= input.sizes()[i];
    }
}

void check_args(
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    at::Tensor beta
    )
{
    TORCH_CHECK(!gamma.defined() || gamma.sizes().equals(normalized_shape));
    TORCH_CHECK(!beta.defined() || beta.sizes().equals(normalized_shape));
}

void check_args(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    int& n1,
    int& n2
    )
{
    int64_t normalized_ndim = normalized_shape.size();

    if (normalized_ndim < 1) {
      std::stringstream ss;
      ss << "Expected normalized_shape to be at least 1-dimensional, i.e., "
         << "containing at least one element, but got normalized_shape="
         << normalized_shape;
      throw std::runtime_error(ss.str());
    }

    auto input_shape = input.sizes();
    auto input_ndim = input.dim();

    if (input_ndim < normalized_ndim ||
        !input_shape.slice(input_ndim - normalized_ndim).equals(normalized_shape)) {
      std::stringstream ss;
      ss << "Given normalized_shape=" << normalized_shape
         << ", expected input with shape [*";
      for (auto size : normalized_shape) {
        ss << ", " << size;
      }
      ss << "], but got input of size" << input_shape;
      throw std::runtime_error(ss.str());
    }

    compute_n1_n2(input,normalized_shape,n1,n2);
}


void check_args(
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    at::Tensor beta,
    int& n1,
    int& n2
    )
{
    check_args(input,normalized_shape,n1,n2);
    check_args(normalized_shape,gamma,beta);
}
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta
    );

std::vector<at::Tensor> layer_norm_gradient(
    at::Tensor dout,
    at::Tensor mean,
    at::Tensor invvar,
    at::Tensor input,
    at::IntArrayRef normalized_shape,
    at::Tensor gamma,
    at::Tensor beta,
    double epsilon) {
  CHECK_INPUT(dout);
  CHECK_INPUT(mean);
  CHECK_INPUT(invvar);
  CHECK_INPUT(input);
  CHECK_INPUT(gamma);
  CHECK_INPUT(beta);
  int n1,n2;
  check_args(input,normalized_shape,gamma,beta,n1,n2);
  TORCH_CHECK(n2 == 64 || n2 == 128 || n2 == 192 || n2 == 256 || n2 == 320 || n2 == 384 || n2 == 512 || n2 == 640 || n2 == 768 || n2 == 1024 || n2 == 1280 ||
              n2 == 1536 || n2 == 1792 || n2 == 2048 || n2 == 2560 || n2 == 5120, "dimension is not supported");
  at::Tensor grad_gamma = at::empty_like(gamma);
  at::Tensor grad_beta = at::empty_like(beta);
  cuda_layer_norm_gradient(&dout,&mean,&invvar,&input,n1,n2,
      normalized_shape,&gamma,&beta,epsilon,
      &grad_gamma,&grad_beta);
  return {grad_gamma, grad_beta};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &layer_norm_gradient,
    "LayerNorm fast backward for computing gamma and beta (CUDA)");
}
