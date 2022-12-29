#include <torch/extension.h>

// CUDA function declaration

torch::Tensor cuda_add(
		torch::Tensor a,
		torch::Tensor b);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor add(
		torch::Tensor a,
		torch::Tensor b) 
{
	CHECK_INPUT(a);
	CHECK_INPUT(b);
	return cuda_add(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
	m.def("add", &add, "cuda add");
}
