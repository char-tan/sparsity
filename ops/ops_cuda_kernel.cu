#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


torch::Tensor cuda_add(
		torch::Tensor a,
		torch::Tensor b)
{
	auto c = torch::add(a, b);
	return c;
}
