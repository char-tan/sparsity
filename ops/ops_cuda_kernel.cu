#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__global__ void cuda_add_kernel(
    scalar_t* __restrict__ a,
    scalar_t* __restrict__ b,
    scalar_t* __restrict__ c
    ) {
        c = a + b
      }


torch::Tensor cuda_add(
		torch::Tensor a,
		torch::Tensor b)
{

  auto c = torch::zeros_like(a);

  AT_DISPATCH_FLOATING_TYPES(a.type(), "cuda_add", ([&] {
    cuda_add_kernel<scalar_t><<<1, 1>>>(
        a.data<scalar_t>(),
        b.data<scalar_t>(),
        c.data<scalar_t>(),
  })));

  return {c}
}