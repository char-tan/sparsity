#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
//   return 1.0 / (1.0 + exp(-z));
// }

torch::Tensor ops_cuda_add(torch::Tensor a, torch::Tensor b) {

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(a.type(), "ops_add_cuda", ([&] {
    ops_cuda_add_kernel<scalar_t><<<blocks, threads>>>(
        a.data<scalar_t>(),
        b.data<scalar_t>());
  }));

  return {a, b};
}


