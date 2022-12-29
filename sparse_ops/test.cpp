#include <torch/extension.h>

#include <iostream>

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
  torch::Tensor c = a + b;
  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Sparse Add");
}
