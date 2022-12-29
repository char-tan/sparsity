#include <torch/extension.h>

#include <iostream>

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
  torch::Tensor c = a + b;
  return c;
}
