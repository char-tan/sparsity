#ifndef FOOP_H
#define FOOP_H
#include <torch/extension.h>

int cpp_sparsemm(torch::Tensor a, torch::Tensor b);
#endif
