#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>


torch::Tensor shift_op(torch::Tensor input, int move);