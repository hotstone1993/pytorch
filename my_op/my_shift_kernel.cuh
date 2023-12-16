#pragma once
#include <torch/script.h>

torch::Tensor shift_op(torch::Tensor input, int64_t move);