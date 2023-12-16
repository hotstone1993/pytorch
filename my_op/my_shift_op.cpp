#include <torch/torch.h>
#include <torch/script.h>
#include "my_shift_kernel.cuh"

TORCH_LIBRARY (my_ops, m){
    m.def("shift_op", shift_op);
}