#include "torch/script.h"
#include "my_shift_op.cuh"

torch::Tensor shift_op(torch::Tensor input, int move) {
    torch::Device device(torch::kCUDA, 0);
    torch::Tensor output = torch::zeros(input.size(0), torch::kFloat);
    
    if (input.device() == device){

    } else {

    }

    return output;
}