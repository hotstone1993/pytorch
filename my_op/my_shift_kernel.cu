#include "my_shift_kernel.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>

using namespace at::cuda::detail;

__global__ void gup_shift(float* input, float* output, int64_t move) {
    // ...
}

void cpu_shift(float* input, float* ouput, int64_t move) {
    // ...
}


torch::Tensor shift_op(torch::Tensor input, int64_t move) {
    torch::Device device(torch::kCUDA, 0);
    torch::Tensor output = torch::zeros(input.size(0), torch::kFloat);
    output = output.to(input);

    if (input.device() == device){
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        const dim3 grid(GET_BLOCKS(input.size(0)));
        const dim3 block(CUDA_NUM_THREADS);
        
        gup_shift<<<grid, block, 0, stream>>>(input.data_ptr<float>(), output.data_ptr<float>(), move);
    } else {
        cpu_shift(input.data_ptr<float>(), output.data_ptr<float>(), move);
    }

    return output;
}