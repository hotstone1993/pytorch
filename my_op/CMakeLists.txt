cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(my_shift_op)

find_package(Torch REQUIRED)

add_library(my_shift_op SHARED my_shift_op.cpp my_shift_kernel.cu)
target_compile_features(my_shift_op PRIVATE cxx_std_14)
target_link_libraries(my_shift_op "${TORCH_LIBRARIES}")
target_link_libraries(my_shift_op )