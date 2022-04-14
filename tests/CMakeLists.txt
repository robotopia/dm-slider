cmake_minimum_required(VERSION 3.10)

# set the project name
project(tests CUDA)

# add the executable
add_executable(vdif_benchmarks vdif_benchmarks.cu)

#target_compile_options(vdif_benchmarks PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)