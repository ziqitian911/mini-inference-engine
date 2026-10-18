# minimum version required
cmake_minimum_required(VERSION 3.20)

# project name and language
project(mini_inference_engine CXX)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main src/main.cpp src/tensor.cpp src/gemm.cpp src/ops.cpp)

target_include_directories(main PRIVATE include)