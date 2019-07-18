#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

#define CUDA_CHECK(condition)                                \
  do {                                                       \
    cudaError_t error = condition;                           \
    CHECK_EQ(error, cudaSuccess, cudaGetErrorString(error)); \
  } while (0)                                                \

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

#define CHECK_EQ(val1, val2, message)                              \
  do {                                                             \
    if (val1 != val2) {                                            \
      std::cerr << __FILE__ << "(" << __LINE__ << "): " << message \
                << std::endl;                                      \
      exit(1);                                                     \
    }                                                              \
  } while (0)

#define RAW_PTR(vector) thrust::raw_pointer_cast(vector.data())