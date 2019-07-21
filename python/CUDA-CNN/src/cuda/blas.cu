#include "blas.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cfloat>
#include <cmath>


void operator_mul(const Storage *input1, const Storage *input2, Storage *outputs) {
    CHECK_EQ(input1->get_data().size(), input2->get_data().size(), "operator_mul: size error");

    thrust::transform(
        input1->get_data().begin(), input1->get_data().end(),
        input2->get_data().begin(), outputs->get_data().begin(),
        thrust::multiplies<float>()
    );
}

struct mul_functor {
    const float e;
    mul_functor(float _e): e(_e) {}
    __host__ __device__ float operator()(const float &x) const { return x * e; }

};

void operator_mul(const Storage *input1, float value, Storage *outputs) {
    thrust::transform(
        input1->get_data().begin(), input1->get_data().end(),
        outputs->get_data().begin(), mul_functor(value)
    );
}

void operator_matmul(const Storage *input1, const Storage *input2,
                    Storage *outputs, int broadcast) {
    int height = *(input1->get_shape().rbegin() + 1);
    int k = *(input1->get_shape().rbegin());
    int width = *(input2->get_shape().rbegin());
    CHECK_EQ(k, *(input2->get_shape().rbegin() + 1),
           "operator_matmul: shape error");

    std::vector<int> base_shape = 
        input1->get_shape().size() > input2->get_shape().size()
          ? input1->get_shape()
          : input2->get_shape();
    int batch_size = 1;
    for (auto i = base_shape.rbegin() + 2; i != base_shape.rend(); i++) {
        batch_size *= *i;
    }

    // pointer
    const float *input1_ptr = RAW_PTR(input1->get_data());
    const float *input2_ptr = RAW_PTR(input2->get_data());
    float *output_ptr = RAW_PTR(outputs->get_data());

    dim3 dim_block(TILE_SIZE, TILE_SIZE);
    dim3 dim_grid(ceil((float)width / TILE_SIZE), ceil((float)height / TILE_SIZE),
                batch_size);
}