#include <blas.cuh>
#include <utils.cuh>

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cfloat>


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
    __host__ __device__ float operator(const float &x) const { return x * e; }

};

void operator_mul(const Storage *input1, float value, Storage *outputs) {
    thrust::transform(
        input1->get_data().begin(), input1->get_data().end(),
        input2->get_data().begin(), outputs->get_data().begin(),
        mul_functor(value)
    );
}