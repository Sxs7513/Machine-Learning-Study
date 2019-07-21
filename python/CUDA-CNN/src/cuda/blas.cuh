#pragma once

#include "storage.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


void operator_mul(const Storage *input1, const Storage *input2, Storage *outputs);

void operator_mul(const Storage *input, float value, Storage *outputs);