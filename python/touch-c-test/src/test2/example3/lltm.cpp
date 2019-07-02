#include <torch/torch.h>
#include <iostream>
#include <vector>

at::Tensor d_sigmiod(at::Tensor z) {
    auto s = at::sigmoid(z);
    return (1 - s) * s
}

std::vector<at::Tensor> lltm_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell
) {
    auto X = at::cat({ old_h, input }, /* dim= */1);

    auto gate_weights = at::addmm(bias, X, weights.transpose(0, 1));
    auto gates = gate_weights.chunk(3, /* dim= */1)
}