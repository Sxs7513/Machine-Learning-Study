#include <vector>
#include "cuda/dcn_v2_im2col_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

extern THCState *state;


at::Tensor
dcn_v2_cuda_forward(const at::Tensor &input, 
                    const at::Tensor &weight,
                    const at::Tensor &bias,
                    const at::Tensor &offset,
                    const at::Tensor &mask,
                    const int kernel_h,
                    const int kernel_w,
                    const int stride_h,
                    const int stride_w,
                    const int pad_h,
                    const int pad_w,
                    const int dilation_h,
                    const int dilation_w,
                    const int deformable_group)
{
    using scalar_t = float;

    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(weight.type().is_cuda(), "weight must be a CUDA tensor");
    AT_ASSERTM(bias.type().is_cuda(), "bias must be a CUDA tensor");
    AT_ASSERTM(offset.type().is_cuda(), "offset must be a CUDA tensor");
    AT_ASSERTM(mask.type().is_cuda(), "mask must be a CUDA tensor");
    
    const int batch = input.size(0);
    const int channels = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int channels_kernel = weight.size(1);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
        "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
        "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

}