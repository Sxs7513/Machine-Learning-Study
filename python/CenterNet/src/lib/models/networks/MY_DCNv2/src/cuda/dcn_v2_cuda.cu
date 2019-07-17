#include <vector>
#include "cuda/dcn_v2_im2col_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

extern THCState *state;


__global__ void createBatchGemmBuffer(const float **input_b,
                                      float **output_b,
                                      float **columns_b,
                                      const float **ones_b,
                                      const float **weight_b,
                                      const float **bias_b,
                                      float *input, 
                                      float *output,
                                      float *columns,
                                      float *ones,
                                      float *weight,
                                      float *bias,
                                      const int input_stride,
                                      const int output_stride,
                                      const int columns_stride,
                                      const int ones_stride,
                                      const int num_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_batches)
    {
        input_b[idx] = input + idx * input_stride;
        output_b[idx] = output + idx * output_stride;
        columns_b[idx] = columns + idx * columns_stride;
        ones_b[idx] = ones + idx * ones_stride;
        // weight and bias are same in different batches
        weight_b[idx] = weight;
        bias_b[idx] = bias;
    }
}


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
    
    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    auto ones = at::ones({batch, height_out, width_out}, input.options());
    auto columns = a::empty({batch, channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());
    
    // use float * because matrices_size contains pointer not numerical 
    int matrices_size = batch * sizeof(float *);
    // what is ** mean
    // https://stackoverflow.com/questions/2893129/what-does-mean-in-c
    // cast to ** because it's what cudaMalloc out
    // https://www.cnblogs.com/scut-fm/p/3756242.html
    auto input_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto output_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto columns_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto ones_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto weight_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));
    auto bias_b = static_cast<const float **>(THCudaMalloc(state, matrices_size));

    const int input_stride = channels * width * height;
    const int output_stride = channels_out * width_out * height_out;
    const int columns_stride = channels * kernel_w * kernel_h * width_out * height_out;
    const int ones_stride = height_out * width_out;
    
    const int block = 128;
    const int grid = (batch + block - 1) / block;

    createBatchGemmBuffer<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        input_b,
        output_b,
        columns_b,
        ones_b,
        weight_b,
        bias_b,
        input.data<scalar_t>(),
        output.data<scalar_t>(),
        columns.data<scalar_t>(),
        ones.data<scalar_t>(),
        weight.data<scalar_t>(),
        bias.data<scalar_t>(),
        input_stride,
        output_stride,
        columns_stride,
        ones_stride,
        batch
    );

    long m_ = channels_out;
    long n_ = height_out * width_out;
    long k_ = 1;

    // bias is one numerical in one batch, so we make ones
    // to simulate a matrix like output
    // and matrix mutiply in cuda see the following link
    // https://blog.csdn.net/xfortius/article/details/9225799
    THCudaBlas_SgemmBatched(
        state,
        "t",
        "n",
        n_,
        m_,
        k_,
        1.0f,
        ones_b,
        k_,
        bias_b,
        k_,
        0.0f,
        output_b,
        n_,
        batch
    )

    modulated_deformable_im2col_cuda(
        THCState_getCurrentStream(state),
        input.data<scalar_t>(),
        offset.data<scalar_t>(),
        mask.data<scalar_t>(),
        batch,
        channels,
        height,
        width,
        height_out,
        width_out,
        kernel_h,
        kernel_w,
        pad_h, 
        pad_w, 
        stride_h, 
        stride_w, 
        dilation_h, 
        dilation_w,                             
        deformable_group,                             
        columns.data<scalar_t>()
    )

    long m = channels_out;
    long n = height_out * width_out;
    long k = channels * kernel_h * kernel_w;

    THCudaBlas_SgemmBatched(
        state,
        "n",
        "n",
        n,
        m,
        k,
        1.0f,
        (const float **)columns_b,
        n,
        weight_b,
        k,
        1.0f,
        output_b,
        n,
        batch
    );

    THCudaFree(state, input_b);
    THCudaFree(state, output_b);
    THCudaFree(state, columns_b);
    THCudaFree(state, ones_b);
    THCudaFree(state, weight_b);
    THCudaFree(state, bias_b);
    return output;
}


std::vector<at::Tensor> dcn_v2_cuda_backward(
    const at::Tensor &input,
    const at::Tensor &weight,
    const at::Tensor &bias,
    const at::Tensor &offset,
    const at::Tensor &mask,
    const at::Tensor &grad_output,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int deformable_group
)
{
    THArgCheck(input.is_contiguous(), 1, "input tensor has to be contiguous");
    THArgCheck(weight.is_contiguous(), 2, "weight tensor has to be contiguous");

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
    const int kernel_h_ = weight.size(2);
    const int kernel_w_ = weight.size(3);

    AT_ASSERTM(kernel_h_ == kernel_h && kernel_w_ == kernel_w,
        "Input shape and kernel shape wont match: (%d x %d vs %d x %d).", kernel_h_, kernel_w, kernel_h_, kernel_w_);

    AT_ASSERTM(channels == channels_kernel,
        "Input shape and kernel channels wont match: (%d vs %d).", channels, channels_kernel);

    const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

    auto ones = at::ones({height_out, width_out}, input.options());
    auto columns = at::empty({channels * kernel_h * kernel_w, 1 * height_out * width_out}, input.options());
    auto output = at::empty({batch, channels_out, height_out, width_out}, input.options());

    auto grad_input = at::zeros_like(input);
    auto grad_weight = at::zeros_like(weight);
    auto grad_bias = at::zeros_like(bias);
    auto grad_offset = at::zeros_like(offset);
    auto grad_mask = at::zeros_like(mask);

    using scalar_t = float;

    for (int b = 0; b < batch; b++)
    {
        // https://pytorch-cn.readthedocs.io/zh/latest/package_references/Tensor/#selectdim-index-tensor-or-number
        auto input_n = input.select(0, b);
        auto offset_n = offset.select(0, b);
        auto mask_n = mask.select(0, b);

        auto grad_output_n = grad_output.select(0, b);
        auto grad_input_n = grad_input.select(0, b);
        auto grad_offset_n = grad_offset.select(0, b);
        auto grad_mask_n = grad_mask.select(0, b);

        long m = channels * kernel_h * kernel_w
        long n = height_out * width_out;
        long k = channels_out;

        THCudaBlas_Sgemm(
            state,
            "n",
            "t",
            n,
            m,
            k,
            1.0f,
            grad_output_n.data<scalar_t>(),
            n,
            weight.data<scalar_t>(), 
            m, 
            0.0f,
            columns.data<scalar_t>(), 
            n
        )

        modulated_deformable_col2im_coord_cuda(
            THCState_getCurrentStream(state),
            columns.data<scalar_t>(),
            input_n.data<scalar_t>(),
            offset_n.data<scalar_t>(),
            mask_n.data<scalar_t>(),
            1, channels, height, width,
            height_out, width_out, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deformable_group,
            grad_offset_n.data<scalar_t>(),
            grad_mask_n.data<scalar_t>()
        );

        modulated_deformable_col2im_cuda(
            THCState_getCurrentStream(state),
            columns.data<scalar_t>(),
            offset_n.data<scalar_t>(),
            mask_n.data<scalar_t>(),
            1, channels, height, width,
            height_out, width_out, kernel_h, kernel_w,
            pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deformable_group,
            grad_input_n.data<scalar_t>()
        );
    }
}