#include <cstdio>
#include <algorithm>
#include <cstring>

#define CUDA_KERNEL_LOOP(i, n)                          \      
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__global__ void modulated_deformable_im2col_gpu_kernel(
    const int n,
    const float *data_input,
    const float *data_offset,
    const float *data_mask,
    const int height_in,
    const int width_in,
    const int kernel_h,
    const int kernel_w,
    const int pad_h, 
    const int pad_w,
    const int stride_h, 
    const int stride_w,
    const int dilation_h, 
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size, 
    const int in_channels, 
    const int deformable_group,
    const int height_out, 
    const int width_out,
    float *data_col
)
{   
    // cores num => batch_size * in_channels * height_out * width_out
    CUDA_KERNEL_LOOP(index, n)
    {
        const int w_coord_out = index % width_out;
        const int h_coord_out = (index / width_out) % height_out;
        const int b_index = (index / width_out / height_out / in_channels) % batch_size;
        const int c_in_index = (index / width_out / height_out) % in_channels;
        // which kernel in im2col
        const int kernel_index = c_in_index * kernel_h * kernel_w;

        const int deformable_group_index = c_in_index / channel_per_deformable_group;

        const int h_coord_in = h_coord_out * stride_h - pad_h;
        const int w_coord_in = w_coord_out * stride_w - pad_w;

        float *data_col_pos = data_col + \
            (b_index * in_channels * kernel_w * kernel_h + c_in_index) \ 
            * height_out + h_coord_out) * width_out + w_coord_out;
    }
}


void modulated_deformable_im2col_cuda(
    cudaStream_t stream,
    const float *data_input,
    const float *data_offset,
    const float *data_mask,
    const int batch_size,
    const int in_channels,
    const int height_in,
    const int width_in,
    const int height_out,
    const int width_out,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h, 
    const int stride_w,
    const int dilation_h, 
    const int dilation_w,
    const int deformable_group, 
    float* data_col
)
{
    const int channel_per_deformable_group = channels / deformable_group;
    
    const int num_kernels = batch_size * channels * height_out * width_out;

    modulated_deformable_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
    0, stream>>>(
        num_kernels,
        data_input,
        data_offset,
        data_mask,
        height_in,
        width_in,
        kernel_h,
        kernel_w,
        pad_h, 
        pad_w, 
        stride_h, 
        stride_w, 
        dilation_h, 
        dilation_w, 
        channel_per_deformable_group,
        batch_size, 
        in_channels, 
        deformable_group, 
        height_out, 
        width_out, 
        data_col
    )
}