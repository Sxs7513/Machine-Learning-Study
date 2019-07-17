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


__device__ flaot dmcn_im2col_bilinear(
    const float *bottom_data,
    const int data_width,
    const int height,
    const int width,
    float h,
    float w
)
{
    int h_low = floor(h);
    int w_low = floor(W);

    int h_high = h_low + 1;
    int w_hight = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = 0;
    if (h_low > 0 && w_low > 0)
        v1 = bottom_data[h_low * data_width + w_low];

    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low * data_width + w_high];

    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high * data_width + w_low];
    
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high * data_width + w_high];

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val
}


__global__ void modulated_deformable_col2im_coord_gpu_kernel(
    const int n,
    const float *data_col, 
    const float *data_im,
    const float *data_offset, 
    const float *data_mask,
    const int channels, 
    const int height, 
    const int width,
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
    const int offset_channels, 
    const int deformable_group,
    const int height_col, 
    const int width_col,
    float *grad_offset, 
    float *grad_mask
)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        float val = 0, mval = 0;
        // offset-size => [batch_size, offset_channels, height_col, width_col]
        // b * C * H * W + c * H * W + h * W + w
        int w = index % width_col;
        int h = (index / width_col) % height_col;
        int c = (index / width_col / height_col) % offset_channels;
        int b = (index / width_col / height_col) / offset_channels;

        // when deformable_group = 1, deformable_group_index is always 0
        const int deformable_group_index = c / (2 * kernel_h * kernel_w);
        const int col_step = kernel_h * kernel_w;
        int cnt = 0;

        const float *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
        const float *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channels / deformable_group * height * width;
        const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
        
        // 
        const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

        // colc += col_step means different channel of im2col
        // channel_per_deformable_group => channels / deformabel_groip * kernel_w * kernel_h
        // the initValue of col_c is (offset_c / 2), because it means the value of kernel_h kernel_w in im2col
        for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
        {
            // im2col => [batch_size * channels * kernel_h * kernel_w, height_col * width_col]
            // 
            const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
            const int bp_dir = offset_c % 2;

            // offset => [batch_size, offset_channels(2 * kernel_w * kernel_h * deformalble_group), height_col, width_col]
            int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
            int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;

            int w_out = col_pos % width_col;
            int h_out = (col_pos / width_col) % height_col;

            int w_in = w_out * stride_w - pad_w;
            int h_in = h_out * stride_h - pad_h;

            const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
            const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
            const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);

            const float offset_h = data_offset_ptr[data_offset_h_ptr];
            const float offset_w = data_offset_ptr[data_offset_w_ptr];
            const float mask = data_mask_ptr[data_mask_hw_ptr];

            float inv_h = h_in + i * dilation_h + offset_h;
            float inv_w = w_in + j * dilation_w + offset_w;

            if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
            {
                inv_h = inv_w = -2;
            }
            else
            {
                // mask-gradient 
                mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
            }
            
            const float weight = dmcn_get_coordinate_weight(
                inv_h,
                inv_w,
                height,
                width,
                data_im_ptr + cnt * height * width, 
                width, 
                bp_dir
            )
            val += data_col_ptr[col_pos] * weight * mask;
            cnt += 1
        }
    }
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

        const float *data_input_ptr = data_input + (b_index * in_channels + c_in_index) * height_out * width_out;
        const float *data_offset_ptr = data_offset + (b_index * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_in * width_in;
        const float *data_mask_ptr = data_mask + (b_index * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_out * width_out;
        
        for (int i=0; i<kernel_h; ++i)
        {
            for (int j=0; j<kernel_w; ++j)
            {
                const int data_offset_h_pos = ((2 * (i * kernel_w + j)) * height_out) + h_coord_out) width_out + w_coord_out;
                const int data_offset_w_pos = ((2 * (i * kernel_w + j) + 1) * height_out + h_coord_out) * width_out + w_coord_out;
                const int data_mask_hw_pos = ((i * kernel_w + j) * height_out + h_coord_out) * width_out + w_coord_out;

                const float offset_h = data_offset_ptr[data_offset_h_pos];
                const float offset_w = data_offset_ptr[data_offset_w_pos];
                const float mask = data_mask_ptr[data_mask_hw_pos];

                float val = static_cast<float>(0);

                const float h_input = h_coord_in + i * dilation_h + offset_h;
                const float w_input = w_coord_in + j * dilation_w + offset_w;

                if (h_input > - 1 && w_input > -1 && h_coord_in < height && w_im < width_in)
                {
                    val = dmcn_im2col_bilinear(data_input_ptr, width, height, wdith, h_input, w_input);
                }
                *data_col_ptr = val * mask;
                data_col_ptr += height_out * width_out;
            }
        }
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
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
    }
}


void modulated_deformable_col2im_coord_cuda(
    cudaStream_t stream,
    const float* data_col,
    const float* data_im,
    const float* data_offset,
    const float* data_mask,
    const int height_col,
    const int channels,
    const int height_im,
    const int width_im,
    const int height_col, 
    const int width_col, 
    const int kernel_h, 
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h, 
    const int stride_w,
    const int dilation_h, 
    const int dilation_w,
    const int deformable_group,
    float* grad_offset,
    float* grad_mask
)
{
    const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
    const int channel_per_deformable_group = channels / deformable_group * kernel_h * kernel_w;

    modulated_deformable_col2im_coord_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels, 
        data_col, 
        data_im, 
        data_offset, 
        data_mask, 
        channels, 
        height_im, 
        width_im,
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
        2 * kernel_h * kernel_w * deformable_group, 
        deformable_group, 
        height_col, 
        width_col, 
        grad_offset, 
        grad_mask
    )
}
