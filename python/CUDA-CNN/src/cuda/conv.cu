#include "conv.cuh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdlib>
#include <memory>
#include <vector>
#include <cmath>


void operator_conv_bias(const Storage *inputs, const Storage *bias,
                        Storage *output) {
    CHECK_EQ(bias->get_data().size(), *(inputs->get_shape().begin() + 1),
           "operator_conv_bias: size error");
    
    const float *inputs_ptr = RAW_PTR(inputs->get_data());
    const float *bias_ptr = RAW_PTR(bias->get_data());
    float *output_ptr = RAW_PTR(output->get_data());

    int channel_in = *(inputs->get_shape().rbegin() + 2);
    int height = *(inputs->get_shape().rbegin() + 1);
    int width = *(inputs->get_shape().rbegin());

    int size = inputs->get_data().size();

}


__global__ void im2col_h(const int n, const float *data_im, const int height,
                         const int width, const int kernel_h,
                         const int kernel_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int height_col, const int width_col,
                         float *data_col, int im_stride, 
                         int col_stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        const int batch_idx = blockIdx.y;
        data_im += batch_idx * im_stride;
        data_col += batch_idx * col_stride;

        // index = bCHW + cHW + hW + w
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        // bC + c
        const int c_im = index / width_col / height_col;
        // (bC + c) * (kernel_h * kernel_w)
        const int c_col = c_im * kernel_h * kernel_w;
        
        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;

        float *data_col_ptr = data_col;
        data_col_ptr += (c_col * height_col + w_col) * width_col + w_col;
        const float *data_im_ptr = data_im;
        data_im_ptr += ((c_im * height) + h_in) * width + w_in;

        for (int i=0; i<kernel_h; ++i) {
            for (int j=0; j<kernel_w; ++j) {
                int h_im = h_in + i;
                int w_im = w_in + j;

                *data_col_ptr = 
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                        ? data_im_ptr[i * width + j]
                        : 0;
                data_col_ptr += height_col * width_col;
            }
        }

    }
};


void im2col(const float *data_im, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_col) {
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    int size = channels * height_col * width_col;

    int im_stride = channels * height * width;
    int col_stride = channels * kernel_h * kernel_w * height_col * width_col;
    dim3 dim_grid(ceil((float)size / BLOCK_SIZE), batch_size);

    im2col_h<<<dim_grid, BLOCK_SIZE>>>(
        size, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
        stride_w, height_col, width_col, data_col, im_stride, 
        col_stride
    );
};


void operator_conv(const Storage *inputs, Storage *filters, Storage *cols,
                   const int pad_h, const int pad_w, const int stride_h,
                   const int stride_w, Storage *output) {
    CHECK_EQ(inputs->get_shape().size(), 4, "operator_conv: inputs shape error");
    CHECK_EQ(filters->get_shape().size(), 4, "operator_conv: filters shape error");

    int batch_size = *(inputs->get_shape().rbegin() + 3);
    int channel_in = *(inputs->get_shape().rbegin() + 2);
    int height = *(inputs->get_shape().rbegin() + 1);
    int width = *(inputs->get_shape().rbegin());

    int channel_out = *(filters->get_shape().rbegin() + 3);
    int kernel_h = *(filters->get_shape().rbegin() + 1);
    int kernel_w = *(filters->get_shape().rbegin());

    CHECK_EQ(*(filters->get_shape().rbegin() + 2), channel_in,
           "operator_conv: channel size error");

    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    // im2col
    const float *inputs_ptr = RAW_PTR(inputs->get_data());
    const float *filters_ptr = RAW_PTR(filters->get_data());
    float *cols_ptr = RAW_PTR(cols->get_data());
    im2col(inputs_ptr, batch_size, channel_in, height, width, kernel_h, kernel_w,
         pad_h, pad_w, stride_h, stride_w, cols_ptr);
    
    filters->reshape({ channel_out, channel_in * kernel_h * kernel_w });
    
}


Conv::Conv(int height, int width, int channel_in, int channel_out, int kernel_h,
        int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
        bool is_bias)
    : height(height),
      width(width),
      channel_in(channel_in),
      channel_out(channel_out),
      kernel_h(kernel_h),
      kernel_w(kernel_w),
      pad_h(pad_h),
      pad_w(pad_w),
      stride_h(stride_h),
      stride_w(stride_w),
      is_bias(is_bias) {
    
    int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    this->filters.reset(
        new Storage({ channel_out, channel_in, kernel_h, kernel_w }));
    this->filters->xavier(
        channel_in * width * height, channel_out * width_out * height_out);
    this->filters_grad.reset(
        new Storage({ channel_out, channel_in, kernel_h, kernel_w }));
    
    if (is_bias) {
        this->bias.reset(new Storage({ 1, channel_out }));
        this->bias_grad.reset(new Storage({ 1, channel_out }));
        this->bias->xavier(channel_in * width * height, channel_out * height_out * width_out);
    }
}

std::vector<std::pair<Storage *, Storage *>> Conv::parameters() {
    if (this->is_bias) {
        return {
            std::make_pair(this->filters.get(), this->filters_grad.get()),
            std::make_pair(this->bias.get(), this->bias_grad.get())
        };
    } else {
        return {
            std::make_pair(this->filters.get(), this->filters_grad.get())            
        };
    }
}

 
void Conv::forward() {
    const Storage *input = this->pre->get_output();
    int height_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    std::vector<int> output_shape{
        input->get_shape()[0], channel_out,
        height_out, width_out
    };
    std::vector<int> cols_shape{
        input->get_shape()[0],
        channel_in * kernel_h * kernel_w,
        height_out * width_out
    };

    INIT_STORAGE(this->output, output_shape);
    INIT_STORAGE(this->cols, cols_shape);

    operator_conv(
        input, this->filters.get(), this->cols.get(),
        pad_h, pad_w, stride_h, stride_w, this->output.get()        
    );

    if (this->is_bias) {

    }

}