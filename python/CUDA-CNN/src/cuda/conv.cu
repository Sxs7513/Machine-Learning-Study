#include <conv.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdlib>
#include <memory>
#include <vector>


void operator_conv(const Storage *inputs, Storage *filters, Storage *cols,
                   const int pad_h, const int pad_w, const int stride_h,
                   const int stride_h, const int stride_w) {
    CHECK_EQ(inputs->get_shape().size(), 4, "operator_conv: inputs shape error");
    CHECK_EQ(filters->get_shape().size(), 4, "operator_conv: filters shape error");

    int batch_size = *(inputs->get_shape().rbegin() + 3);
    int channel_in = *(inputs->get_shape().rbegin() + 2);
    int height = *(inputs->get_shape().rbegin() + 1);
    int width = *(inputs->get_shape().rbegin());
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
        new Storage({ channel_out, channel_in, kernel_h, kernel_w }))
    
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
        channel_in * kernel_h * kernel_w
        height_out * width_out
    };

    INIT_STORAGE(this->output, output_shape);
    INIT_STORAGE(this->cols, cols_shape);

    operator_conv(
        input, this->filters.get(), this->cols.get(),
        pad_h, pad_w, stride_h, stride_w, this->output.get()        
    );

}