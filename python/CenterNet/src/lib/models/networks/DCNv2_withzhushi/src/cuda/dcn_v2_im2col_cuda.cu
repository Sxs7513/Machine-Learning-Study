#include "dcn_v2_im2col_cuda.h"
#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  // 第几个 block * block 的大小 + block中第几个线程       
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      // 每次加所有线程的数量，因为有时候 n 会大于线程数量
      // 所以可能有些线程要串行
      i += blockDim.x * gridDim.x)

// 每个block中有多少个线程
const int CUDA_NUM_THREADS = 1024;
// 计算 block 的数量
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// 由于偏移很难是整数，所以采用双线性差值取到偏移之后的值
// https://blog.csdn.net/xbinworld/article/details/65660665
__device__ float dmcn_im2col_bilinear(const float *bottom_data, const int data_width,
                                      const int height, const int width, float h, float w)
{
  // 向下取整
  int h_low = floor(h);
  int w_low = floor(w);
  // 到此左上角和右下角的位置都以及确定
  // 那么周围四个点的位置也都确定了下来
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  // 左上 h_low， w_low 对应的点，如果超出边界则为 0
  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  // 右上 h_low, w_high, 如果超出边界则为 0
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  // 左下
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  // 右下
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];
  // 双线性差值的权重
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

// 双线性差值方向传播，对四周的点的梯度
__device__ float dmcn_get_gradient_weight(float argmax_h, float argmax_w,
                                          const int h, const int w, const int height, const int width)
{
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  // 对左上角的梯度
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

// 双线性差值反向传播， 求对差值位置的梯度
__device__ float dmcn_get_coordinate_weight(float argmax_h, float argmax_w,
                                            const int height, const int width, const float *im_data,
                                            const int data_width, const int bp_dir)
{
  // 偏移出输入的没有梯度
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;

  // 对 h  的梯度
  if (bp_dir == 0)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }
  // 对 w 的梯度
  else if (bp_dir == 1)
  {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

__global__ void modulated_deformable_im2col_gpu_kernel(const int n,
                                                       const float *data_im, const float *data_offset, const float *data_mask,
                                                       const int height, const int width, const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int num_channels, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *data_col)
{
  // launch channels * batch_size * height_col * width_col cores
  CUDA_KERNEL_LOOP(index, n)
  {
    // NOTE(CharlesShang): different from Dai Jifeng's MXNet implementation, col_buffer is of shape (c*kw*kh, N, oh, ow)
    // here columns is of shape (N, c*kw*kh, oh * ow), need to adapt axis
    // data_col => [N, c*kw*kh, oh * ow]
    // data_offset => [N, 2 * kernel_h * kernel_w * 输入通道分成的份数(deformable_group), oh * ow]
    // data_mask_ptr => [N, 1 * kernel_h * kernel_w * 输入通道分成的份数(deformable_group), oh * ow]

    // index 代表的没有特别的意义
    // https://www.zhihu.com/question/279648139
    // https://zhuanlan.zhihu.com/p/58185157
    // https://blog.csdn.net/sinat_22336563/article/details/69808612
    // https://blog.csdn.net/hujingshuang/article/details/53097222

    // index index of output matrix
    // 该线程对应的输出特征图的 w 坐标
    const int w_col = index % width_col;
    // 该线程对应的输出特征图的 h 坐标
    const int h_col = (index / width_col) % height_col;
    // const int b_col = (index / width_col / height_col) % batch_size;
    // 该线程对应的输入的第几个 batch
    const int b_col = (index / width_col / height_col / num_channels) % batch_size;
    // const int c_im = (index / width_col / height_col) / batch_size;
    // 该线程对应输入的第几个 channel
    const int c_im = (index / width_col / height_col) % num_channels;
    // const int c_col = c_im * kernel_h * kernel_w;
    // 该线程对应 im2col 的某个 kernel
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    // 输出特征图的 h w 坐标反推出对应的输入特征图上的坐标
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    // float *data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    // data_col 是数组指针(matrix 本质是数组), 所以修改它的指向, 指向 im2col 矩阵某个 kernel 的起始点
    // im2col 行数是 kernel-size, 列数实际上就是kernel在图像上滑动的次数即输出图的size
    float *data_col_ptr = data_col + ((b_col * num_channels * kernel_w * kernel_h + c_col) * height_col + h_col) * width_col + w_col;
    //const float* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    // 同上, 指向原图对应 channel 特征图的左上角
    const float *data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    // 在 deformable_group 等于 1 的时候, 只是定位到某个 batch
    const float *data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    // 同上
    const float *data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    // 在该线程循环, 完成 im2col 某一"列"的生成
    for (int i = 0; i < kernel_h; ++i)
    {
      for (int j = 0; j < kernel_w; ++j)
      {
        // kernel 里该点的 offset 的内存偏移, 这个较为简单
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        // 获得该点偏移的值
        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        // 该点偏移的置信度
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        // 确定偏移后的点在原图上面的位置
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
        {
          //const float map_h = i * dilation_h + offset_h;
          //const float map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          // 双线性差值求到偏移后最终的值
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        // 乘置信度
        *data_col_ptr = val * mask;
        // data_col_ptr += batch_size * height_col * width_col;
        // 该点结束后，移动到下个点
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

__global__ void modulated_deformable_col2im_gpu_kernel(const int n,
                                                       const float *data_col, const float *data_offset, const float *data_mask,
                                                       const int channels, const int height, const int width,
                                                       const int kernel_h, const int kernel_w,
                                                       const int pad_h, const int pad_w,
                                                       const int stride_h, const int stride_w,
                                                       const int dilation_h, const int dilation_w,
                                                       const int channel_per_deformable_group,
                                                       const int batch_size, const int deformable_group,
                                                       const int height_col, const int width_col,
                                                       float *grad_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    // kernel_w 与 kernel_h 上的位置
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    // 对应输入的哪个通道
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output

    // 暂时不用考虑，默认为 c
    const int deformable_group_index = c / channel_per_deformable_group;

    // 对应输出的点的位置
    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    // 对应的输入的点的位置
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    // im2col 上面该点对应的偏移与置信度位置
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float mask = data_mask_ptr[data_mask_hw_ptr];
    // 输入上的点偏移后的位置
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;

    // 
    const float cur_top_grad = data_col[index] * mask;
    // 输入上的点偏移后的位置全部转换为整数，因为
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    // 前向传播的时候，im2col 上每个点的值是由偏移后的点双线性差值得来的，所以在计算 input 的该点梯度的时候
    // 其实要扩散到它四周的点，对它们均有梯度的
    for (int dy = -2; dy <= 2; dy++)
    {
      for (int dx = -2; dx <= 2; dx++)
      {
        // 四周的点在原图内，并且是在紧邻的四周
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1)
        {
          // 四周的某个点的相对于 input 起始的内存偏移
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          // 对四周某个点的梯度
          float weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          // https://www.cnblogs.com/biglucky/p/4283476.html
          // 需要原子操作，因为对 input 梯度的特殊性，input 上面每个点从不同的线程都可能传过来梯度
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

__global__ void modulated_deformable_col2im_coord_gpu_kernel(const int n,
                                                             const float *data_col, const float *data_im,
                                                             const float *data_offset, const float *data_mask,
                                                             const int channels, const int height, const int width,
                                                             const int kernel_h, const int kernel_w,
                                                             const int pad_h, const int pad_w,
                                                             const int stride_h, const int stride_w,
                                                             const int dilation_h, const int dilation_w,
                                                             const int channel_per_deformable_group,
                                                             const int batch_size, const int offset_channels, const int deformable_group,
                                                             const int height_col, const int width_col,
                                                             float *grad_offset, float *grad_mask)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    float val = 0, mval = 0;
    // https://www.zhihu.com/question/279648139，原理一样的
    // 该线程对应的 offset 特征图的 w 坐标
    int w = index % width_col;
    // 该线程对应的 offset 特征图的 h 坐标
    int h = (index / width_col) % height_col;
    // 该线程对应的 offset 特征图的第几个通道
    int c = (index / width_col / height_col) % offset_channels;
    // batch，暂时不用考虑，为 0
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    // 默认为 0(c < (2 * kernel_h * kernel_w), / 得到的肯定是 0)，暂时不用考虑，
    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    // 
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    // 暂时不用考虑
    const float *data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    // 暂时不用考虑
    const float *data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    // 暂时不用考虑
    const float *data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    // 暂时不用考虑
    const float *data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    // 暂时不用考虑，等于 c
    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    // 首先说明 channel_per_deformable_group = inchannels * kernel_h * kernel_w / deformable_group
    // data_col 即 im2col => [N(1), inChannels * kernel_h * kernel_w, out_w * out_h]
    // 循环这样做的原因是在前向传播中input不同通道的同一位置用的是同一个偏移，但是在反向传播的时候，通过 im2col 的梯度
    // 来找偏移的梯度的时候, 输入的每一个通道的相同位置的梯度都要传播到该偏移上面
    // offset_c / 2 是因为 im2col 上面一个点对应的是俩偏移，它代表的是 kernel 上哪个点
    // 不断累加 col_step 即不断的往下个 channel 跳(注意看 im2col 的形状)
    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step)
    {
      // 该偏移的偏移对相对于 im2col 起始的内存偏移，该公式来的很及时，下面的那些都可以用该公式来推哈哈
      // col_c 即等于 (c(准确的点) * kw(准确的点) * kh(准确的点))
      // 不考虑batch的情况下可以简化成 col_pos = ((col_c * height_col) + h) * width_col + w;
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      // 代表该点是 x偏移还是y偏移
      const int bp_dir = offset_c % 2;
      
      // i j 表示该点是 kernel 里面的哪一块的偏移
      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      // 该偏移对对应输出上的 w h 位置
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      // 该偏移对对应输入上 w h 位置
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      // 取到偏移对的值
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      const float mask = data_mask_ptr[data_mask_hw_ptr];
      // 输入上的点被偏移后的位置
      float inv_h = h_in + i * dilation_h + offset_h;
      float inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width)
      {
        inv_h = inv_w = -2;
      }
      else
      {
        // 对 mask 的梯度，data_im_ptr + cnt * height * width 代表input不同通道在内存中的起始位置
        // 梯度等于 im2col 上对应的点的梯度与双线性差值的结果乘机
        mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
      }
      // 获得input某通道该点对该偏移的梯度，注意在计算梯度的时候需要 x 和 y
      // 的偏移，但是最终获得的梯度是针对 x 或者 y 的哦（根据 bp_dir 判断）， 仔细想一下
      const float weight = dmcn_get_coordinate_weight(
          inv_h, inv_w,
          height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    // 所有 channel 对该偏移的梯度都加过来
    grad_offset[index] = val;
    // 一个 mask 对应俩偏移，所以在计算第一个偏移的时候赋值梯度即可
    if (offset_c % 2 == 0)
      // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w] = mval;
  }
}

void modulated_deformable_im2col_cuda(cudaStream_t stream,
  const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w,
  const int deformable_group, float* data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  // 该 kernel 非卷积的核的意思哦. 而是 cuda 核的意思, 这里计算总共应该有多少线程
  // 输入的通道数 * N * 输出特征图的长 * 输出特征图的宽, 注意相对于 im2col 矩阵
  // kernel_w 与 kernel_h 没有算进去, 因为一个线程里面要进行循环，计算该点对应的 kernel
  const int num_kernels = channels * batch_size * height_col * width_col;
  modulated_deformable_im2col_gpu_kernel
      // 
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
      num_kernels, data_im, data_offset, data_mask, height_im, width_im, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, channel_per_deformable_group,
      batch_size, channels, deformable_group, height_col, width_col, data_col);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

void modulated_deformable_col2im_cuda(cudaStream_t stream,
  const float* data_col, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int deformable_group, float* grad_im){

  const int channel_per_deformable_group = channels / deformable_group;
  // 核心数是 im2col 的参数数量
  const int num_kernels = channels * kernel_h * kernel_w * batch_size * height_col * width_col;
  modulated_deformable_col2im_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
          0, stream>>>(
        num_kernels, data_col, data_offset, data_mask, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_h, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, deformable_group, height_col, width_col, grad_im);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}

void modulated_deformable_col2im_coord_cuda(cudaStream_t stream,
  const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
  const int batch_size, const int channels, const int height_im, const int width_im, 
  const int height_col, const int width_col, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
  const int dilation_h, const int dilation_w, 
  const int deformable_group,
  float* grad_offset, float* grad_mask) {
  // 线程数为 offset 参数数量，mask 不需要额外的线程
  const int num_kernels = batch_size * height_col * width_col * 2 * kernel_h * kernel_w * deformable_group;
  const int channel_per_deformable_group = channels * kernel_h * kernel_w / deformable_group;
  modulated_deformable_col2im_coord_gpu_kernel
      <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS,
        0, stream>>>(
        num_kernels, data_col, data_im, data_offset, data_mask, channels, height_im, width_im,
        kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, channel_per_deformable_group,
        batch_size, 2 * kernel_h * kernel_w * deformable_group, deformable_group, height_col, width_col, 
        grad_offset, grad_mask);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in modulated_deformable_col2im_coord_cuda: %s\n", cudaGetErrorString(err));
  }
}