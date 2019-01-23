import numpy as np
from functools import reduce
import math

class Conv2D(object):
    # shape = [N,W,H,C] N=Batchsize/ W=width / H=height / C=channels
    # ouput_channels 卷积核个数(很容易引起歧义啊。。)，ksize 卷积核尺寸，stride 卷积的步长
    # method 卷积的方法，即是否通过 padding 保持输出图像与输入图像的大小不变
    def __init__(self, shape, output_channels, ksize = 3, stride = 1, method="VALID"):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        # Batchsize 为每次训练的样本个数
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method

        # batch_size * width * height * channel 代表每次训练训练集有多少像素
        # msra 方法
        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)
        # 初始化卷积核矩阵，矩阵要做成这样的原因是为了提高运算性能，具体可以看 https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf
        # 里面最下面提供的图，对应的就是图里 kernel-Matrix，当然不是完全一致的，理解就好
        # 反正记住一点，卷积矩阵被打开成一维的了，这是为了方便计算
        self.weights = np.random.standard_normal((ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        # eta 用于存储 backward 传回来的损失函数对该层的输出的导数，同时它与该层的 out 的维度一致
        # 同样可以看上面给的链接里面的图
        if method == 'VALID':
            # 不加 padding
            self.eta = np.zeros(
                (   
                    self.batchsize,
                    # 
                    int((shape[1] - ksize + 1) / self.stride),
                    int((shape[1] - ksize + 1) / self.stride),
                    self.output_channels
                )
            )
        
        # 
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

    def forward(self, x):
        # 打掉一个维度，变成 25 * 12 的矩阵(假设卷积核长宽为5，个数为12)，一列代表一个卷积核
        # 正好是上面链接里给的图里的 kernel-Matrix
        col_weights = self.weights.reshape([-1, self.output_channels])

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)

        # just for test. not final code
        for i in range(0, self.batchsize):
            img_i = x[i][None, :]
            # 经过此步骤后，某张图片已被处理成链接里图中的 Input features
            col_image_i = im2col(img_i, self.ksize, self.stride)
            # dot 是用来进行图里的点乘操作的，然后将其还原为类似于参数 x 里的这样形式
            # 目的是方便之后的池化层操作
            conv_out[i] = np.reshape(np.dot(col_image_i, col_weights), self.eta[0].shape)
            self.col_image.append(col_image_i)
        
        self.col_image = np.array(self.col_image)
        return conv_out
        


def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize + 1, stride):
        for j in range(0, image.shape[2] - ksize + 1, stride):
            # 取出这一对应卷积核的所有元素并打平
            col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
            image_col.append(col)

    return np.array(image_col)