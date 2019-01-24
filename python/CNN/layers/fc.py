import numpy as np
from functools import reduce
import math

class FullyConnect(object):
    def __init__(self, shape, output_num=10):
        self.input_shape = np.array(shape).astype(np.int32)
        self.batchsize = shape[0]
        
        # 计算某个样本的全连接层神经元个数
        input_len = int(reduce(lambda x, y: x * y, shape[1:]))

        # 神经元连接输出层的初始权重矩阵
        self.weights = np.random.standard_normal((input_len, output_num)) / 100
        self.bias = np.random.standard_normal(output_num) / 100

        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        # 将每个图片打平, 方便进行全连接
        self.x = x.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.weights) + self.bias

        return output

    def gradient(self, eta):
        for i in range(0, self.batchsize):
            col_x = self.x[i][:, None]
            eta_i = eta[i][:, None].T
            # 对权重求导较为简单,没啥可说
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)

        # 对该层输入的求导也较为简单,没啥可说
        next_eta = np.dot(eta, self.weights.T)
        # 还原成该层上一层(可能是池化)输入时候的 shape,方便池化求导
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay 是 L2 正则化的系数
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)

        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
