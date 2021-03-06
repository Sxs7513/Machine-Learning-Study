import numpy as np
import matplotlib.pyplot as plt

class MaxPooling(object):
    def __init__(self, shape,  ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = np.zeros(shape)
        self.output_shape = [shape[0], int(shape[1] / self.stride), int(shape[2] / self.stride), self.output_channels]

    def forward(self, x):
        out = np.zeros([x.shape[0], int(x.shape[1] / self.stride), int(x.shape[2] / self.stride), self.output_channels])

        # 没什么好的办法,只能硬干
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, int(i / self.stride), int(j / self.stride), c] = np.max(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        # 记录最大值位置,在最大值位置处填充1,用于反向传播
                        # 原因看 https://www.jianshu.com/p/6928203bf75b
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        # 最大值位置直接为 1, 方便乘以梯度,
                        # index / self.stride 永远为 0, index % self.stride 在 index 为 1 的时候为 1
                        # 这是因为 np.argmax 给出的位置问题,自己打 log 就知道了
                        self.index[b, i + int(index / self.stride), int(j + index % self.stride), c] = 1

        return out

    def gradient(self, eta):
        # repeat 需要两次, 因为 repeat 只能同时操作一个 axis
        return np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2) * self.index

