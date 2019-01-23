from loadData import load_mnist
from layers.base_conv import Conv2D
from layers.relu import Relu
from layers.pooling import MaxPooling

def main():
    images, labels = load_mnist('./data/mnist')
    test_images, test_labels = load_mnist('./data/mnist', 't10k')
    
    batch_size = 64
    conv1 = Conv2D(shape = [batch_size, 28, 28, 1], output_channels = 12, ksize = 5, stride = 1)
    relu1 = Relu(conv1.output_shape)
    pool1 = MaxPooling(relu1.output_shape)

    # just for test, not final code
    i = 0
    # 每个图片被重新排成 28 * 28 * 1，即 28 个28 * 1 的矩阵是一张图片
    # 为的是方便在 Conv2D 的 im2col 计算
    img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
    label = labels[i * batch_size:(i + 1) * batch_size]
    # 前向传播
    # 卷积层的提取
    conv_extract = conv1.forward(img)
    # 卷积层的提取经过 relu 的激活
    conv_out = relu1.forward(conv_extract)
    pool1.forward(conv_out)
    
main()