import numpy as np
from loadData import load_mnist
from layers.base_conv import Conv2D
from layers.relu import Relu
from layers.pooling import MaxPooling
from layers.fc import FullyConnect
from layers.softmax import Softmax

import time
import pickle

def build_model():
    images, labels = load_mnist('./data/mnist')
    test_images, test_labels = load_mnist('./data/mnist', 't10k')
    testImg = test_images[0]
    
    batch_size = 64
    # 初始化各个层, 主要是确定它们的大小
    conv1 = Conv2D(shape = [batch_size, 28, 28, 1], output_channels = 12, ksize = 5, stride = 1)
    relu1 = Relu(conv1.output_shape)
    pool1 = MaxPooling(relu1.output_shape)
    conv2 = Conv2D(pool1.output_shape, 24, 3, 1)
    relu2 = Relu(conv2.output_shape)
    pool2 = MaxPooling(relu2.output_shape)
    fc = FullyConnect(pool2.output_shape, 10)
    sf = Softmax(fc.output_shape)

    for epoch in range(20):

        learning_rate = 1e-4

        batch_loss = 0
        batch_acc = 0

        val_acc = 0
        val_loss = 0

        # train
        train_acc = 0
        train_loss = 0

        for i in range(int(images.shape[0] / batch_size)):
            # 每个图片被重新排成 28 * 28 * 1，即 28 个28 * 1 的矩阵是一张图片
            # 为的是方便在 Conv2D 的 im2col 计算
            img = images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
            label = labels[i * batch_size:(i + 1) * batch_size]
            # 前向传播
            # 卷积层的提取
            conv1_extract = conv1.forward(img)
            # 卷积层的提取经过 relu 的激活
            conv1_out = relu1.forward(conv1_extract)
            # 第一层池化层
            pool1_out = pool1.forward(conv1_out)
            # 第二层卷积层提取
            conv2_extract = conv2.forward(pool1_out)
            # 第二层卷积层输出
            conv2_out = relu2.forward(conv2_extract)
            # 第二层池化层输出
            pool2_out = pool2.forward(conv2_out)
            # 全连接层输出,至此前向传播已经结束
            fc_out = fc.forward(pool2_out)
            
            # 计算误差,正确率等
            batch_loss += sf.cal_loss(fc_out, np.array(label))
            for j in range(batch_size):
                if (np.argmax(sf.softmax[j]) == label[j]):
                    batch_acc += 1

            # 反向传播
            # 首先进行对各个层输入与权重(有必要的话)的求导
            sf.gradient()
            conv1.gradient(
                relu1.gradient(
                    pool1.gradient(
                        conv2.gradient(
                            relu2.gradient(
                                pool2.gradient(
                                    fc.gradient(
                                        sf.eta
                                    )
                                )
                            )
                        )
                    )
                )
            )


            if i % 1 == 0:
                # 然后更新权重
                fc.backward(alpha=learning_rate, weight_decay=0.0004)
                conv2.backward(alpha=learning_rate, weight_decay=0.0004)
                conv1.backward(alpha=learning_rate, weight_decay=0.0004)

            
            if i % 50 == 0:
                print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + \
                        "  epoch: %d ,  batch: %5d , avg_batch_acc: %.4f  avg_batch_loss: %.4f  learning_rate %f" % (epoch,
                                                                                                    i, batch_acc / float(
                            batch_size), batch_loss / batch_size, learning_rate))

            # 记住要归零
            batch_loss = 0
            batch_acc = 0

        print (time.strftime("%Y-%m-%d %H:%M:%S",
                            time.localtime()) + "  epoch: %5d , train_acc: %.4f  avg_train_loss: %.4f" % (
            epoch, train_acc / float(images.shape[0]), train_loss / images.shape[0]))

        # validation
        for i in range(int(test_images.shape[0] / batch_size)):
            img = test_images[i * batch_size:(i + 1) * batch_size].reshape([batch_size, 28, 28, 1])
            label = test_labels[i * batch_size:(i + 1) * batch_size]
            conv1_out = relu1.forward(conv1.forward(img))
            pool1_out = pool1.forward(conv1_out)
            conv2_out = relu2.forward(conv2.forward(pool1_out))
            pool2_out = pool2.forward(conv2_out)
            fc_out = fc.forward(pool2_out)
            val_loss += sf.cal_loss(fc_out, np.array(label))

            for j in range(batch_size):
                if np.argmax(sf.softmax[j]) == label[j]:
                    val_acc += 1

        print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  epoch: %5d , val_acc: %.4f  avg_val_loss: %.4f" % (
            epoch, val_acc / float(test_images.shape[0]), val_loss / test_images.shape[0]))

    def save_model():
        weight1 = conv1.weights
        weight2 = conv2.weights
        weight3 = fc.weights

        models = {
            "conv1": conv1,
            "conv2": conv2,
            "relu1": relu1,
            "relu2": relu2,
            "pool1": pool1,
            "pool2": pool2,
            "fc": fc,
            "sf": sf,
            "testImg": testImg
        }

        output = open('./data/model/index.pkl', 'wb')
        
        pickle.dump(models, output)

        output.close()

        np.save('./data/model/weight1.npy', weight1)
        np.save('./data/model/weight2.npy', weight2)
        np.save('./data/model/weight3.npy', weight3)

    save_model()

    return (conv1, conv2, pool1, pool2, fc, sf, relu1, relu2, testImg)

def predict(model):
    conv1, conv2, pool1, pool2, fc, sf, relu1, relu2, testImg = model

    img = testImg
    img = np.array([img]).reshape([1, 28, 28, 1])
    conv1_out = relu1.forward(conv1.forward(img))
    pool1_out = pool1.forward(conv1_out)
    conv2_out = relu2.forward(conv2.forward(pool1_out))
    pool2_out = pool2.forward(conv2_out)
    fc_out = fc.forward(pool2_out)
    sf.predict(fc_out)
    print(np.argmax(sf.softmax))
    return np.argmax(sf.softmax)