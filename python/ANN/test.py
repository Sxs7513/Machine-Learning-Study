import numpy as np
from sklearn import * 
import matplotlib.pyplot as plt

np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)

# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()

# logistic 并不适用于这个，验证如下
# clf = linear_model.LogisticRegressionCV()
# clf.fit(X, y)

# 咱们先定一个函数来画决策边界
def plot_decision_boundary(pred_func):
 
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 用预测函数预测一下
    # np.c_ 作用为
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 然后画出图
    # 先绘制等高线图，即把边界绘制出来
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    # 然后把原始散点绘制上去
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# plot_decision_boundary(lambda x: clf.predict(x))
# plt.title("Logistic Regression")
# # 发现逻辑分类只能线性分类，并不能捕捉到数据的“月形”特征
# plt.show()

# 样本数
num_examples = X.shape[0]
# 输入层节点数量
nn_input_dim = X.shape[1]
# 输出层数量
nn_output_dim = 2
epsilon = 0.01
reg_lambda = 0.01

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 前向计算一遍预测值
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # 计算误差
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    # 可选项,加上正则项防止过拟合
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return 1. / num_examples * data_loss

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 前向计算一遍预测值
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs, axis=1)

# nn_hdim 为隐藏层节点的数量
# num_passes 最大训练次数
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    np.random.seed(0)
    # 初始 W1 W2 b1 b2 矩阵，随机生成
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    for i in range(0, num_passes):
        # 前向传播
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        # 输出层接受到的值
        z2 = a1.dot(W2) + b2

        # 输出层使用 softmax 来计算 loss
        # https://segmentfault.com/a/1190000010933271
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 反向传播
        delta3 = probs
        # 计算 softmax 损失函数对于输出层的求导
        # https://blog.csdn.net/fireflychh/article/details/73794270
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        # 计算 softmax 损失函数对于隐藏层的求导
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print ("Loss after iteration %i: %f" %(i, calculate_loss(model)))

    return model

model = build_model(3, print_loss=True) 
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()