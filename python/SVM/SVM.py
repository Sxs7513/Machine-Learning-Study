import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt


def loadDataSet():
    dataFile = 'nonlinearData'
    data = scio.loadmat(dataFile)
    dataset = data['nonlinear'].T
    dataset = dataset[2300:2500, :]

    #+1的样本集, -1的样本集
    positive = np.array([[0, 0, 0]])
    negative = np.array([[0, 0, 0]])

    for i in range(dataset.shape[0]):
        if (dataset[i][2] == 1):
            positive = np.row_stack((positive, np.array([dataset[i]])))
        else:
            negative = np.row_stack((negative, np.array([dataset[i]])))

    return positive[1:, :], negative[1:, :], dataset


# 线性核函数
# def kernel(xi, xj):
#     return xi.dot(xj.T)


# 高斯核函数
def kernel(xi, xj):
    M = xi.shape[0]
    K = np.zeros((M, 1))

    for l in range(M):
        A = np.array([xi[l]]) - xj
        K[l] = [np.exp(-0.5 * float(A.dot(A.T)) / (sigma**2))]

    return K


class SVM(object):
    def __init__(self, X, Y, C, epsilon):
        self.X = X
        self.Y = Y
        # 训练集大小
        self.N = X.shape[0]
        # 惩罚系数
        self.C = C
        # 容错率
        self.epsilon = epsilon
        # 拉格朗日乘子
        self.alpha = np.zeros((self.N, 1))
        # 超平面 b
        self.b = 0
        # 误差缓存表 N*2，第一列为更新状态（0-未更新，1-已更新），第二列为缓存值
        self.E = np.zeros((self.N, 2))

    # 计算某个 alpha 所对应的数据的误差
    def computeEk(self, k):
        xk = np.array([self.X[k]])
        y = np.array([self.Y]).T
        # 对数据点的预测值
        gxk = float(self.alpha.T.dot(y * kernel(self.X, xk))) + self.b
        Ek = gxk - self.Y[k]

        return Ek

    # 内循环，根据 i 选择 j
    def selectJ(self, i, Ei):
        # 更新误差缓存
        self.E[i] = [1, Ei]
        # 获取已经缓存的 j
        validE = np.zeros(self.E[:, 0])[0]

    def inner(self, i):
        Ei = self.computeEk(i)

        # 违反了 KKT 条件，加入了容错，不严格执行 KKT
        if (
            (self.Y[i] * Ei > self.epsilon and float(self.alpha[i] > 0))
            or
            (self.Y[i] * Ei < -self.epsilon and float(self.alpha[i]) < self.C)
        ):
            


def SMO(X, Y, C, epsilon, maxIters):
    SVMClassifier = SVM(X, Y, C, epsilon)

    iters = 0
    #由于alpha被初始化为零向量，所以先遍历整个样本集
    iterEntire = True

    while (iters < maxIters):
        iters += 1

        if (iterEntire):
            alphaPairChanges = 0

            # 外层循环
            for i in range(SVMClassifier.N):
                alpha


if __name__ == '__main__':
    positive, negative, dataset = loadDataSet()
    X = dataset[:, 0:2]
    Y = dataset[:, 2]
    