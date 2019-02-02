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

sigma = 10.0


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

    def updateEk(self, k):
        Ek = self.computeEk(k)
        self.E[k] = [1, Ek]

    # 内循环，根据 i 选择 j
    def selectJ(self, i, Ei):
        # 更新误差缓存
        self.E[i] = [1, Ei]
        # 获取已经缓存的 j
        validE = np.nonzero(self.E[:, 0])[0]

        if (len(validE) > 1):
            j = 0
            maxDelta = 0
            Ej = 0

            # 寻找最大的 |Ei-Ej|
            for k in validE:
                if (k == i): continue
                Ek = self.computeEk(k)
                if (abs(Ei - Ek) > maxDelta):
                    j = k
                    maxDelta = abs(Ei - Ek)
                    Ej = Ek
        else:
            j = i
            while (j == i):
                j = int(np.random.uniform(0, self.N))
            Ej = self.computeEk(j)

        return j, Ej

    def inner(self, i):
        Ei = self.computeEk(i)

        # 违反了 KKT 条件，加入了容错，不严格执行 KKT
        if ((self.Y[i] * Ei > self.epsilon and float(self.alpha[i] > 0))
                or (self.Y[i] * Ei < -self.epsilon
                    and float(self.alpha[i]) < self.C)):
            j, Ej = self.selectJ(i, Ei)
            alphaI = float(self.alpha[i])
            alphaJ = float(self.alpha[j])

            # 两种情况下 L 与 H
            if (self.Y[i] != self.Y[j]):
                L = max(0, alphaJ - alphaI)
                H = min(self.C, self.C + alphaJ - alphaI)
            else:
                L = max(0, alphaJ + alphaI - self.C)
                H = min(self.C, alphaJ + alphaI)

            if (L == H): return 0

            xi = np.array([self.X[i]])
            xj = np.array([self.X[j]])
            # K11 + K22 − 2K12
            eta = float(kernel(xi, xi) + kernel(xj, xj) - 2 * kernel(xi, xj))
            if (eta <= 0): return 0

            alphaJnewunc = alphaJ + self.Y[j] * (Ei - Ej) / eta
            # 更新 alphaJ
            if (alphaJnewunc > H): self.alpha[j] = [H]
            elif (alphaJnewunc < L): self.alpha[j] = [L]
            else: self.alpha[j] = [alphaJnewunc]

            # 更新 Ej
            self.updateEk(j)
            if (abs(float(self.alpha[j]) - alphaJ) < 0.00001): return 0

            # 更新 alphaI
            self.alpha[i] = [
                alphaI +
                self.Y[i] * self.Y[j] * (alphaJ - float(self.alpha[j]))
            ]

            # 更新 Ei
            self.updateEk(i)

            # 更新b
            bi = - Ei - self.Y[i] * float(kernel(xi, xi)) * (float(self.alpha[i]) - alphaI) -\
                self.Y[j] * float(kernel(xj, xi)) * (float(self.alpha[j]) - alphaJ) + self.b

            bj=- Ej - self.Y[i] * float(kernel(xi, xj)) * (float(self.alpha[i]) - alphaI) -\
                self.Y[j] * float(kernel(xj, xj)) * (float(self.alpha[j]) - alphaJ) + self.b

            if (0 < float(self.alpha[i]) and float(self.alpha[i]) < self.C):
                self.b = bi
            elif (0 < float(self.alpha[j]) and float(self.alpha[j]) < self.C):
                self.b = bj
            else:
                self.b = 0.5 * (bi + bj)

            return 1
        else:
            return 0

    # 寻找非边界点
    def findNonBound(self):
        nonbound = []

        for i in range(len(self.alpha)):
            # 该情况下必满足 KKT
            if (0 < self.alpha[i] and self.alpha[i] < self.C):
                nonbound.append(i)

        return nonbound

    def visualize(self, positive, negative):
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.scatter(positive[:, 0], positive[:, 1], c='r', marker='o')
        plt.scatter(negative[:, 0], negative[:, 1], c='g', marker='o')

        # 通过非零 alpha 找到支持向量
        nonZeroAlpha = self.alpha[np.nonzero(self.alpha)]
        supportVector = X[np.nonzero(self.alpha)[0]]
        y = np.array([self.Y]).T[np.nonzero(self.alpha)]
        plt.scatter(
            supportVector[:, 0],
            supportVector[:, 1],
            s=100,
            c='y',
            alpha=0.5,
            marker='o')
        print("支持向量个数:", len(nonZeroAlpha))

        X1 = np.arange(-50, 50, 0.1)
        X2 = np.arange(-50, 50, 0.1)
        x1, x2 = np.meshgrid(X1, X2)
        g = self.b
        for i in range(len(nonZeroAlpha)):
            # g += nonZeroAlpha[i] * y[i] * (x1 * supportVector[i][0] + x2 * supportVector[i][1])
            g += nonZeroAlpha[i] * y[i] * np.exp(-0.5 * (
                (x1 - supportVector[i][0]) ** 2 +
                (x2 - supportVector[i][1]) ** 2) / (sigma**2))

        # 画出超平面
        plt.contour(x1, x2, g, 0, colors='b')
        plt.title("sigma: %f" % sigma)
        plt.show()


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
                alphaPairChanges += SVMClassifier.inner(i)

            # 训练集上无 alpha 变化时推出循环
            if (alphaPairChanges == 0):
                break
                # 如果有变化那么遍历非边界数据
            else:
                iterEntire = False

        # 遍历非边界数据
        else:
            alphaPairChanges = 0
            nonbound = SVMClassifier.findNonBound()
            # 外层循环
            for i in nonbound:
                alphaPairChanges += SVMClassifier.inner(i)

            # 非边界点全满足KKT条件，则循环遍历整个样本集
            if (alphaPairChanges == 0):
                iterEntire = True

    return SVMClassifier


if __name__ == '__main__':
    positive, negative, dataset = loadDataSet()
    X = dataset[:, 0:2]
    Y = dataset[:, 2]

    SVMClassifier = SMO(X, Y, 1, 0.001, 40)
    SVMClassifier.visualize(positive, negative)