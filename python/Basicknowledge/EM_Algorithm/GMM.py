# http://sofasofa.io/tutorials/gmm_em/
# https://zhuanlan.zhihu.com/p/55826713
# http://frankchen.xyz/2016/11/18/Understanding-EM-algorithm/
# 高斯混合模型EM算法实现

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')


def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)

    X = np.concatenate([X1, X2, X3], axis=0)

    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


def plot_clusters(X, Mu, Var, Mu_true=None, Var_true=None):
    colors = ["b", "g", "r"]
    n_clusters = Mu.shape[0]
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X[:, 0], X[:, 1], s=5)
    ax = plt.gca()
    for i in range(n_clusters):
        plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
        ellipse = Ellipse(Mu[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)
        ax.add_patch(ellipse)
    if (Mu_true is not None) & (Var_true is not None):
        for i in range(n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
            ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
            ax.add_patch(ellipse)         
    plt.show()


# 计算对数似然函数的 logLH 以及用来可视化数据的 plot_clusters
def logLH(X, Pi, Mu, Var):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros(((n_points, n_clusters)))
    for i in range(n_clusters):
        # 计算每个点属于该簇的概率, 该方法支持同时计算多个点的概率
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    # 
    return np.mean(np.log(pdfs.sum(axis=1)))


def update_W(X, Mu, Var, Pi):
    n_points, n_clusters = len(X), len(Pi)
    pdfs = np.zeros((n_points, n_clusters))
    for i in range(n_clusters):
        # 计算每个点属于该簇的概率
        pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
    # 概率归一化
    W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return W


def update_Pi(W):
    Pi = W.sum(axis=0) / W.sum()
    return Pi


# 更新均值
def update_Mu(X, W):
    n_clusters = W.shape[1]
    Mu = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Mu[i] = np.average(X, axis=0, weights=W[:, i])
    return Mu


# 更新标准差
def update_Var(X, Mu, W):
    n_clusters = W.shape[1]
    Var = np.zeros((n_clusters, 2))
    for i in range(n_clusters):
        Var[i] = np.average((X - Mu[i]) ** 2, axis=0, weights=W[:, i])
    return Var


if __name__ == "__main__":
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    # 初始化聚类的个数
    n_clusters = 3
    # 样本个数
    n_points = X.shape[0]
    # 每个高斯分布的初始均值
    Mu = np.array([[0, -1], [6, 0], [0, 9]])
    # 每个高斯分布的初始标准差
    Var = np.array([[1, 1], [1, 1], [1, 1]])
    Pi = [1 / n_clusters] * 3
    # 初始的每个点属于哪个簇的概率即隐变量, 初始的时候每个样本属于每一簇的概率为 1 / 3
    W = np.ones((n_points, n_clusters)) / n_clusters
    # 每一簇的比重, 可以根据 W 求得，在初始时，Pi = [1/3, 1/3, 1/3]
    Pi = W.sum(axis=0) / W.sum()
    
    # 迭代
    loglh = []
    for i in range(5):
        plot_clusters(X, Mu, Var, true_Mu, true_Var)
        loglh.append(logLH(X, Pi, Mu, Var))
        # 先计算w再计算Pi
        W = update_W(X, Mu, Var, Pi)
        Pi = update_Pi(W)
        Mu = update_Mu(X, W)
        print('log-likehood:%.3f'%loglh[-1])
        Var = update_Var(X, Mu, W)