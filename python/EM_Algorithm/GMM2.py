# https://msgsxj.cn/2018/09/02/EM%E7%AE%97%E6%B3%95/

import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

samples = np.load('./GMM2.npz')
X = samples["data"]
pi0 = samples['pi0']
mu0 = samples['mu0']
sigma0 = samples['sigma0']

# plt.scatter(X[:, 0], X[:, 1], c="grey", s=30)
# plt.axis("equal")
# plt.show()


# E 步, 先计算在当前 mu, sigma 下每个样本属于哪个簇的概率
def E_step(X, pi, mu, sigma):
    N = X.shape[0]
    C = pi.shape[0]
    d = mu.shape[1]

    gamma = np.zeros((N, C))
    sigma_det = []
    # for j in range(C):
    #     sign, logdet = np.linalg.slogdet(sigma[j])
    #     sigma_det.append(sign * np.exp(logdet))

    for j in range(C):
        # 计算每个点在当前条件下属于该簇的概率, 该方法支持同时计算多个点的概率
        gamma[:, j] = pi[j] * multivariate_normal.pdf(X, mu[j], np.diag(sigma[j]))

    return gamma / gamma.sum(axis=1).reshape(-1, 1)


def M_step(X, gamma):
    N = X.shape[0]
    C = gamma.shape[1]
    d = X.shape[1]

    mu = np.zeros((C, d))
    for j in range(C):
        mu[j] = np.sum(X * np.expand_dims(gamma[:, j], -1), axis=0) / np.sum(gamma[:, j])
    
    sigma = np.zeros((C, d, d))
    for j in range(C):
        sigma[j, :, :] = np.sum(np.expand_dims(gamma[:, j], -1) *  ((X - mu[j]) ** 2), axis=0) / np.sum(gamma[:, j])

    pi = np.zeros((C, ))
    for j in range(C):
        pi[j] = np.sum(gamma[:, j]) / N
    return pi, mu, sigma


def train_EM(X, C, rtol=1e-3, max_iter=10, restarts=10):
    N = X.shape[0]
    d = X.shape[1]

    best_loss = 0
    best_pi = np.zeros((C, ))
    best_mu = np.zeros((C, d))
    best_sigma = np.zeros((C, d, d))

    temp = np.random.random(C)
    # 初始pi，即每一簇的比重
    best_pi = temp / np.sum(temp)
    # 初始标准差权重
    best_sigma[:, :, :] = np.eye(2)
    # 初始平均值, 随机取某个样本的值
    best_mu = X[np.random.randint(N, size=3)]

    for i in range(max_iter):
        gamma = E_step(X, best_pi, best_mu, best_sigma)
        best_pi, best_mu, best_sigma = M_step(X, gamma)

    return best_pi, best_mu, best_sigma

if __name__ == "__main__":
    best_pi, best_mu, best_sigma = train_EM(X, 3)
    gamma = E_step(X, best_pi, best_mu, best_sigma)
    labels = np.argmax(gamma, axis=1)
    plt.figure()
    plt.figure(figsize=(16, 12))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30)
    plt.axis('equal')
    plt.show()