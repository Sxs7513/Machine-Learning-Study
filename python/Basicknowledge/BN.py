# BN 层的前向传播与反向传播算法
# https://zhuanlan.zhihu.com/p/26138673
# https://blog.csdn.net/yuechuen/article/details/71502503

# 反向传播，dout 为下一层对 BN 层输出的导数
def batchnorm_backward(dout, cache):
    x, gamma, beta, x_hat, sample_mean, sample_var, eps = cache

    N = x.shape[0]
    dgamma = np.sum(dout, x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_hat = dout * gamma
    dsigma = -0.5 * np.sum(dx_hat * (x - sample_mean), axis=0) * np.power(sample_var + eps, -1.5)
    dmu = -np.sum(dx_hat / np.sqrt(sample_var + eps), axis=0) - 2 * dsigma*np.sum(x-sample_mean, axis=0) / N
    dx = dx_hat * (1 / np.sqrt(sample_var  + eps)) + dsigma * 2.0 * (x - sample_mean) / N + dmu / N

    return dx, dgamma, dbeta