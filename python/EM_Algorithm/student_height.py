# https://cfonheart.github.io/2018/09/18/EM%E7%AE%97%E6%B3%95%E5%8F%8A%E7%AE%80%E5%8D%95%E4%BE%8B%E5%AD%90python%E5%AE%9E%E7%8E%B0/
# 学生身高采样em算法实现(高斯分布)

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# 男生均值 1.7, 标准差 0.05, 女生同理
men = np.random.normal(1.7, 0.05, 500)
women = np.random.normal(1.60, 0.1, 500)

plt.figure(21)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.set_title('distribution of men')
ax1.hist(men, bins=60, histtype="stepfilled", normed=True, alpha=0.6, color='r')
ax2.set_title('distribution of women')
ax2.hist(women, bins=60, histtype="stepfilled", normed=True, alpha=0.6)
plt.show()


def em_single(observations, param_a, param_b):
    menpro_list = []
    womenpro_list = []

    # E 步
    for i, v in enumerate(observations):
        # 计算在当前均值与标准差下, 该身高为男生的概率
        p1 = norm.pdf(v, loc=param_a[0], scale=param_a[1])
        # 计算在当前均值与标准差下, 该身高为女生的概率
        p2 = norm.pdf(v, loc=param_b[0], scale=param_b[1])

        # 归一化
        menpro_list.append(p1 / (p1 + p2))
        womenpro_list.append(p2 / (p1 + p2))

    # M 步
    sum1 = 0; sum2 = 0
    for i, v in enumerate(observations):
        sum1 += menpro_list[i] * v
        sum2 += womenpro_list[i] * v

    # 求均值, 仔细想下为什么除以 sum(menpro_list)
    loc1 = sum1 / sum(menpro_list)
    loc2 = sum2 / sum(womenpro_list)

    scale1 = 0; scale2 = 0
    for i, v in enumerate(observations):
        scale1 += menpro_list[i] * (v - loc1) * (v - loc1)
        scale2 += womenpro_list[i] * (v - loc2) * (v - loc2)

    # 求标准差
    scale1 = np.sqrt(scale1 / sum(menpro_list))
    scale2 = np.sqrt(scale2 / sum(womenpro_list))

    return [[loc1, scale1], [loc2, scale2]]

    
def em(observations, param_a, param_b, tol = 1e-6, iterations=1000):
    for iter in range(iterations):
        param_a_new, param_b_new = em_single(observations, param_a, param_b)
        if abs(param_a_new[0]-param_a[0])+abs(param_a_new[1]-param_a[1]) < tol and abs(param_b_new[0]-param_b[0])+abs(param_b_new[1]-param_b[1]) < tol:
            return ([param_a_new, param_b_new], iter+1)
        param_a = param_a_new.copy()
        param_b = param_b_new.copy()

    return ([param_a, param_b], iterations)


if __name__ == "__main__":
    observations = []
    for v in men:
        observations.append(v)
    for v in women:
        observations.append(v)
    param_a = [1.7, 1]
    param_b = [1,4, 1]
    print(em(observations, param_a, param_b))