# 二硬币投掷
# https://zhuanlan.zhihu.com/p/31345125
import numpy as np
from scipy import stats

# 硬币投掷结果观测序列
observations = np.array(
    [[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]]
)

def em_single(priors, observation):
    counts = {'A': {'H': 0, 'T': 0}, 'B': {'H': 0, 'T': 0}}
    theta_A = priors[0]
    theta_B = priors[1]

    for observation in observations:
        len_observation = len(observation)
        num_heads = observation.sum()
        num_tails = len_observation - num_heads
        # 注意在这里认为扔硬币没有先后顺序，所以用二项分布
        # 来判断比如在十次中有五次正面的概率，而不是一个一个乘
        contribution_A = stats.binom.pmf(num_heads, len_observation, theta_A)
        contribution_B = stats.binom.pmf(num_heads, len_observation, theta_B)
        weight_A = contribution_A / (contribution_A + contribution_B)
        weight_B = contribution_B / (contribution_A + contribution_B)

        counts['A']['H'] += weight_A * num_heads
        counts['A']['T'] += weight_A * num_tails
        counts['B']['H'] += weight_B * num_heads
        counts['B']['T'] += weight_B * num_tails

    new_theta_A = counts['A']['H'] / (counts['A']['H'] + counts['A']['T'])
    new_theta_B = counts['B']['H'] / (counts['B']['H'] + counts['B']['T'])
    return [new_theta_A, new_theta_B]


def em(observations, prior, tol=1e-6, iterations=10000):
    import math
    itertaion = 0
    while itertaion < iterations:
        new_prior = em_single(prior, observations)
        delta_change = np.abs(prior[0] - new_prior[0])
        if delta_change < tol:
            break
        else:
            prior = new_prior
            itertaion += 1
    return [new_prior, itertaion]

if __name__ == "__main__":
    print(em(observations, [0.6, 0.5]))