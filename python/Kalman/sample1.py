# https://blog.csdn.net/codesamer/article/details/81191487
# http://czy13.com/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/kalman-filters/
import numpy as np
import matplotlib.pyplot as plt

# 创建真实值
z = np.arange(0, 100, 1)
z = np.expand_dims(z, 0)

# 创建一个方差为1的高斯噪声，精确到小数点后两位
noise = np.round(np.random.normal(0, 1, 100), 2)
noise = np.expand_dims(noise, 0)

# 将z的真实和噪声相加，作为 mock 数据
z_mat = z + noise

# 定义 x 的初始状态，即位置和速度
x_mat = np.array([[0,], [0,]])
# 定义初始状态噪声协方差矩阵
p_mat = np.array([[1, 0], [0, 1]])
# 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
f_mat = np.array([[1, 1], [0, 1]])
# 定义误差矩阵，这里把值设置的较小，代表该系统误差较小
q_mat = np.array([[0.0001, 0], [0, 0.0001]])
# 定义观测转移矩阵, 因为是匀速运动并且物理量能直接通过传感器测量，所以 1，0 即可
h_mat = np.array([[1, 0]])
# 定义观测噪声协方差
r_mat = np.array([[1]])

for i in range(100):
    # 更新预测的状态矩阵
    x_predict = np.dot(f_mat, x_mat)
    # 更新预测噪声协方差矩阵
    p_predict = f_mat.dot(p_mat).dot(f_mat.T) + q_mat
    # 更新卡尔曼系数
    kalman = np.dot(p_predict, h_mat.T) / (h_mat.dot(p_predict).dot(h_mat.T) + r_mat)
    # 更新最佳的状态矩阵
    x_mat = x_predict + np.dot(kalman, (z_mat[0, i] - np.dot(h_mat, x_predict)))
    # 更新最佳噪声协方差矩阵
    p_mat = np.dot((np.eye(2) - np.dot(kalman, h_mat)), p_predict)
    
    plt.plot(x_mat[0, 0], x_mat[1, 0], 'ro', markersize = 1)

plt.show()