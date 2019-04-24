# https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/extended_kalman_filter/extended_kalman_filter.py
import numpy as np
import math
import matplotlib.pyplot as plt

# 预测的误差矩阵
Q = np.diag([1.0, 1.0]) ** 2
# 预测的噪声协方差矩阵
R = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 0.1  # simulation time [s]


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v, yawrate]]).T
    return u


def observation(xTrue, xd, u):
    # 预测的状态
    xTrue = motion_model(xTrue, u)


# 计算预测的状态矩阵
def motion_model(x, u):
    # 状态转移矩阵，最后一行全是 0 这个没有问题，仔细想想
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    # 控制矩阵，控制每 DT 的状态变化
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F.dot(x) + B.dot(u)

    return x


def main():
    print(__file__ + " start!!")

    time = 0.0

    # 
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)


    xEst = np.zeros((4, 1))

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

if __name__ == '__main__':
    main()