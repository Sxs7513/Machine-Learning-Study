import math
import numpy as np

MAX_RANGE = 20.0

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v, yawrate]]).T
    return u


# 创建真实值，观察值
def observation(xTrue, xd, u, RFID):
    xTrue = motion_model(xTrue, u)

    z = np.zeros((0, 3))

    # 计算四个点的观测值，并模拟加上噪音
    for i in range(len(RFID[:, 0])):
        dx = xTrue[0, 0] - RFID[i, 0]
        dy = xTrue[0, 1] - RFID[i, 1]
        d = math.sqrt(dx ** 2 + dy ** 2)
        if d <= MAX_RANGE = 20.0:
            dn = d + np.random.randn() * Qsim[0, 0]
            zi = np.array([[dn,  [i, 0], RFID[i, 1]]])
            z = np.vstack((z, zi))

    # 控制矩阵加上模拟噪音
    ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
    ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
    ud = np.array([[ud1, ud2]]).T

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def motion_model(x, u):

    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F.dot(x) + B.dot(u)

    return x


def pf_localization(px, pw, xEst, PEst, z, u):

    for ip in range(NP):
        # 获得某粒子的状态
        x = np.array([px[:, ip]]).T
        # 获得它的权重
        w = pw[0, ip]


def main():
    print(__file__ + " start!!")

    time = 0.0

    RFID = np.array([
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 15.0],
        [-5.0, 20.0],
    ])

    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    # 存储所有粒子的状态
    px = np.zeros((4, NP))
    # 存储所有粒子的权重
    pw = np.zeros((1, NP)) + 1.0 / NP
    # 初始的航位状态
    xDR = np.zeros((4, 1))

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        # 获得本次的模拟数据
        # xTrue => 下一个真实的状态
        # z => 当次的观察值
        # xDR => 下一个航位推算状态，为了表明在控制矩阵有噪声的情况下，不用 kalman 偏移会有多大
        # 相对定位是通过测量机器人相对于初始位置的距离和方向来确定机器人的当前位置，通常称为航迹推算法，常用的传感器包括里程计及惯性导航系统(陀螺仪、加速度计等)
        # ud => 当次模拟的控制矩阵
        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID)

        xEst, PEst, px, pw = pf_localization(px, pw, xEst, PEst, z, ud)


if __name__ == '__main__':
    main()