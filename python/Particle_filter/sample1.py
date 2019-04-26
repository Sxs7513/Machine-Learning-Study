# https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/particle_filter/particle_filter.py#L146
# https://www.cnblogs.com/21207-iHome/p/5237701.html
# https://blog.csdn.net/heyijia0327/article/details/41122125
# https://blog.csdn.net/u011624019/article/details/80559397

# 粒子滤波
import math
import numpy as np
import matplotlib.pyplot as plt

# Estimation parameter of PF
Q = np.diag([0.1]) ** 2  # range error
R = np.diag([1.0, np.deg2rad(40.0)])**2  # input error

#  Simulation parameter
Qsim = np.diag([0.2])**2
Rsim = np.diag([1.0, np.deg2rad(30.0)])**2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range

# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling

show_animation = True

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
        dy = xTrue[1, 0] - RFID[i, 1]
        d = math.sqrt(dx ** 2 + dy ** 2)
        if d <= MAX_RANGE:
            # 加上噪声，作为观测距离
            dn = d + np.random.randn() * Qsim[0, 0]
            # 存储观测距离，观测点的 x 值，观测点的 y 值
            zi = np.array([[dn,  RFID[i, 0], RFID[i, 1]]])
            z = np.vstack((z, zi))

    # 控制矩阵加上模拟噪音
    ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
    ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
    ud = np.array([[ud1, ud2]]).T

    # 航位推算状态
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


def gauss_likelihood(x, sigma):
    return 1.0 / (math.sqrt(2 * math.pi) * sigma) * math.exp(-x ** 2 / (2 * sigma ** 2))


# xEst => [4, 1]
# px => [4, NP]
# pw => [1, NP]
def calc_covariance(xEst, px, pw):
    cov = np.zeros((3, 3))

    for i in range(px.shape[1]):
        dx = (px[:, i] - xEst)[0:3]
        cov += pw[0, i] * dx.dot(dx.T)

    return cov


# px => [4, NP]
# pw => [1, NP]
def resampling(px, pw):
    # 衡量粒子权值的退化程度, 有效粒子数越小，表明权值退化越严重
    # 有效粒子数越小，即权重的方差越大，也就是说权重大的和权重小的之间差距大，表明权值退化越严重
    Neff = 1.0 / (pw.dot(pw.T))[0, 0]
    # 当小于阈值的时候，应当重采样
    if Neff < NTh:
        wcum = np.cumsum(pw)
        base = np.cumsum(pw * 0.0 + 1 / NP) - 1 / NP
        resampleid = base + np.random.rand(base.shape[0]) / NP

        inds = []
        ind = 0
        for ip in range(NP):
            while resampleid[ip] > wcum[ind]:
                ind += 1
            inds.append(ind)
        
        px = px[:, inds]
        pw = np.zeros((1, NP)) + 1.0 / NP

    return px, pw


def pf_localization(px, pw, xEst, PEst, z, u):

    for ip in range(NP):
        # 获得某粒子的状态
        x = np.array([px[:, ip]]).T
        # 获得它的权重
        w = pw[0, ip]

        # 在控制矩阵原来噪音的基础上面再加上噪音
        # 然后计算该粒子该次的状态
        ud1 = u[0, 0] + np.random.randn() * Rsim[0, 0]
        ud2 = u[1, 0] + np.random.randn() * Rsim[1, 1]
        ud = np.array([[ud1, ud2]]).T
        x = motion_model(x, ud)

        # 遍历所有观测点，计算权重
        for i in range(len(z[:, 0])):
            dx = x[0, 0] - z[i, 1]
            dy = x[1, 0] - z[i, 2]
            prez = math.sqrt(dx ** 2 + dy ** 2)
            dz = prez - z[i, 0]
            w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0]))

        px[:, ip] = x[:, 0]
        pw[0, ip] = w

    # 归一化操作，使他们的和为1
    pw = pw / pw.sum()

    # 计算本次最佳，shape => [4, 1]
    xEst = px.dot(pw.T)

    # 计算所有粒子与当前最佳位置的协方差矩阵, 注意它不会进入计算中，只是用来展示衡量而已
    PEst = calc_covariance(xEst, px, pw)

    # 重采样
    # px, pw = resampling(px, pw)

    return xEst, PEst, px, pw


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

        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:
            plt.cla()

            for i in range(len(z[:, 0])):
                plt.plot([xTrue[0, 0], z[i, 1]], [xTrue[1, 0], z[i, 2]], "-k")
            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(px[0, :], px[1, :], ".r")
            plt.plot(np.array(hxTrue[0, :]).flatten(),
                     np.array(hxTrue[1, :]).flatten(), "-b")
            plt.plot(np.array(hxDR[0, :]).flatten(),
                     np.array(hxDR[1, :]).flatten(), "-k")
            plt.plot(np.array(hxEst[0, :]).flatten(),
                     np.array(hxEst[1, :]).flatten(), "-r")
            # plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()