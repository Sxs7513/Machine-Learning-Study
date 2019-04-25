# https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/extended_kalman_filter/extended_kalman_filter.py
# https://blog.csdn.net/codesamer/article/details/81191487
# https://blog.csdn.net/qth515/article/details/54601413
# https://www.cnblogs.com/TIANHUAHUA/p/8473029.html
# 扩展雅可比
import numpy as np
import math
import matplotlib.pyplot as plt

# 预测的误差矩阵
Q = np.diag([0.1, 0.1, np.deg2rad(1.0), 1.0]) ** 2
# 预测的噪声协方差矩阵
R = np.diag([1.0, 1.0])**2

# 误差噪声协方差矩阵
Qsim = np.diag([1.0, np.deg2rad(30.0)]) ** 2
# 观测的噪声协方差矩阵
Rsim = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50  # simulation time [s]

show_animation = True


# 输出此时的速度与角度
def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v, yawrate]]).T
    return u


# 生成观测值, 与真实值
def observation(xTrue, xd, u):
    # 计算真实的下一步的状态
    xTrue = motion_model(xTrue, u)

    # 给真实坐标加上噪声, 模拟观察值
    zx = xTrue[0, 0] + np.random.randn() * Rsim[0, 0]
    zy = xTrue[1, 0] + np.random.randn() * Rsim[1, 1]
    z = np.array([[zx, zy]]).T
    
    # 给控制矩阵加上噪声, 模拟控制量也是不精确的
    ud1 = u[0, 0] + np.random.randn() * Qsim[0, 0]
    ud2 = u[1, 0] + np.random.randn() * Qsim[1, 1]
    ud = np.array([[ud1, ud2]]).T

    # 计算航位推算的下一步状态
    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


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


def jacobF(x, u):
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


# 输出观察矩阵的雅可比矩阵
def jacobH():
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH

# 计算 h(xPred)
def observation_model(x):
    # 
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    # h(xPred), 但是在这里可以看出来并不是非线性哦, 只是为了模拟
    z = H.dot(x)

    return z


# predict and update
def ekf_estimation(xEst, PEst, z, u):
    # predict
    # 注意这里只是模拟非线性, 其实并不是非线性
    xPred = motion_model(xEst, u)
    # 计算状态转移矩阵的雅可比矩阵
    jF = jacobF(xPred, u)
    PPred = jF.dot(PEst).dot(jF.T) + Q

    #  Update
    # 观察矩阵的雅可比矩阵
    jH = jacobH()
    # h(xPred), 非线性 
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH.dot(PPred).dot(jH.T) + R
    K = PPred.dot(jH.T).dot(np.linalg.inv(S))
    xEst = xPred + K.dot(y)
    PEst = (np.eye(len(xEst)) - K.dot(jH)).dot(PPred)

    return xEst, PEst


def main():
    print(__file__ + " start!!")

    time = 0.0

    # 生成初始的状态预测值，状态真值，预测噪声协方差矩阵
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    # 初始的航位状态
    xDR = np.zeros((4, 1))

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        # 获得本次的模拟数据
        # xTrue => 下一个真实的状态
        # z => 当次的观察值
        # xDR => 下一个航位推算状态，为了表明在控制矩阵有噪声的情况下，不用 kalman 偏移会有多大
        # 相对定位是通过测量机器人相对于初始位置的距离和方向来确定机器人的当前位置，通常称为航迹推算法，常用的传感器包括里程计及惯性导航系统(陀螺仪、加速度计等)
        # ud => 当次模拟的控制矩阵
        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        # xEst => 本次的最佳状态, 和下次的初始状态
        # PEst => 本次的最佳预测协方差矩阵, 下次的初始矩预测协方差矩阵
        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            # plot_covariance_ellipse(xEst, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)

if __name__ == '__main__':
    main()