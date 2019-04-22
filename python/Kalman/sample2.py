# https://blog.csdn.net/qq_23981335/article/details/82968422
# https://blog.csdn.net/angelfish91/article/details/61768575
import cv2
import numpy as np

frame = np.zeros((800, 800, 3), np.uint8)

last_measurement = current_measurement = np.array((2, 1), np.float32)
last_predicition = current_prediction = np.zeros((2, 1), np.float32)


def mousemove(event, x, y, s, p):
    global frame, current_measurement, measurements, last_measurement, current_prediction, last_prediction
    last_measurement = current_measurement
    last_prediction = current_prediction

    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])

    kalman.correct(current_measurement)

    current_prediction = kalman.predict()

    #上一次测量值
    lmx, lmy = last_measurement[0], last_measurement[1]
    #当前测量值
    cmx, cmy = current_measurement[0], current_measurement[1]
    #上一次预测值
    lpx, lpy = last_prediction[0], last_prediction[1]
    #当前预测值
    cpx, cpy = current_prediction[0], current_prediction[1]
    #绘制测量值轨迹（绿色）
    cv2.line(frame, (lmx, lmy), (cmx, cmy), (0, 100, 0))
    #绘制预测值轨迹（红色）
    cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200))


cv2.namedWindow("kalman_tracker")

cv2.setMouseCallback("kalman_tracker", mousemove)

# 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
kalman = cv2.KalmanFilter(4, 2)
#设置测量矩阵, x y 方向的速度与位置
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
#设置转移矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
#设置过程噪声协方差矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

while True:
    cv2.imshow("kalman_tracker", frame)
    if (cv2.waitKey(30) & 0xff) == 27:
        break

cv2.destroyAllWindows()
