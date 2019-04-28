# https://algorithm.joho.info/programming/python/opencv-particle-filter-py/

import cv2
import numpy as np

# 追跡対象の色範囲（Hueの値域）
def is_target(roi): 
    return (roi <= 30) | (roi >= 50)

# 
def max_moment_point(mask):
    # 找到二值掩膜中的连通区域
    # return => [区域的数量，，区域的左上角和右下角坐标以及大小，区域的中心坐标]
    label = cv2.connectedComponentsWithStats(mask)
    # 都删除第一个是什么鬼
    data = np.delete(label[2], 0, 0) 
    center = np.delete(label[3], 0, 0)
    moment = data[:, 4]
    # 找到面积最大的区域并返回
    max_index = np.argmax(moment)
    return center[max_index]

# 
def initialize(img, N): 
    mask = img.copy()
    # 将不追踪的像素值置为 0
    mask[is_target(mask) == False] = 0
    # 找到掩膜中最大连通区域的中心最表
    x, y = max_moment_point(mask)
    # 计算区域中存在目标颜色的概率
    w = calc_likelihood(x, y, img)
    ps = np.ndarray((N, 3), dtype=np.float32)
    # 初始的粒子的位置全部为连通区域的中心，并且权重均一致
    ps[:] = [x, y, w]
    return ps


# 重采样, 使用的是残差重采样？不太像，回头来补
def resampling(ps):
    ws = ps[:, 2].cumsum()
    last_w = ws[-1]
    new_ps = np.empty(ps.shape)
    # 
    for i in range(ps.shape[0]):
        w = np.random.rand() * last_w
        new_ps[i] = ps[(ws > w).argmax()]
        new_ps[i, 2] = 1.0

    return new_ps

# 让粒子随机散开
def predict_position(ps, var=13.0): 
    ps[:, 0] += np.random.randn((ps.shape[0])) * var
    ps[:, 1] += np.random.randn((ps.shape[0])) * var


# 
def calc_likelihood(x, y, img, w=30, h=30):
    # 获取图像中指定的区域的像素值
    x1, y1 = max(0, x-w/2), max(0, y-h/2)
    x2, y2 = min(img.shape[1], x+w/2), min(img.shape[0], y+h/2)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    roi = img[y1:y2, x1:x2]
    # 计算区域中存在跟踪颜色的概率
    count = roi[is_target(roi)].size
    return (float(count) / img.size) if count > 0 else 0.0001


# 
def calc_weight(ps, img): 
    # 计算每个粒子的权重
    for i in range(ps.shape[0]): 
        ps[i][2] = calc_likelihood(ps[i, 0], ps[i, 1], img)

    # 权重归一化
    ps[:, 2] *= ps.shape[0] / ps[:, 2].sum()


# 计算每个粒子的权重，并求得加权平均后得中心位置
def observer(ps, img):
    # 计算每个粒子的权重
    calc_weight(ps, img)   

    x = (ps[:, 0] * ps[:, 2]).sum()
    y = (ps[:, 1] * ps[:, 2]).sum()
    
    return (x, y) / ps[:, 2].sum()

# パーティクルフィルタ
def particle_filter(ps, img, N=300):
    # 如果是第一桢，初始化粒子
    if ps is None:
        ps = initialize(img, N)
    
    # 重采样
    ps = resampling(ps)
    # 让粒子随机散开
    predict_position(ps)
    # 计算每个粒子的权重，并求得加权平均后得中心位置
    x, y = observer(ps, img)
    return ps, int(x), int(y)

def main():
    # パーティクル格納用の変数
    ps = None

    # 读取视频
    cap = cv2.VideoCapture('./Person.wmv')
    
    while cv2.waitKey(30) < 0:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
        h = hsv[:, :, 0]

        # 二值化
        ret, s = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        ret, v = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        h[(s == 0) | (v == 0)] = 100  

        # 经过重采样，随机散开粒子，后求得加权平均后得中心位置
        ps, x, y = particle_filter(ps, h, 300)
        
        if ps is None:
            continue

        # 
        ps1 = ps[(ps[:, 0] >= 0) & (ps[:, 0] < frame.shape[1]) & (ps[:, 1] >= 0) & (ps[:, 1] < frame.shape[0])]
        # 将有效粒子标红
        for i in range(ps1.shape[0]): 
            frame[int(ps1[i, 1]), int(ps1[i, 0])] = [0, 0, 200]     
        # 画出以 x，y 为中心得指定大小得矩形
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (0, 0, 200), 5)

        cv2.imshow('Result', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
        