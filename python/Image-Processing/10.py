# https://cloud.tencent.com/developer/article/1011698
# https://blkstone.github.io/2015/08/20/single-image-haze-removal-using-dark-channel/
# 何凯明去雾算法

import cv2
import numpy as np

# 未加 guided filter 的实现
# https://github.com/pfchai/Haze-Removal/blob/master/HazeRemoval.py
def haze_removal(image, windowSize=24, w0=0.6, t0=0.1):
    darkImage = image.min(axis=2)
    # 论文中最原始的求 A 的方法
    A = darkImage.max()
    darkImage = darkImage.astype(np.double)
    
    t = 1 - w0 * (darkImage / A)
    T = t * 255
    T.dtype = 'uint8'

    t[t < t0] = t0

    J = image
    J[:, :, 0] = (image[:, :, 0] - (1 - t) * A) / t
    J[:, :, 1] = (image[:, :, 1] - (1 - t) * A) / t
    J[:, :, 2] = (image[:, :, 2] - (1 - t) * A) / t

    return J


# https://github.com/pfchai/Haze-Removal/blob/master/HazeRemovalWidthGuided.py
# https://blog.csdn.net/zmshy2128/article/details/53443227
class HazeRemoval:
    def __init__(self, img, omega = 0.85, r = 40):
        self.img = img
        self.omega = omega
        self.r = r
        self.eps = 10 ** (-3)
        self.t = 0.1

    
    def _rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


    def haze_removal(self):
        img = np.array(self.img).astype(np.double) / 255.0
        grayImage = self._rgb2gray(img)

        darkImage = img.min(axis=2)

        # # 该算法直接取最大值并取平均作为 A, 但是根据 https://cloud.tencent.com/developer/article/1011698
        # # 的看法, 这是不可取的
        # i, j = np.argmax(darkImage)
        # # 原始论文中的A最终是取原始像素中的某一个点的像素, 但是该算法中取的符合条件点的平均值作为A的值
        # A = np.mean(img[i, j, :])

        # 取的符合条件的所有点的平均值作为 A 的值
        bins = 2000
        # https://blog.csdn.net/zmshy2128/article/details/53443227
        ht = np.histogram(darkImage, bins)
        # 仔细想想就明白了, 用了很骚的操作来找到前 0.1% 的像素, 然后取它们的三通道平均再找到最大值
        d = np.cumsum(ht[0]) / float(darkImage.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        A = np.mean(m, axis=2)[darkImage >= ht[1][lmax]].max()       

        # 导向滤波 https://blog.csdn.net/baimafujinji/article/details/74750283



if __name__ == '__main__':
    img = cv2.imread('./img/5.jpeg')

    img = haze_removal(img / 255.0)
    cv2.imshow('image', img)
    cv2.waitKey(0) 