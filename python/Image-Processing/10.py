# https://cloud.tencent.com/developer/article/1011698
# https://blkstone.github.io/2015/08/20/single-image-haze-removal-using-dark-channel/
# 何凯明去雾算法
# 如果目标场景内在的就和大气光类似，比如雪地、白色背景墙、大海等，则由于前提条件就不正确，因此一般无法获得满意的效果，而对于一般的风景照片这个算法能处理的不错

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
# https://blog.csdn.net/weixin_40647819/article/details/88775598
# https://blog.csdn.net/lxy201700/article/details/25104887
# https://zhuanlan.zhihu.com/p/36813673
class HazeRemoval:
    def __init__(self, img, omega = 0.85, r = 40):
        self.img = img
        self.omega = omega
        self.r = r
        self.eps = 10 ** (-3)
        self.t = 0.1

    
    def boxFilter(self, img, r):
        (rows, cols) = img.shape
        imDst = np.zeros_like(img)
        
        # 首先第一维度进行累加
        imCum = np.cumsum(img, 0)
        # 上边的边缘, 等于 box 中心在 r 处时的累加和
        imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
        # 中间部分, 没有边缘的影响, 直接计算它们的行累加和即可. 矩阵相减可以直接计算得
        imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
        # 下边的边缘
        imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]
        
        # 然后第二维度进行累加, 同样得套路, 即可以搞定
        imCum = np.cumsum(imDst, 1)
        imDst[:, 0:r+1] = imCum[:, r:2*r+1]
        imDst[:, r+1:cols-r] = imCum[:, 2*r+1:cols] - imCum[:, 0:cols-2*r-1]
        imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]

        return imDst

    
    # 导向滤波
    # I => 导向图, 常与 p 是一个东西
    # p => 带有噪声的图
    # r => 窗口大小
    def guidedfilter(self, I, p, r, eps):
        boxFilter = self.boxFilter
        rows, cols = I.shape
        N = boxFilter(np.ones([rows, cols]), r)

        meanI = boxFilter(I, r) / N
        meanP = boxFilter(p, r) / N
        corrIp = boxFilter(I * p, r) / N
        corrI = boxFilter(I * I, r) / N

        varI = corrI - meanI * meanI
        covIp = corrIp - meanI * meanP

        a = covIp / (varI + eps)
        b = meanP - a * meanI

        meanA = boxFilter(a, r) / N
        meanB = boxFilter(b, r) / N

        q = meanA * I + meanB
        return q

    
    def _ind2sub(self, array_shape, ind):
        rows = (ind.astype('int') % array_shape[1])
        cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
        return (rows, cols)


    def _rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


    def haze_removal(self):
        img = np.array(self.img).astype(np.double) / 255.0
        grayImage = self._rgb2gray(img)

        darkImage = img.min(axis=2)

        # # 该算法直接取最大值并取平均作为 A, 但是根据 https://cloud.tencent.com/developer/article/1011698
        # 的看法, 这是不可取的。但是在实践中发现，A值对于去雾的影响不大，但是会影响去雾后的亮度，该方法得到的 A
        # 值较小，会得到较亮的背景色，所以如果原图本身亮度不足的话，可以考虑用该方法来增加亮度。
        # (i, j) = self._ind2sub(darkImage.shape, darkImage.argmax())
        # # i, j = np.where(darkImage == np.max(darkImage))
        # # 原始论文中的A最终是取原始像素中的某一个点的像素, 但是该算法中取的符合条件点的平均值作为A的值
        # A = np.mean(img[i, j, :])

        # 如果原图亮度已经不低了，那么用该方法来获得一个较小的 A
        # 取的符合条件的所有点的平均值作为 A 的值
        bins = 500
        # https://blog.csdn.net/zmshy2128/article/details/53443227
        ht = np.histogram(darkImage, bins)
        # 仔细想想就明白了, 用了很骚的操作来找到前 0.1% 的像素, 然后取它们的三通道平均再找到最大值
        d = np.cumsum(ht[0]) / float(darkImage.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        # 如果使用传统的方法，直接选取图像中的亮度值最高的点作为全局大气光值，
        # 这样原始有雾图像中的白色物体会对此有影响，使得其值偏高。
        # 暗通道的运算可以抹去原始图像中小块的白色物体，所以这样估计的全局大气光值会更准确
        # 经实践发现，取 mean 更好，max 会导致 A 值过大，使得背景色过暗，当然如果原图足够亮的话
        # 那么用 max 也是无所谓的
        # A = np.mean(img, axis=2)[darkImage >= ht[1][lmax]].max()
        A = np.mean(img, axis=2)[darkImage >= ht[1][lmax]].mean()
        print(A)


        t = 1 - self.omega * darkImage / A       

        # 导向滤波 https://blog.csdn.net/baimafujinji/article/details/74750283
        # https://github.com/pfchai/GuidedFilter/blob/master/guidedfilter.py
        t = self.guidedfilter(grayImage, t, self.r, self.eps)
        t[t < self.t] = self.t

        resultImage = np.zeros_like(img)
        for i in range(3):
            resultImage[:, :, i] = (img[:, :, i] - A) / t + A

        resultImage[resultImage < 0] = 0
        resultImage[resultImage > 1] = 1

        return resultImage


if __name__ == '__main__':
    img = cv2.imread('./img/12.jpg')

    # result = haze_removal(img / 255.0)
    hz = HazeRemoval(img)
    result = hz.haze_removal()

    # 保存的像素必须为真实值
    # cv2.imwrite('./img/11-quwu.jpg', result * 255)

    cv2.imshow('image-origin', img)
    cv2.imshow('image-processed', result)
    cv2.waitKey(0) 