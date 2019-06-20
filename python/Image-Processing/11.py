# 维纳滤波
# https://blog.csdn.net/bluecol/article/details/46242355
# http://cynhard.com/2018/09/11/LA-Complex-Vectors-and-Matrices/#%E5%A4%8D%E6%95%B0%E7%9A%84%E8%BF%90%E7%AE%97

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

# https://blog.csdn.net/qq_29769263/article/details/85330933
# https://blog.csdn.net/bingbingxie1/article/details/79398601
def get_motion_dsf(image_size, motion_angle, motion_dis):
    PSF = np.zeros(image_size)
    x_center = (image_size[0] - 1) / 2
    y_center = (image_size[1] - 1) / 2

    sin_val = math.sin(motion_angle * math.pi / 180)
    cos_val = math.cos(motion_angle * math.pi / 180)

    for i in range(motion_dis):
        x_offset = round(sin_val * i)
        y_offset = round(cos_val * i)
        PSF[int(x_center - x_offset), int(y_center + y_offset)] = 1

    return PSF / np.sum(PSF)


# 依据的原理是运动模糊实际上就等于一个滤波(特定方向值为1的卷积核)
# 同时空间域的卷积等于频域的乘积。这一切的目的当然是为了加快速度
def make_blurred(input, PSF, eps):
    # 原图傅里叶变换
    input_fft = np.fft.fft2(input)
    # 退化模型傅里叶变换，顺便加点料
    PSF_fft = np.fft.fft2(PSF) + eps
    # 相乘再反傅里叶回去
    blurred = np.fft.ifft2(input_fft * PSF_fft)
    blurred = np.abs(np.fft.fftshift(blurred))
    return blurred


def inverse(input, PSF, eps):
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) + eps
    result = np.fft.ifft2(input_fft / PSF_fft)
    result = np.abs(np.fft.fftshift(result))
    return result


def wiener(input,PSF,eps,K=0.01):        #维纳滤波，K=0.01
    input_fft = np.fft.fft2(input)
    PSF_fft = np.fft.fft2(PSF) +eps
    PSF_fft_1 = np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + K)
    result = np.fft.ifft2(input_fft * PSF_fft_1)
    result = np.abs(np.fft.fftshift(result))
    return result



if __name__ == "__main__":
    image = cv2.imread('./img/17.jpg') / 255
    cv2.imshow('origin', image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    img_h = image.shape[0]
    img_w = image.shape[1]
    
    # 先模拟运动模糊
    PSF = get_motion_dsf((img_h, img_w), motion_angle=0, motion_dis=55)
    # blurred = np.abs(make_blurred(image, PSF, 1e-3))
    # plt.gray()
    # plt.imshow(blurred)
    # plt.show()

    # # 加点随机的噪声
    # blurred_noisy = blurred + 0.1 * blurred.std() * np.random.standard_normal(blurred.shape)
    # plt.imshow(blurred_noisy)
    # plt.show()

    # # 对添加了随机噪声的图像进行逆滤波，只要眼神正常就可以发现，参杂了噪声的情况下
    # # 反卷积会把噪声强烈放大，导致逆滤波失败
    # result = inverse(blurred_noisy, PSF, 1e-3)
    # plt.imshow(result)
    # plt.show()

    # 对添加了随机噪声的图像进行维纳滤波
    # result = wiener(image, PSF, 0)
    # plt.gray()
    # plt.imshow(result)
    # plt.show()


    # 三通道尝试进行 motion_process, 失败
    result = []
    for d in range(image.shape[2]):
        rr = image[:, :, d]
        ss = wiener(rr, PSF, 0, K=0.01)
        result.append(ss)
    result = np.dstack(result)
    result[result < 0] = 0
    result[result > 1] = 1

    cv2.imshow('image', result)
    cv2.waitKey(0)
    

    
    # 滤波卷积核的形式来进行运动模糊，因为空间域的卷积等于频域的乘积
    # size = 40
    # #创建一个卷积核
    # kernel = np.zeros((size,size))
    # # 列对应上下模糊，所以这里是上下模糊
    # kernel[:, int((size-1) / 2.0)] = np.ones(size)
    # # 取平均保证亮度不会增强很多，核的中心一列为 0.067，其他的均为 0
    # kernel = kernel / size
    # result = []
    # for d in range(image.shape[2]):
    #     rr = image[:, :, d]
    #     ss = cv2.filter2D(rr, -1, kernel)
    #     result.append(ss)
    # result = np.dstack(result)
    # result[result < 0] = 0
    # result[result > 255] = 255
    # plt.imshow(result)
    # plt.show()