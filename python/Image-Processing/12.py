# 图像增强retinex算法
# https://blog.csdn.net/weixin_38285131/article/details/88097771
# https://www.cnblogs.com/wangyong/p/8665434.html

import numpy as np
import cv2


def singleScaleRetinex(img, sigma=300):
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.01, _temp)
    retinex = np.log10(img + 0.01) - np.log10(gaussian)

    return retinex


def multiScaleRetinex(img, sigma_list=[15, 80, 200]):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    # 这里使用相等权重
    img_msr = retinex / len(sigma_list)

    return img_msr


# https://blog.csdn.net/yayan01/article/details/50129391
# 上面连接里面作者说 MSRCR 没有效果, 按照文中提出的方法实践后, 发现
# dynamic 参数不能用文中所说的 2 或者 3, 20 左右可以得到较好的效果 
def MSRCRScaleRetinexSimple(img, dynamic=5, sigma=300):
    img_ssr = singleScaleRetinex(img)
    
    mean = np.mean(img_ssr)
    var = np.var(img_ssr)
    
    min_gimp = mean - dynamic * var
    max_gimp = mean + dynamic * var
    
    img_ssr = (img_ssr - min_gimp) / (max_gimp - min_gimp) * 255
    
    return img_ssr


def simplestColorBalance(img, low_clip, high_clip):    

    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        # unique 自带排序
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):            
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
                
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


# 15 号图片用其他方法颜色都偏深, 唯独用该方法可以, 这是由于其他方法都存在色偏的问题
# https://www.cnblogs.com/Imageshop/p/3810402.html
# http://www.jsjkx.com/CN/article/openArticlePDF.jsp?id=503 => 里面有说 MSRCP
# https://blog.csdn.net/weixin_38285131/article/details/88097771 => 里面有说 MSRCP
def MSRCP(img, sigma_list=[15, 80, 200], low_clip=0.01, high_clip=0.99):
    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / (np.max(intensity1) - np.min(intensity1)) * 255.0 + 1.0

    img_msrcp = np.zeros_like(img)
    
    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            # 放大因子 A
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    return img_msrcp



def convert(img, useMethod='msrcp'):
    img = np.array(img, dtype=np.float)
    # 单尺度计算
    if useMethod == 'single':
        img = singleScaleRetinex(img, 300)
    # 多尺度计算
    elif useMethod == 'multi':
        img = multiScaleRetinex(img, sigma_list=[15, 80, 200])
    elif useMethod == 'msrcrSimple':
        img = MSRCRScaleRetinexSimple(img)
    elif useMethod == 'msrcp':
        img = MSRCP(img)
    
    # 量化到 0-255，量化公式：R(x,y) = ( Value - Min ) / (Max - Min) * (255-0)
    # （注：无需将Log[R(x,y)]进行Exp函数的运算,而是直接利用Log[R(x,y)]进行线性映射）
    if useMethod == 'single' or useMethod == 'multi':
        for i in range(img.shape[2]):
            img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255.0
    
    img = np.uint8(
        np.minimum(
            np.maximum(img, 0), 
            255
        )
    )

    return img


if __name__ == "__main__":
    image = cv2.imread('./img/15.jpg')
    processed = convert(image)

    cv2.imshow('origin-image', image)
    cv2.imshow('processed-image', processed)
    cv2.waitKey(0)