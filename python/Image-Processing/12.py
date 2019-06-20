# 图像增强retinex算法
# https://blog.csdn.net/weixin_38285131/article/details/88097771
# https://www.cnblogs.com/wangyong/p/8665434.html

import numpy as np
import cv2


def singleScaleRetinex(img, sigma=300):
    _temp = cv2.GaussianBlur(img, (0, 0), sigma)
    gaussian = np.where(_temp == 0, 0.01, _temp)
    img_ssr = np.log10(img + 0.01) - np.log10(gaussian)
    
    return img_ssr


def multiScaleRetinex(img, sigma_list=[15, 80, 200]):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    # 这里使用相等权重
    img_msr = retinex / len(sigma_list)

    return img_msr


# https://blog.csdn.net/yayan01/article/details/50129391
# https://www.cnblogs.com/Imageshop/p/3810402.html
# http://www.jsjkx.com/CN/article/openArticlePDF.jsp?id=503 => 里面有说 MSRCP
# https://blog.csdn.net/weixin_38285131/article/details/88097771 => 里面有说 MSRCP
# 上面连接里面作者说 MSRCR 没有效果, 按照文中提出的方法实践后, 发现
# dynamic 参数不能用文中所说的 2 或者 3, 20 左右可以得到较好的效果 
def MSRCRScaleRetinexSimple(img, dynamic=5, sigma=300):
    img_ssr = singleScaleRetinex(img)
    
    mean = np.mean(img_ssr)
    var = np.var(img_ssr)
    
    min_gimp = mean - dynamic * var
    max_gimp = mean + dynamic * var
    
    img_ssr = (img_ssr - min_gimp) / (max_gimp - min_gimp) * 255

    # 比如第15号图片的色彩不能很好的还原, 采用 MSRCP 方法尝试
    
    
    return img_ssr


def convert_2d(img, useMethod='single'):
    # 单尺度计算
    if useMethod == 'single':
        img = singleScaleRetinex(img, 300)
    # 多尺度计算
    elif useMethod == 'multi':
        img = multiScaleRetinex(img, sigma_list=[15, 80, 200])
    elif useMethod == 'msrcrSimple':
        img = MSRCRScaleRetinexSimple(img)

    return img


def convert_3d(r, useMethod='msrcrSimple'):
    r = np.array(r, dtype=np.float)
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr, useMethod)
        s_dsplit.append(ss)
    img = np.dstack(s_dsplit)

    if useMethod == 'single' or useMethod == 'multi':
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

    # 量化到 0-255，量化公式：R(x,y) = ( Value - Min ) / (Max - Min) * (255-0)
    # （注：无需将Log[R(x,y)]进行Exp函数的运算,而是直接利用Log[R(x,y)]进行线性映射）
    # for i in range(img.shape[2]):
    #     img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255 

    img = np.uint8(
        np.minimum(
            np.maximum(img, 0), 
            255
        )
    )

    return img


if __name__ == "__main__":
    image = cv2.imread('./img/15.jpg')
    processed = convert_3d(image)

    cv2.imshow('origin-image', image)
    cv2.imshow('processed-image', processed)
    cv2.waitKey(0)