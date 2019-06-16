import numpy as np
import PIL.Image
import scipy.misc
import scipy.signal

# http://accu.cc/content/pil/spatial_filter_mean/
# 空间滤波-均值滤波

def convert_2d(r):
    n = 3
    # 3*3 滤波器, 每个系数都是 1/9
    window = np.ones((n, n)) / n ** 2
    # 使用滤波器卷积图像
    # mode = same 表示输出尺寸等于输入尺寸
    # boundary 表示采用对称边界条件处理图像边缘
    s = scipy.signal.convolve2d(r, window, mode='same', boundary='symm')
    return s.astype(np.uint8)


def convert_3d(r):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s


im = PIL.Image.open('./img/3.jpg')
im_mat = scipy.misc.fromimage(im)
im_converted_mat = convert_3d(im_mat)
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()