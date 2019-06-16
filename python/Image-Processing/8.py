# 傅里叶变换
# https://zhuanlan.zhihu.com/p/23607336
# http://accu.cc/content/pil/frequency_filter/
# https://www.cnblogs.com/youmuchen/p/8361713.html
import numpy as np
import math

data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 20]])
def myDFT(data):
    w, h = np.shape(data)
    coe = np.zeros((w, h))

    x = np.arange(w)
    y = np.arange(h)

    for col_ind in range(h):
        row_ww = np.exp(-2 * math.pi * 1j * (y + col_ind) / h)
        for row_ind in range(w):
            col_ww = np.exp(-2 * math.pi * 1j * (x * row_ind) / w)

            coe[row_ind + 1, col_ind + 1] = row_ww * (col_ww * data)

    return coe

print(myDFT(data))