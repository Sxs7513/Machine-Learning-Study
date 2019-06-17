# 傅里叶变换
# https://zhuanlan.zhihu.com/p/23607336 这个一定要仔细看
# http://accu.cc/content/pil/frequency_filter/
import numpy as np
import scipy.misc
import PIL.Image
import matplotlib.pyplot as plt
import math

im = PIL.Image.open('./img/3.jpg')
im = im.convert('L')
im_mat = scipy.misc.fromimage(im)
rows, cols = im_mat.shape

# 扩展 M * N 图像到 2M * 2N
im_mat_ext = np.zeros((rows * 2, cols * 2))
for i in range(rows):
    for j in range(cols):
        im_mat_ext[i][j] = im_mat[i][j]

# 快速傅里叶变换, 直接使用 api
im_mat_fu = np.fft.fft2(im_mat_ext)
# 将低频信号移植中间, 等效于在时域上对 f(x, y) 乘以 (-1)^(m + n)
# 可以直接看下 fftshift 的源码, 很简单, 就是互换角落和图片中心位置的数值
im_mat_fu = np.fft.fftshift(im_mat_fu)

# 原生傅里叶变换 # https://www.cnblogs.com/youmuchen/p/8361713.html
# 实际计算过程中可以发现，根本没法算，一个几百 * 几百的图片就足以卡死了
# 因为是四层循环, 所以必须要借助快速傅里叶变换
# h, w = im_mat.shape
# F = np.zeros((h, w), 'complex128')
# for u in range(h):
#     for v in range(w):
#         res = 0
#         for x in range(h):
#             for y in range(w):
#                 res += im_mat[x, y] * np.exp(-1 * j * 2 * math.pi * (u * x / h + v * y / w))
#         F[u, v] = res
# log_F = np.log(1 + np.abs(F))
# im_mat_fu = log_F 


# 显示原图
plt.subplot(121)
plt.imshow(im_mat, 'gray')
plt.title('original')
plt.subplot(122)
# 在显示频率谱之前, 对频率谱取实部并进行对数变换
plt.imshow(np.log(np.abs(im_mat_fu)), 'gray')
plt.title('fourier')
plt.show()


# 傅里叶反变换
im_converted_mat = np.fft.ifft2(np.fft.ifftshift(im_mat_fu))
# 得到傅里叶反变换结果的实部
im_converted_mat = np.abs(im_converted_mat)
# 提取左上象限
im_converted_mat = im_converted_mat[0:rows, 0:cols]
# 显示图像
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()