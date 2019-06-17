# 低通高通滤波器代码实现
# http://accu.cc/content/pil/frequency_filter_lpf/
import numpy as np
import PIL.Image
import scipy.misc

# 理想低通滤波器
# def convert_2d(r):
#     r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
#     for i in range(r.shape[0]):
#         for j in range(r.shape[1]):
#             r_ext[i][j] = r[i][j]
    
#     r_ext_fu = np.fft.fft2(r_ext)
#     r_ext_fu = np.fft.fftshift(r_ext_fu)

#     # 截止频率为 100
#     d0 = 100
#     # 频率域中心坐标
#     center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
#     h = np.empty(r_ext_fu.shape)
#     # 绘制滤波器 H(u, v)
#     for u in range(h.shape[0]):
#         for v in range(h.shape[1]):
#             duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
#             h[u][v] = duv < d0

#     s_ext_fu = r_ext_fu * h
#     print(s_ext_fu)
#     s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
#     s_ext = np.abs(s_ext)
#     s = s_ext[0:r.shape[0], 0:r.shape[1]]

#     for i in range(s.shape[0]):
#         for j in range(s.shape[1]):
#             s[i][j] = min(max(s[i][j], 0), 255)

#     return s.astype(np.uint8)


# 巴特沃斯低通滤波器
# def convert_2d(r):
#     r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
#     for i in range(r.shape[0]):
#         for j in range(r.shape[1]):
#             r_ext[i][j] = r[i][j]

#     r_ext_fu = np.fft.fft2(r_ext)
#     r_ext_fu = np.fft.fftshift(r_ext_fu)

#     # 截止频率为 100, 数值越大通过的低频越多, 图像模糊效果越小
#     d0 = 100
#     # 2 阶巴特沃斯
#     n = 2
#     # 频率域中心坐标
#     center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
#     h = np.empty(r_ext_fu.shape)
#     # 绘制滤波器 H(u, v)
#     for u in range(h.shape[0]):
#         for v in range(h.shape[1]):
#             duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
#             h[u][v] = 1 / ((1 + (duv / d0)) ** (2*n))

#     s_ext_fu = r_ext_fu * h
#     s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
#     s_ext = np.abs(s_ext)
#     s = s_ext[0:r.shape[0], 0:r.shape[1]]

#     for i in range(s.shape[0]):
#         for j in range(s.shape[1]):
#             s[i][j] = min(max(s[i][j], 0), 255)

#     return s.astype(np.uint8)


# 高斯低通滤波器
# def convert_2d(r):
#     r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
#     for i in range(r.shape[0]):
#         for j in range(r.shape[1]):
#             r_ext[i][j] = r[i][j]

#     r_ext_fu = np.fft.fft2(r_ext)
#     r_ext_fu = np.fft.fftshift(r_ext_fu)

#     # 截止频率为 100
#     d0 = 100
#     # 频率域中心坐标
#     center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
#     h = np.empty(r_ext_fu.shape)
#     # 绘制滤波器 H(u, v)
#     for u in range(h.shape[0]):
#         for v in range(h.shape[1]):
#             duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
#             h[u][v] = np.e ** (-duv**2 / d0 ** 2)

#     s_ext_fu = r_ext_fu * h
#     s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
#     s_ext = np.abs(s_ext)
#     s = s_ext[0:r.shape[0], 0:r.shape[1]]

#     for i in range(s.shape[0]):
#         for j in range(s.shape[1]):
#             s[i][j] = min(max(s[i][j], 0), 255)

#     return s.astype(np.uint8)


# 巴特沃斯高通滤波器
# def convert_2d(r):
#     r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
#     for i in range(r.shape[0]):
#         for j in range(r.shape[1]):
#             r_ext[i][j] = r[i][j]

#     r_ext_fu = np.fft.fft2(r_ext)
#     r_ext_fu = np.fft.fftshift(r_ext_fu)

#     # 截止频率为 20
#     d0 = 20
#     # 2 阶巴特沃斯
#     n = 2
#     # 频率域中心坐标
#     center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
#     h = np.empty(r_ext_fu.shape)
#     # 绘制滤波器 H(u, v)
#     for u in range(h.shape[0]):
#         for v in range(h.shape[1]):
#             duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
#             if duv == 0:
#                 h[u][v] = 0
#             else:
#                 h[u][v] = 1 / ((1 + (d0 / duv)) ** (2*n))

#     s_ext_fu = r_ext_fu * h
#     s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
#     s_ext = np.abs(s_ext)
#     s = s_ext[0:r.shape[0], 0:r.shape[1]]

#     for i in range(s.shape[0]):
#         for j in range(s.shape[1]):
#             s[i][j] = min(max(s[i][j], 0), 255)

#     return s.astype(np.uint8)


# 高斯带组滤波器
def convert_2d(r):
    r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r_ext[i][j] = r[i][j]

    r_ext_fu = np.fft.fft2(r_ext)
    r_ext_fu = np.fft.fftshift(r_ext_fu)

    # 截止频率为 100
    d0 = 200
    # 频率域中心坐标
    center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
    h = np.empty(r_ext_fu.shape)
    # 绘制滤波器 H(u, v)
    for u in range(h.shape[0]):
        for v in range(h.shape[1]):
            duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
            if duv == 0:
                h[u][v] = 0
            else:
                h[u][v] = 1 - np.e ** -(((duv ** 2 - d0 ** 2) / (duv * 100)) ** 2)

    s_ext_fu = r_ext_fu * h
    s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
    s_ext = np.abs(s_ext)
    s = s_ext[0:r.shape[0], 0:r.shape[1]]

    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i][j] = min(max(s[i][j], 0), 255)

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