import imageio
import PIL.Image
import scipy.misc
import numpy as np

# http://accu.cc/content/pil/contrast/
# 幂次变换
# 等于大幅提升暗的亮度，小幅提升亮的亮度，达到增亮即将暗处细节展示出来效果
def convert_3d(r):
    s = np.empty(r.shape, dtype=np.uint8)
    for j in range(r.shape[0]):
        for i in range(r.shape[1]):
            s[j][i] = (r[j][i] / 255) ** 0.67 * 255
    return s

im_mat = imageio.imread('img/2.jpg')
im_converted_mat = convert_3d(im_mat)
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()