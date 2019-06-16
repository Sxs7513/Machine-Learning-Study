import numpy as np
import PIL.Image
import scipy.misc
import scipy.ndimage

# http://accu.cc/content/pil/spatial_filter_medium/
# 中值滤波

def convert_2d(r):
    n = 10
    s = scipy.ndimage.median_filter(r, (n, n))
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