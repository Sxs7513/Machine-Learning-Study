# https://yq.aliyun.com/articles/647272?utm_content=m_1000017701

import imageio
import matplotlib.pyplot as plt
import numpy as np

pic = imageio.imread('img/1.png')

# plt.figure(figsize=(6,6))
# plt.imshow(pic)
# plt.axis('off')
# plt.show()


# 强度变换, 每个像素值都减去255。这样的操作导致的结果是，较亮的像素变暗，较暗的图像变亮，类似于图像底片
# negative = 255 - pic
# plt.figure(figsize= (6,6))
# plt.imshow(negative)
# plt.axis('off')
# plt.show()
# print(pic.shape)


# 对数变换
gray=lambda rgb:np.dot(rgb[...,:3],[0.299,0.587,0.114])
gray=gray(pic)
'''
log transform
-> s = c*log(1+r)

So, we calculate constant c to estimate s
-> c = (L-1)/log(1+|I_max|)

'''
max_=np.max(gray)

def log_transform():
    return(255/np.log(1+max_))*np.log(1+gray)

plt.figure(figsize=(5,5))
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))
plt.axis('off')
plt.show()