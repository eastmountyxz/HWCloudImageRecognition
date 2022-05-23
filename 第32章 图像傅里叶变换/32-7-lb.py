# -*- coding: utf-8 -*-
# By: Eastmount
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

#读取图像
img = cv2.imread('lena-hd.png', 0)

#傅里叶变换
dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
fshift = np.fft.fftshift(dft)

#设置低通滤波器
rows, cols = img.shape
crow,ccol = int(rows/2), int(cols/2) #中心位置
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

#掩膜图像和频谱图像乘积
f = fshift * mask
print(f.shape, fshift.shape, mask.shape)

#傅里叶逆变换
ishift = np.fft.ifftshift(f)
iimg = cv2.idft(ishift)
res = cv2.magnitude(iimg[:,:,0], iimg[:,:,1])

#设置字体
matplotlib.rcParams['font.sans-serif']=['SimHei']

#显示原始图像和低通滤波处理图像
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('(a)原始图像')
plt.axis('off')
plt.subplot(122), plt.imshow(res, 'gray'), plt.title('(b)结果图像')
plt.axis('off')
plt.show()
