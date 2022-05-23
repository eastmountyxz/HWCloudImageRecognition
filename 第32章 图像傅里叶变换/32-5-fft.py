# -*- coding: utf-8 -*-
# By: Eastmount
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

#读取图像
img = cv.imread('luo.png', 0)

#傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
res = np.log(np.abs(fshift))

#傅里叶逆变换
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)

#设置字体
matplotlib.rcParams['font.sans-serif']=['SimHei']

#展示结果
plt.subplot(131), plt.imshow(img, 'gray'), plt.title('(a)原始图像')
plt.axis('off')
plt.subplot(132), plt.imshow(res, 'gray'), plt.title('(b)傅里叶变换处理')
plt.axis('off')
plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('(c)傅里叶逆变换处理')
plt.axis('off')
plt.show()
