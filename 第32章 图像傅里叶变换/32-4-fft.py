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

#转移像素做幅度普
fshift = np.fft.fftshift(f)       

#取绝对值：将复数变化成实数取对数的目的为了将数据变化到0-255
res = np.log(np.abs(fshift))

#设置字体
matplotlib.rcParams['font.sans-serif']=['SimHei']

#展示结果
plt.subplot(121), plt.imshow(img, 'gray'), plt.title(u'(a)原始图像'), plt.axis('off')
plt.subplot(122), plt.imshow(res, 'gray'), plt.title(u'(b)傅里叶变换处理'), plt.axis('off')
plt.show()
