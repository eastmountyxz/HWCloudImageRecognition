# -*- coding: utf-8 -*-
# By: Eastmount
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

#读取图像
img = cv2.imread('test.png')

#灰度转换
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#显示原始图像
plt.subplot(121), plt.imshow(gray, 'gray'), plt.title(u'(a)原始图像')
plt.axis('off')

#霍夫变换检测圆
circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param2=30)
print(circles1)

#提取为二维
circles = circles1[0, :, :]

#四舍五入取整
circles = np.uint16(np.around(circles))

#绘制圆
for i in circles[:]: 
    cv2.circle(img, (i[0],i[1]), i[2], (255,0,0), 5) #画圆
    cv2.circle(img, (i[0],i[1]), 2, (255,0,255), 10) #画圆心

#设置字体
matplotlib.rcParams['font.sans-serif']=['SimHei']
plt.subplot(122), plt.imshow(img), plt.title('(b)结果图像')
plt.axis('off')
plt.show()
