# -*- coding: utf-8 -*-
# By：Eastmount
import cv2  
import numpy as np
import matplotlib.pyplot as plt

#读取图像
src = cv2.imread('lena-hd.png')

#绘制直方图
#plt.hist(src.ravel(), 256)
plt.hist(src.ravel(), bins=256, density=1, facecolor='green', alpha=0.75)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#显示原始图像
cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
