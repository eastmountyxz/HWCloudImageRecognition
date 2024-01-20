# -*- coding: utf-8 -*-
# By：Eastmount
import cv2  
import numpy as np  
import matplotlib.pyplot as plt

#读取原始图像
img_BGR = cv2.imread('luo.png')

img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)     #BGR转换为RGB
img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)   #灰度化处理
img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)     #BGR转HSV
img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb) #BGR转YCrCb
img_HLS = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HLS)     #BGR转HLS
img_XYZ = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2XYZ)     #BGR转XYZ
img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)     #BGR转LAB
img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)     #BGR转YUV

#调用matplotlib显示处理结果
titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']  
images = [img_BGR, img_RGB, img_GRAY, img_HSV, img_YCrCb,
          img_HLS, img_XYZ, img_LAB, img_YUV]  
for i in range(9):  
   plt.subplot(3, 3, i+1), plt.imshow(images[i], 'gray')  
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()
