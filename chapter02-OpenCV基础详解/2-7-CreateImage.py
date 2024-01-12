# -*- coding:utf-8 -*-
# By：Eastmount
import cv2
import numpy as np

#读取图片
img = cv2.imread("Lena.png")

#创建空图像
emptyImage = np.zeros(img.shape, np.uint8)

#显示图像
cv2.imshow("Demo", emptyImage)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
