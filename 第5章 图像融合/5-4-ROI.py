# -*- coding:utf-8 -*-
# By：Eastmount
import cv2
import numpy as np

#读取图片
img = cv2.imread("Lena.png")
test = cv2.imread("luo.png",)

#定义150×150矩阵 3对应BGR
face = np.ones((150, 150, 3))

#显示原始图像
cv2.imshow("Demo", img)

#显示ROI区域
face = img[200:350, 200:350]
test[250:400, 250:400] = face
cv2.imshow("Result", test)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
