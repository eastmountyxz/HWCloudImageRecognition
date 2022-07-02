# -*- coding: utf-8 -*-
# By:Eastmount
import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread("word.png" )

#转换成灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#中值滤波去除噪声
median = cv2.medianBlur(gray, 3)

#图像直方图均衡化
equalize = cv2.equalizeHist(median)

#显示图像
cv2.imshow('Equalize', equalize)
cv2.waitKey(0)
cv2.destroyAllWindows()
