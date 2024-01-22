# -*- coding: utf-8 -*-
# By：Eastmount
import numpy as np
import matplotlib.pyplot as plt
import cv2

#绘制曲线
def log_plot(c):
    x = np.arange(0, 256, 0.01)
    y = c * np.log(1 + x)
    plt.plot(x, y, 'r', linewidth=1)
    plt.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
    plt.title('对数变换函数')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.show()

#对数变换
def log(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output

#读取原始图像
img = cv2.imread('dark.png')

#绘制对数变换曲线
log_plot(42)

#图像灰度对数变换
output = log(42, img)

#显示图像
cv2.imshow('Input', img)
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
