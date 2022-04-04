# -*- coding:utf-8 -*-
import cv2

#读取图片
img = cv2.imread("huawei.png")

#显示图像
cv2.imshow("Demo", img)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
