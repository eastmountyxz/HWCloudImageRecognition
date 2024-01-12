# -*- coding:utf-8 -*-
# By：Eastmount
import cv2
import numpy as np

#创建黑色图像
img = np.zeros((256,256,3), np.uint8)

#绘制文字
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'I love Python!I love Huawei!',
            (10, 100), font, 0.5, (255, 255, 0), 2)

#显示图像
cv2.imshow("polylines", img)

#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
