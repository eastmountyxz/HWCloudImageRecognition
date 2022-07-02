# -*- coding: utf-8 -*-
# By:Eastmount
import cv2  
import numpy as np  
import matplotlib.pyplot as plt

"""
函数：stretch
功能：图像灰度拉伸
参数：原始图像
"""
def stretch(img):
    maxi=float(img.max())
    mini=float(img.min())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]=(255/(maxi-mini)*img[i,j]-(255*mini)/(maxi-mini))
    return img

"""
函数：LocateLicense
功能：结合颜色定位车牌图像
参数：处理后的图像、原始图像
"""
def LocateLicense(img, sourceimg):
    #定位车牌号
    contours,hierarchy = cv2.findContours(img,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
    #找出最大的三个区域
    block = []
    for contour in contours:
        #寻找矩形轮廓（找出轮廓的左上点和右下点）
        y,x = [],[]
        for p in contour:
            y.append(p[0][0])
            x.append(p[0][1])
        r = [min(y),min(x),max(y),max(x)]
        
        #计算面积和长度比
        a = (r[2]-r[0])*(r[3]-r[1])   #面积
        s = (r[2]-r[0])*(r[3]-r[1])   #长度比
        block.append([r,a,s])
        
    #选出面积最大的3个区域
    block = sorted(block,key=lambda b: b[1])[-3:]

    #使用颜色识别判断找出最像车牌的区域
    maxweight,maxindex = 0,-1
    for i in range(len(block)):
        b = sourceimg[block[i][0][1]:block[i][0][3],
                      block[i][0][0]:block[i][0][2]]
        #BGR转HSV
        hsv = cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
        #蓝色车牌的范围
        lower = np.array([100,50,50])
        upper = np.array([140,255,255])
        #根据阈值构建掩膜
        mask = cv2.inRange(hsv,lower,upper)
        #统计权值
        w1 = 0
        for m in mask:
            w1 += m/255
        w2 = 0
        for n in w1:
            w2 += n
        #选出最大权值的区域
        if w2 > maxweight:
            maxindex = i
            maxweight = w2
    return block[maxindex][0]

"""
函数：ShowImage
功能：显示图像
参数：9种处理步骤的图像
"""
def ShowImage(img,gray,stretchedimg,openimg,diffimg,binary,canny,openimg3,result):
    titles = ['Source Image','Gray Image', 'Stretched Image', 'Open Image',
          'Diff Image', 'Binary Image', 'Canny Image', 'Locate Image', 'Result Image']  
    images = [img,gray,stretchedimg,openimg,
              diffimg,binary,canny,openimg3,result]  
    for i in range(9):  
       plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')  
       plt.title(titles[i])  
       plt.xticks([]),plt.yticks([])  
    plt.show()

"""
函数：ImagePreprocessing
功能：图像预处理
参数：原始图像
"""
def ImagePreprocessing(img):

    source = img
    
    #1.形状变换和压缩图像
    m = 400*img.shape[0]/img.shape[1]
    img = cv2.resize(img,(400,int(m)),interpolation=cv2.INTER_CUBIC)

    #2.BGR转换为灰度图像
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #3.灰度拉伸
    stretchedimg = stretch(gray)

    #4.开运算：去除噪声
    r = 16
    h = w = r*2+1
    kernel = np.zeros((h,w),np.uint8)
    cv2.circle(kernel,(r,r),r,1,-1)
    openimg = cv2.morphologyEx(stretchedimg,cv2.MORPH_OPEN,kernel)

    #5.获取差分图：两幅图像做差
    diffimg = cv2.absdiff(stretchedimg,openimg)

    #6.图像二值化
    maxi=float(diffimg.max())
    mini=float(diffimg.min())
    x = maxi-((maxi-mini)/2)
    ret,binary = cv2.threshold(diffimg,x,255,cv2.THRESH_BINARY)
    
    #7.Canny边缘检测
    canny = cv2.Canny(binary,binary.shape[0],binary.shape[1])

    #8.消除小区域保留大区域
    #闭运算
    kernel = np.ones((5,19),np.uint8)
    closeimg = cv2.morphologyEx(canny,cv2.MORPH_CLOSE,kernel)
    #开运算
    openimg2 = cv2.morphologyEx(closeimg,cv2.MORPH_OPEN,kernel)
    #再次开运算
    kernel = np.ones((11,5),np.uint8)
    openimg3 = cv2.morphologyEx(openimg2,cv2.MORPH_OPEN,kernel)

    #9.结合颜色定位车牌位置并消除小区域
    rect = LocateLicense(openimg3,img)
    print(rect)

    #10.框出车牌号
    result = cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),2)
    cv2.imshow('afterimg',result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #显示图像
    ShowImage(source,gray,stretchedimg,openimg,
              diffimg,binary,canny,openimg3,result)

    return rect,img

"""
函数：CutLicense
功能：图像分割
参数：预处理后图像 识别区域
"""
def CutLicense(result, rect):
    CutImg = result[rect[1]:rect[3], rect[0]:rect[2]]
    return CutImg

#-------------------------------------------------------------------------
#主函数
if __name__=='__main__':
    
    #读取图片
    imagePath = 'car-03.png'
    img = cv2.imread(imagePath, cv2.IMREAD_COLOR)

    #图像预处理：车牌位置 处理图像
    rect, result = ImagePreprocessing(img)

    #车牌图像分割
    cutimg = CutLicense(result, rect)
    cv2.imshow('cutimg', cutimg)
    cv2.imwrite('save-'+imagePath, cutimg)

    #去除上下边缘噪声
    



    
