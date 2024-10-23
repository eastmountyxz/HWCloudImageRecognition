# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:43:21 2020 
@author: Eastmount CSDN YXZ
O(∩_∩)O Wuhan Fighting!!!
"""
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from PIL import Image

#---------------------------载入数据及预处理---------------------------
# 下载MNIST数据 
# X shape(60000, 28*28) y shape(10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#------------------------------显示图片------------------------------
def show_mnist(train_image, train_labels):
    n = 6
    m = 6
    plt.figure()
    for i in range(n):
        for j in range(m):
            plt.subplot(n,m,i*n+j+1)
            index = i * n + j #当前图片的标号
            img_array = train_image[index]
            img = Image.fromarray(img_array)
            plt.title(train_labels[index])
            plt.imshow(img, cmap='Greys')
    plt.show()

show_mnist(X_train, y_train)

# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255     # normalize

# 将类向量转化为类矩阵  数字 5 转换为 0 0 0 0 0 1 0 0 0 0 矩阵
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#---------------------------创建神经网络层---------------------------
# Another way to build your neural net
model = Sequential([
        Dense(32, input_dim=784),  # 输入值784(28*28) => 输出值32
        Activation('relu'),        # 激励函数 转换成非线性数据
        Dense(10),                 # 输出为10个单位的结果
        Activation('softmax')      # 激励函数 调用softmax进行分类
        ])

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #学习率lr

# We add metrics to get more results you want to see
# 激活神经网络
model.compile(
        optimizer = rmsprop,                 # 加速神经网络
        loss = 'categorical_crossentropy',   # 损失函数
        metrics = ['accuracy'],               # 计算误差或准确率
        )

#------------------------------训练及预测------------------------------
print("Training")
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)    # 训练次数及每批训练大小
print("Testing")
loss, accuracy = model.evaluate(X_test, y_test)

print("loss:", loss)
print("accuracy:", accuracy)
