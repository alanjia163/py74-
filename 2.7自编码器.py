#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
Keras自编码器将街景门牌号SVHN基准库从32*32图像解码为32个浮点数,最后比较原图质量
'''

import numpy as np
from matplotlib import pyplot as plt
import scipy.io

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam

# data
mat = scipy.io.loadmat('train_32x32.mat')
mat = mat['X']
b, h, d, n = mat.shape

# pre_processing
img_gray = np.zeros(shape=(n, b * h))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


for i in range(n):
    # 转换灰度值
    img = rgb2gray(mat[:, :, :, i])
    img = img.reshape(1, 1024)
    img_gray[i, :] = img

# 标准化
x_train = img_gray / 255.
img_size = x_train.shape[1]
#Model
model = Sequential()
model.add(Dense(256,input_dim=img_size,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))

model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(img_size,activation='sigmoid'))

opt = Adam()
model.compile(loss='binary_crossentropy',optimizer=opt)

#train
n_epochs = 100
batch_size =512
model.fit(x_train,x_train,epochs=n_epochs,batch_size=batch_size,shuffle=True,validation_split=0.2)

#
pred = model.predict(x_train)

#plot
n=5
plt.figure(figsize=(10,10))
for i in  range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(mat[i].reshape(32,32),cmap='gray')

    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(pred[i].reshape(32,32),cmap='gray')
plt.show()




