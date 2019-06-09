#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
一般来说,使用隐藏层神经元数量逐层减少,通常在步骤中每层除以2,
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

SEED = 7

# data
data = pd.read_csv('winequality-red.csv', sep=';')
x = data.drop(['quallity'], axis=1)
y = data['quality']

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

# 标准化
scaler = StandardScaler().fit(x_train)
x_train = pd.DataFrame(scaler.transform(x_train))
x_test = pd.DataFrame(scaler.transform(x_test))

# model
model = Sequential()
model.add(Dense(1024, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='linear'))

# optimizer
opt = SGD()

# compile
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

# hyper parameters
n_epoch = 500
batch_size = 256
history = model.fit(
    x_train.values, y_train,
    batch_size=batch_size,
    epochs=n_epoch,
    validation_split=0.2,
    verbose=0
)

# 测试集进行测试
predictions = model.predict(x_test.values)

print('test accuracy:{:f>2}%'.format(
    np.round(np.sum([y_test == predictions.flatten().round()]) / y_test.shape[0] * 100, 2)))

# 绘制训练精度图 和 验证精度图
plt.plot(np.arange(len(history.history['acc'])), history.history['acc'], label='training')
plt.plot(np.arange(len(history.history['val_acc'])), history.history['val_acc'], label='validation')

plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('精度')
plt.show()
