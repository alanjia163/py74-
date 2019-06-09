#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical, np_utils
from keras.callbacks import Callback
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense

# data
(x_train, y_train), (x_val, y_val) = mnist.load_data()

# 显示每个标签的事例,并输出对显示每个标签的计数
unique_labels = set(y_train)
plt.figure(figsize=(12, 10))
i = 1
for label in unique_labels:
    image = x_train[y_train.tolist().index(label)]
    #plt.subplots(10, 10, i)
    plt.axis('off')
    plt.title('{0}:({1})'.format(label, y_train.tolist().count(label)))
    i += 1
    _ = plt.imshow(image, cmap='gray')
plt.show()

# 预处理数据
# 标准化
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.

# 对标号独热编码
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)

# 拼合数据
x_train = np.reshape(x_train, (60000, 784))
x_val = np.reshape(x_val, (10000, 784))

# layer _relu
model_relu = Sequential([
    Dense(700, input_dim=x_train.shape[1], activation='relu'),
    Dense(700, activation='relu'),
    Dense(700, activation='relu'),
    Dense(700, activation='relu'),
    Dense(700, activation='relu'),
    Dense(360, activation='relu'),
    Dense(300, activation='relu'),
    Dense(10, activation='softmax'),
])

model_relu.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'],
)

'''
以下函数参考:https://keras-cn.readthedocs.io/en/latest/other/callbacks/
参数
on_epoch_begin: 在每个epoch开始时调用
on_epoch_end: 在每个epoch结束时调用
on_batch_begin: 在每个batch开始时调用
on_batch_end: 在每个batch结束时调用
on_train_begin: 在训练开始时调用
on_train_end: 在训练结束时调用
编写自己的回调函数
我们可以通过继承keras.callbacks.Callback编写自己的回调函数，回调函数通过类成员self.model访问访问，该成员是模型的一个引用。

这里是一个简单的保存每个batch的loss的回调函数：
'''


class history_loss(Callback):
    def on_train_begin(self, logs={}):
        self.lossses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.lossses.append(batch_loss)


# hyper parameters
n_epochs = 10
batch_size = 256
validation_split = 0.2

# train
history_relu = history_loss()
model_relu.fit(
    x_train, y_train, epochs=n_epochs,
    batch_size = batch_size,
    callbacks = [history_relu],
    validation_split=validation_split,
    verbose=2
    )

#绘制损失分布图列
plt.plot(np.arange(len(history_relu.lossses)),relu,label='relu')
plt.title('losses')
plt.xlabel('number of batches')
plt.ylabel('loss')
plt.legend(loc=1)
plt.show()