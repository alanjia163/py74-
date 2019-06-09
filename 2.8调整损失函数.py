#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin


'''
损失函数有：
mse、交叉熵、其他一些、以及自定义损失
confusion_matrix()
'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 创建不平衡数据集，9s,和 4s
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 提取9s中全部样本，和4s中100样本
y_train_9 = y_train[y_train == 9]
y_train_4 = y_train[y_train == 4][:100]
x_train_9 = y_train[y_train == 9]
x_train_4 = y_train[y_train == 4][:100]
x_train = np.concatenate((x_train_9, x_train_4), axis=0)
y_train = np.concatenate((y_train_9, y_train_4), axis=0)

y_test_9 = x_test[y_test == 9]
y_test_4 = y_test[y_test == 4]
x_test_9 = x_test[y_test == 9]
x_test_4 = x_test[y_test == 4]
x_test = np.concatenate((x_test_9, x_test_4), axis=0)
y_test = np.concatenate((y_test_9, y_test_4), axis=0)

# 标准化和扁平化数据
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

# 将目标转换成二进制分类问题并输出显示计数
y_train_binary = y_train == 9
y_test_bianry = y_test == 9
print(np.unique(y_train_binary, return_counts=True))

# model
model = Sequential()
model.add(Dense(512, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1, activation='sigmoid'))

opt = Adam()
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# 创建一个回调函数，来使用早停法
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

# 定义每个类的损失权重
class_weight_equal = {False: 1, True: 1}
class_weight_imbalanced = {False: 100, True: 1}

# 对两个类训练相同权重的模型
n_epochs = 1000
batch_size = 512
validation_split = 0.01
model.fit(x_train, y_train_binary,
          epochs=n_epochs,
          batch_size=batch_size,
          verbose=0, shuffle=True,
          validation_split=validation_split,
          callbacks=callbacks,
          class_weight=class_weight_equal)

# 第一种： 在测试集上进行测试并输出混淆矩阵
pred_equal = model.predict(x_test)
confusion_matrix(y_test_bianry, np.round(pred_equal), labels=[True, False])
#

# 第二种： 使用不平衡权重训练，并测试集测试
model.compile(loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['binary_accuracy'],
              )
model.fit(x_train, y_test_bianry, epochs=n_epochs, batch_size=batch_size, shuffle=True,
          validation_split=validation_split,
          class_weight=class_weight_imbalanced,
          callbacks=callbacks,
          verbose=0
          )

preds_imbalanced = model.predict(x_test)
confusion_matrix(y_test_bianry, np.round(preds_imbalanced), labels=[True, False])
