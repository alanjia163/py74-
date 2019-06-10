#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin
'''
测试不同的优化器
ModelCheckpoint，加载模型验证 
'''
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adadelta, Adam, RMSprop, Adagrad, nadam, Adamax

data = pd.read_csv('winequality-red.csv', sep=';')
y = data['quality']
x = data.drop(['quality'], axis=1)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2019)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=2019)


# model_function
def create_model(opt):
    '''
    creat a  model
    :param opt:
    :return:model
    '''
    model = Sequential()
    model.add(Dense(100, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='linear'))
    return model


# create_function，用来定义在训练期间将使用的回调函数
def create_callbacks(opt):
    '''
    回调函数
    :return:callbacks，类型list
    '''

    # 一般是在model.fit函数中调用callbacks，fit函数中有一个参数为callbacks。
    # 注意这里需要输入的是list类型的数据，所以通常情况只用EarlyStopping的话也要是[EarlyStopping()]
    callbacks = [
        EarlyStopping(monitor='val_acc', patience=200, verbose=2),
        ModelCheckpoint('best_model_' + opt + '.h5', monitor='val_acc', save_best_only=True, verbose=0)
    ]
    return callbacks


# 创建一个想要尝试的优化器字典
opts = dict({'sgd': SGD(),
             'adam': Adam(),
             })

# train_and_save
results = []
# 遍历优化器
for opt in opts:  # 依次取到每个键
    model = create_model(opt)
    callbacks = create_callbacks(opt)
    model.compile(loss='mse', optimizer=opts[opt], metrics=['accuracy'])
    hist = model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=5000,
                     validation_data=(x_train.values, y_train),
                     callbacks=callbacks,
                     verbose=0
                     )
    best_epoch = np.argmax(hist.history['val_acc'])  # 返回最大下标
    best_acc = hist.history['val_acc']['best_epoch']

    # 加载具有最高验证精度的模型
    best_model = create_model(opt)
    best_model.load_weights('best_model_' + opt + '.h5')
    best_model.compile(loss='mse', optimizer=opts[opt], metrics='accuracy')
    score = best_model.evaluate(x_train.values, y_test, verbose=0)
    test_accuracy = score[1]
    results.append([opt, best_epoch, best_acc, test_accuracy])

    # 比较结果
    res = pd.DataFrame(results)
    res.columns = ['optimizer', 'epoch', 'val_accuracy', 'test_accuracy']
    print(res)  # 或者控制台输入：res
