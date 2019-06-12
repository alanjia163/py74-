#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# data_and_pre_process
data = pd.read_csv('hour.csv')
ohe_features = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for feature in ohe_features:
    dummies = pd.get_dummies(data[feature], prefix=feature, drop_first=False)
    data = pd.concat([data, dummies], axis=1)
    drop_features = ['instant', 'dteday', 'season', 'weathersit', 'weekday']
    data = data.drop(drop_features, axis=1)

# 标准化
norm_features = ['cnt', 'temp', 'hum', 'windspeed']
scaled_features = {}
for feature in norm_features:
    mean, std = data[feature].mean(), data[feature].std()
    scaled_features[feature] = [mean, std]
    data.loc[:, feature] = (data[feature] - mean) / std

# 拆分数据集以进行训练，验证，测试
test_data = data[-31 * 24:]
data = data[:-31 * 24]
# 提取目标域
target_fields = ['cnt']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), data[target_fields)
# 创建验证集
x_train, y_train = features[:-30 * 24], targets[:-30 * 24]
x_val, y_val = features[-30 * 24:], targets[-30:24:]

# model
model = Sequential()
model.add(Dense(250, input_dim=x_train.shape[1], activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(100, activation='relu'),)
model.add(Dropout(0.02))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.02))

model.add(Dense(25, activation='relu'))
model.add(Dropout(0.02))

model.add(Dense(1, activation='linear'))

#compile
model.compile(loss='mse',metrics=['accuracy'],optimizer='sgd')

#train
history =model.fit(x_train.values,y_train['cnt'],validation_data=(x_val.values,y_val['cnt']),batch_size=1024,epochs=50,verbose=0)

#plot
plt.plot(np.arange(len(history.history['loss'])),history.history['loss'],label='training')
plt.plot(np.arange(len(history.history['val_loss'])),history.history['val_loss'],label='validation')
plt.title('complile train_loss and val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc=0)
plt.show()

print('minimun loss:',min(history.history['val_loss']),'\n',np.argmin(history.history['val_loss']),'epochs')
