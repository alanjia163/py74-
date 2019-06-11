#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin



'''
L1正则化
'''


import numpy as np
import pandas as pd
from matplotlib  import pyplot as plt

from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras import regularizers

#data
data = pd.read_csv('hour.csv')

#特征工程
ohe_features = ['season','weathersit','mnth','hr','weekday']
for feature in ohe_features:
    dummies = pd.get_dummies(data[feature],prefix=feature,drop_first=False)
    data = pd.concat([data,dummies],axis=1)
    drop_features = ['instant','dteday','season','weathersit','weekday','atemp','mnth','workingday','hr','casual','registered']
    data = data.drop(drop_features,axis=1)

#标准化数值数据
norm_features = ['cnt','temp','hum','windspeed']
scaled_features = {}
for feature in norm_features:
    mean,std = data[feature].mean(),data[feature].std()
    scaled_features[feature] = [mean,std]

#分割数据集进行训练，验证，测试
#保存最后月份数据用于测试
test_data = data[-31*24:]
data = data[:-31*24]

#提取目标域
target_fields = 'cnt'
features,targets = data.drop(target_fields,axis=1),data[target_fields]
test_features,test_targets = test_data.drop(target_fields,axis=1),data[target_fields]

#创建验证集
x_train ,y_train = features[:30*24],targets[:-30*24]
x_val,y_val =  features[-30*24:],targets[-30*24:]

#model
model = Sequential()
model.add(Dense(250,activation='relu',input_dim=x_train.shape[1],kernel_regularizer=regularizers.l2(0.001)))#正则化
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu',kernel_regularizer=regularizers.l2(0.001)))#正则化
model.add(Dense(1,activation='linear'))

#编译模型
model.compile(loss = 'mse',optimizer='sgd',metrics=['mse','accuracy'])

#

#hyper parameters
n_epochs = 4000
batch_size = 1024
history = model.fit(x_train.values,
                    y_train['cnt'],
                    validation_data=(x_val.values,y_val['cnt']),
                    batch_size=batch_size,
                    epochs=n_epochs,
                    verbose=0)

#绘制结果图
plt.plot(np.arange(len(history.history['loss'])),history.history['loss'],label='training')
plt.plot(np.arange(len(history.history['val_loss'])),history.history['val_loss'],label='validation')
plt.title('use L2')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc=0)
plt.show()

#输出
print('mininum loss:',min(history.history['val_loss']),
      '\n',
      np.argmin(history.history['val_loss']),
      'epochs'
      )






