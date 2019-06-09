#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


SEED = 2019

#DATA
data = pd.read_csv('winequality-red.csv',sep=';')
y =data['quality']
x = data.drop(['quality',axis=1])

#拆分数据
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=SEED)
#显示平均品质和第一行训练集
print('Average quality training set:{:.4f}'.format(y_train.mean()))
X_train.head()

#数据标准化,
scalar = StandardScaler().fit(X_train)
X_train = pd.DataFrame(scalar.transform(X_train))
X_test = pd.DataFrame(scalar.transform(X_test))

#对每个验证输入的训练数据预测其平均质量
print('MSE:',np.mean((y_test-([y_train.mean()]*y_test.shape[0]))**2).round(4))##MSE:0.594


#定义模型
model = Sequential()

model.add(Dense(200,input_dim=X_train.shape[1],activation='relu'))

model.add(Dense(25,activation='relu'))

model.add(Dense(1,activation='linear'))

opt = Adam()

model.compile(loss='mse',metrics=['accuracy'],optimizer=opt)

#定义回调函数,以便使用早停技术并保持最佳模型
callbacks = [
    EarlyStopping(monitor='val_acc',patience=20,verbose=2),
    ModelCheckpoint('checkpoint/multi_layer_best_model.h5',monitor='val_acc',save_best_only=True,verbose=0),
    ]

#hyper parameters
batch_size = 64
n_epochs = 5000

#train
model.fit(X_train.values,
          y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          validation_split=0.2,
          verbose=2,
          callbacks=callbacks
          )

#test,加载最佳权重后,在测试集上输出显示性能
best_model = model
model_path ='checkpoint/multi_layer_best_model.h5'
best_model.load_weights(model_path)
best_model.compile(loss='mse',optimizer='adam',metrics='accuracy')

#评价测试集
score = best_model.evaluate(X_test.values,verbose=0)
print('Test accuracy: {.2f}%'.format(score[1]*100))




