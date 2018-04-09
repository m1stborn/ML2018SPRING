#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D

trainData = "train.csv"
testData = "test.csv"


'''
with open(trainData, 'r') as f:
	width = height = 48
	data = f.read().strip('\r\n').replace(',', ' ').split()[2:]
	data = np.array(data)
	X = np.delete(data, range(0, len(data), width*height+1), axis=0).reshape((-1, width, height, 1)).astype('float')
	Y = data[::width*height+1].astype('int')

print(X.shape)
print(data.shape)
print(Y.shape)
'''

# train_df = pd.read_csv("train.csv")
# train_df = train_df.iloc[4000:]
# x_train = np.array( [ list(map(float,train_df["feature"].iloc[i].split())) for i in range(len(train_df)) ] )
# print(x_train.shape)

data_dim = (48, 48, 1)
nb_class = 7

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=data_dim))
model.add(Activation('relu'))
model.summary()