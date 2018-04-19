#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import normalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

# tf.logging.set_verbosity(tf.logging.ERROR)


# trainData = "train.csv"
# testData = "test.csv"

trainData = sys.argv[1]

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

train_df = pd.read_csv("train.csv")

valid_df = train_df.iloc[:3000]
train_df = train_df.iloc[3000:]

x_train = np.array( [ list(map(float,train_df["feature"].iloc[i].split())) for i in range(len(train_df)) ] )
x_train = x_train.astype('float32')
x_train = x_train.reshape( -1, 48, 48, 1)
x_train /= 255


x_valid = np.array( [ list(map(float, valid_df["feature"].iloc[i].split())) for i in range(len(valid_df)) ] )
x_valid = x_valid.astype('float32')
x_valid = x_valid.reshape( -1, 48, 48, 1)
x_valid /= 255

y_train = np.array( train_df["label"] )
y_valid = np.array( valid_df["label"] )

y_train = np_utils.to_categorical(y_train, 7)
y_valid = np_utils.to_categorical(y_valid, 7)

datagen = ImageDataGenerator(
	rotation_range=30,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	horizontal_flip=True,
	zoom_range=[0.8, 1.2]
)


data_dim = (48, 48, 1)

model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=data_dim))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', input_shape=data_dim))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', input_shape=data_dim))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))

model.add(Conv2D(128,(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))

model.add(Conv2D(256,(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(normalization.BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))

model.add(Conv2D(512,(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512,(3,3),padding='same'))
model.add(normalization.BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(normalization.BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(normalization.BatchNormalization())
# model.add(Dropout(0.5))

model.add(Dense(7))
model.add(Activation('softmax'))
sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9)
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# cb1 = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
# 				patience=5, min_lr=0.001)
# cb2 = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=64, 
# 				write_graph=True, write_grads=False, write_images=False, 
# 				embeddings_freq=0, embeddings_layer_names=None, 
# 				embeddings_metadata=None)
cb3 = ModelCheckpoint('model-{epoch:05d}.h5', monitor='val_acc',
 				save_best_only=True, period=1)

callback = [cb3]

print(x_train.shape)

# model.fit(x_train, y_train, 
# 	batch_size=64, 
# 	epochs=150, 
# 	validation_data=(x_valid, y_valid),
# 	# callbacks=callback,
# 	)
 
model.fit_generator(
	datagen.flow(x_train, y_train, batch_size=64), 
	steps_per_epoch=len(x_train)//128,
	epochs=256,
	validation_data=(x_valid, y_valid),
	callbacks=callback
	)

# model1 = load_model("./model/model-00253-0.67233.h5")
# model2 = load_model("./model/model-00146-0.67200.h5")
# model3 = load_model("./model/model-00132-0.66567.h5")

model.save("modelEns")