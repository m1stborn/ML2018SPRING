import sys
import numpy as np
import pandas as pd

from comet_ml import Experiment

import keras.backend as K
from keras.models import Model
from keras.layers import Embedding,Input,Dot,Add,Concatenate,Dropout,Dense,Flatten
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

ver = 13 

# experiment = Experiment(api_key = 'vkwbLYCgGUf5cuu0g1iGI3Mef')

movies = sys.argv[1]
users = sys.argv[2]
train = sys.argv[3]

def rmse(y_true, y_pred):
	y_pred = K.clip(y_pred, 1.0, 5.0)
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
 
def readData(moviesFile,usersFile,trainFile):
	# movies = pd.read_csv(moviesFile)
	# users = pd.read_csv(usersFile)
	train = pd.read_csv(trainFile).as_matrix()

	np.random.shuffle(train)

	users = train[:,1]
	movies = train[:,2]
	rate = train[:,3]

	# users -= 1
	# movies -= 1
	

	num_user = len(np.unique(users))
	num_movie = len(np.unique(movies))

	num_user1 = max(users)
	num_movie1 = max(movies)

	print(num_user,num_user1)
	print(num_movie,num_movie1)

	# print(train[0:5])
	return users,movies,rate,num_movie1,num_user1

def myModel(num_user,num_movie):

	latent_dim = 128

	user_input = Input(shape = [1])
	movie_input = Input(shape = [1])

	user_weight = Embedding(num_user,latent_dim)(user_input)
	user_weight = Flatten()(user_weight)
	user_weight = Dropout(0.5)(user_weight)
	
	movie_weight = Embedding(num_movie,latent_dim)(movie_input)
	movie_weight = Flatten()(movie_weight)
	movie_weight = Dropout(0.5)(movie_weight)

	# user_bias =  Embedding(num_user,1)(user_input)
	# user_bias = Flatten()(user_bias)
	
	# movie_bias = Embedding(num_movie,1)(movie_input)
	# movie_bias = Flatten()(movie_bias)

	ratings = Dot(axes = 1)([user_weight,movie_weight])
	# ratings =Add()([ratings,user_bias,movie_bias])
	# ratings =Add()([ratings])

	model = Model([user_input,movie_input],ratings)
	model.compile(loss='mse' , optimizer='adam' , metrics=[rmse])
	# model.compile(loss='mse' , optimizer='adam')

	return model

users,movies,rate,num_movie,num_user = readData(movies,users,train)

# mean = np.mean(rate)
# std = np.std(rate)

# print(rate)
# rate = (rate - mean)/std
# print(rate)

model = myModel(num_user,num_movie)
print(model.summary())

file_name = "-v{}.h5".format(ver)
check_point = ModelCheckpoint("./model/model-{epoch:05d}-{val_rmse:.5f}"+file_name,monitor='val_rmse',save_best_only=True)
early_stop = EarlyStopping(monitor="val_rmse", patience=3)

model.fit([users,movies],rate,
			epochs = 1000,
			batch_size = 128,
			# callbacks = [check_point,early_stop,reduce_lr],
			callbacks = [check_point,early_stop],
			validation_split=0.1
			)

'''
file_name = "-v{}.h5".format(ver)
check_point = ModelCheckpoint("./model/model-{epoch:05d}-{val_loss:.5f}"+file_name,monitor='val_loss',save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=0, min_lr=0)
model.fit([users,movies],rate,
			epochs = 150,
			batch_size = 128,
			callbacks = [check_point,early_stop,reduce_lr],
			# callbacks = [check_point,early_stop],
			validation_split=0.1
			)
'''