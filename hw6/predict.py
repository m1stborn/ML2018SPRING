import sys
import numpy as np
import pandas as pd

import keras.backend as K
from keras.models import Model,load_model
from keras.layers import Embedding,Input,Dot,Add,Concatenate,Dropout,Dense,Flatten
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau


def rmse(y_true, y_pred):
	y_pred = K.clip(y_pred, 1.0, 5.0)
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


testFile = sys.argv[1]
outputFile = sys.argv[2]

test = pd.read_csv(testFile).as_matrix()

user = test[:,1]
movie = test[:,2] 

model = load_model('model-00027-0.67851-v9.h5',custom_objects={'rmse': rmse})

predict = model.predict([user,movie])
# predict = (predict+mean)*std
predict = predict.clip(1.0, 5.0)
with open(outputFile, 'w') as f:
	print('TestDataID,Rating', file=f)
	print('\n'.join(['{},{}'.format(i+1, p[0]) for (i, p) in enumerate(predict)]), file=f)
