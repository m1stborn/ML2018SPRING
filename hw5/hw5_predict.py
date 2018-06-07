#-*- coding: big5 -*-
import sys
import numpy as np
import pandas as pd
import gensim
import keras.preprocessing.text as T
import pickle
import re
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential,load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

test_data = sys.argv[1]
outputFile = sys.argv[2]
# train_nolabel = sys.argv[3]


# test_df = pd.read_csv(test_data)
with open(test_data,encoding="utf-8") as f:
	lines = f.readlines()

test_x = np.array( [ line[:-1].split(",",1)[1] for line in lines[1:] ] )

stemmer = gensim.parsing.porter.PorterStemmer()
test_x = [i for i in stemmer.stem_documents(test_x)]

test = test_x

with open('tokenizer_v136.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

word_index = tokenizer.word_index
# max_length = 37
seq = tokenizer.texts_to_sequences(test)
test =pad_sequences(seq,maxlen = 39)

model = load_model('./model/modelw2v-00011-0.82925-0.39304-v136.h5')
print(model.summary())
predict = model.predict(test)

ans = np.argmax(predict, axis=-1)

semi =False

if semi:
	# np.save('./train_nolabel_predict/train_nolabel_y-v{}'.format(ver),predict)
	threshold = 0.97
	add_index = np.where(
					np.logical_or(
						predict[:,0] > threshold,
						predict[:,0] < 1- threshold
 					)
				)
	print(np.logical_or(
						predict[:,0] > threshold,
						predict[:,0] < 1- threshold
 					)[0:10]
	)
	# np.save('',add_index)

	train_x_add = train_nolabel_x[add_index[0]]
	train_y_add = ans[add_index[0]]
	
	print(len(add_index))
	np.save('./train_nolabel_predict/train_x_add',train_x_add)
	np.save('./train_nolabel_predict/train_y_add',train_y_add)
else:
	with open(outputFile, 'w') as f:
		print('id,label', file=f)
		print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(ans)]), file=f)
