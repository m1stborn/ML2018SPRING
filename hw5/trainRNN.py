#-*- coding: big5 -*-
import sys
import numpy as np
import pandas as pd
import gensim
import pickle
import re

from comet_ml import Experiment

import keras.preprocessing.text as T 
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Bidirectional, BatchNormalization,normalization,GRU
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.regularizers import l2

# experiment = Experiment(api_key = 'vkwbLYCgGUf5cuu0g1iGI3Mef')

train_label = sys.argv[1]
train_nolabel = sys.argv[2]


with open(train_label,encoding="utf-8") as f:
	lines = f.readlines()
	# lines =f.read().splitlines()
train_label_x = [line[:-1].strip().split(" +++$+++ ")[1] for line in lines]
train_label_y = [line[:-1].strip().split(" +++$+++ ")[0] for line in lines]
# print(np.array(train_label_x).shape)

valid_label_x = train_label_x[:4000]
train_label_x = train_label_x[4000:]

def preprocess(train_label_x):
	for i in range(len(train_label_x)):
		# train_label_x[i] = train_label_x[i].replace(" ' ", "")
		# train_label_x[i] = train_label_x[i].replace("what ' s", "what is")
		train_label_x[i] = train_label_x[i].replace("' ve", "have")
		# train_label_x[i] = train_label_x[i].replace("can ' t", "can not")
		train_label_x[i] = train_label_x[i].replace("n ' t", " not")
		train_label_x[i] = train_label_x[i].replace("i ' m", "i am")
		train_label_x[i] = train_label_x[i].replace("' re", "are")
		train_label_x[i] = train_label_x[i].replace("' d", "would")
		train_label_x[i] = train_label_x[i].replace("' ll", "will")
		train_label_x[i] = train_label_x[i].replace("im", "i am")	
	return train_label_x

# train_label_x = preprocess(train_label_x)
# valid_label_x = preprocess(valid_label_x)

stemmer = gensim.parsing.porter.PorterStemmer()
train_label_x = [i for i in stemmer.stem_documents(train_label_x)]
valid_label_x = [i for i in stemmer.stem_documents(valid_label_x)]
# train_x_add = np.load('./train_nolabel_predict/train_x_add.npy').tolist()
# train_y_add = np.load('./train_nolabel_predict/train_y_add.npy').tolist()

# print(len(train_x_add))
# print(len(train_y_add))

# train_label_x = train_label_x + train_x_add
# train_label_y = train_label_y + train_y_add

# print(len(train_label_x))
# print(len(train_label_y))

# print(np.array(train_label_x).shape)
# print(np.array(train_label_y).shape)

# print(train_label_x[-10:])
# print(train_label_y[-10:])


###################################################################
tokenizer_test = Tokenizer(
				# num_words = num_word,
				# filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
				filters = '\n\t',
				# filters='"#$%&()*+,-/:;<=>@[\]^_`{|}~\t\n',
				# filters='"#$%&()*+,-/:;<=>@[\]^_`{|}\t\n',
				lower=True,
				split=" ",
				char_level=False
				)
tokenizer_test.fit_on_texts(train_label_x)
word_index_test = tokenizer_test.word_index
word_count_test = tokenizer_test.word_counts

def count(word_count,word_index,num):
	total = 0
	for  word,i in word_index.items():
		if word_count[word] >= num :
			total += 1
	return total
count_array = []
# c = np.arange(15,-1,-3)
# c = np.arange(10,-1,-3)
# c = np.arange(10,16,3)
c = [10]
print(c)
for i in c:
	count_array.append(count(word_count_test,word_index_test,i))

for i,j in zip(c,count_array):
	print(i,",",j)

# for i in range(len(train_label_x)):
	# train_label_x[i]=T.text_to_word_sequence(train_label_x[i],filters='\n\t')

###################################################################
def myMain(ver,min_count,num_word,train_label_x,train_label_y,valid_label_x,semi=False,ttws=True):
	print('Starting:[ver:{},min_count:{},num_word:{}]'.format(ver,min_count,num_word))

	tokenizer = Tokenizer(
					# num_words = num_word,
					# filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
					filters = '\n\t',
					# filters='"#$%&()*+,-/:;<=>@[\]^_`{|}~\t\n',
					# filters='"#$%&()*+,-/:;<=>@[\]^_`{|}\t\n',
					# lower=True,
					# split=" ",
					# char_level=False
					)
	tokenizer.fit_on_texts(train_label_x)
	word_index = tokenizer.word_index
	seq = tokenizer.texts_to_sequences(train_label_x)
	word_count = tokenizer.word_counts

	with open('tokenizer_v{}.pickle'.format(ver), 'wb') as handle:
		pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# for i in range(len(train_label_x)):
		# train_label_x[i]=T.text_to_word_sequence(train_label_x[i],filters='\n\t')

	# train_x_seq = [T.text_to_word_sequence(i,filters='\n\t') for i in train_label_x]
	if ttws:
		train_x_seq = [T.text_to_word_sequence(i,filters='\n\t') for i in train_label_x]
	else:
		train_x_seq = train_label_x

	w2v_model = gensim.models.Word2Vec(train_x_seq,size=200, window=5, min_count=min_count, workers=8,iter=25)
	w2v_model.save('w2v_model{}'.format(ver))

	emb_matrix = np.zeros((len(word_index),200))
	out = 0
	total = 0
	for  word,i in word_index.items():
		try:
			emb_matrix[i] = w2v_model.wv[word]
		except:
			out +=1
	# print(emb_matrix[0:10])
	# print(np.array(seq).shape)
	# print(seq[0])


	max_len = np.max([len(s) for s in seq])
	print('max_len:' , max_len)
	model = Sequential()
	model.add(Embedding(len(word_index),
						output_dim=200,
						weights = [emb_matrix],	
						input_length = max_len,
						mask_zero = True,
						trainable=False,
						))
	model.add(SpatialDropout1D(0.4))
	# model.add(LSTM(128,return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
	# model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
	model.add(Bidirectional(LSTM(128,return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
	model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
	# model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
	# model.add(GRU(512,dropout=0.5,recurrent_dropout=0.5,return_sequences=True))
	# model.add(GRU(512,dropout=0.5,recurrent_dropout=0.5))
	model.add(normalization.BatchNormalization())
	# model.add(Dense(512,activation='selu'))
	# model.add(Dense(512,activation='selu'))
	# model.add(Dropout(0.3))
	# model.add(Dense(256,activation='sigmoid'))
	# model.add(Dense(128,activation='sigmoid'))
	# model.add(Dense(32,activation='relu',kernel_regularizer=l2(0.01)))
	model.add(Dense(32,activation='relu'))
	# model.add(Dropout(0.3))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
	# model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
	# print(model.summary())

	if semi:
		train_x_add =  tokenizer.texts_to_sequences(train_x_add)
		seq = np.concatenate((seq,train_x_add),axis=0)
		train_label_y = train_label_y + train_y_add
		print(len(train_label_y))
		print(seq.shape)

	x = pad_sequences(seq,maxlen = max_len)
	y = to_categorical(train_label_y)

	train_x = x

	valid_seq = tokenizer.texts_to_sequences(valid_label_x)
	valid_x = pad_sequences(valid_seq,maxlen = max_len)

	train_y = y[4000:]
	valid_y = y[:4000]

	file_name = "-{}-v{}.h5".format(min_count,ver)

	check_point = ModelCheckpoint("./model/modelw2v-{epoch:03d}-{val_acc:.5f}-{val_loss:.5f}"+file_name,monitor='val_acc',save_best_only=True)
	early_stop = EarlyStopping(monitor="val_loss", patience=2)
	# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=2, min_lr=0.001)
	model.fit(train_x,train_y,
			validation_data = (valid_x, valid_y),
			epochs = 30,
			batch_size = 64,
			# callbacks = [check_point,early_stop,reduce_lr],
			callbacks = [check_point,early_stop]
			)
	print('Ending:[ver:{},min_count:{},num_word:{}]'.format(ver,min_count,num_word))

ver = 999
myMain(ver,10,count_array[0],train_label_x,train_label_y,valid_label_x)
