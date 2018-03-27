#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys

'''
train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
'''
train_xFile = "train_X"
train_yFile = "train_Y"
testFile = "test_X"
outputFile = "output.csv"

train_x = pd.read_csv(train_xFile,encoding='big5').as_matrix().astype('float')
train_y = pd.read_csv(train_yFile,encoding='big5',header = None).as_matrix().astype('float')

test_x = pd.read_csv(testFile,encoding='big5').as_matrix().astype('float')
'''
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
'''
def sigmoid(x):
	return 	1 / (1+np.exp(-x))
'''
for i in range(0,len(train_x[2])-1):
	if train_x[2,i] > 2:
		print(i)
'''
continus = [0, 10 , 80]


# print(w.shape)

train_x = np.concatenate((
		train_x,
		train_x[:,continus] ** 2,
		train_x[:,continus] ** 3,
	),axis=1)

test_x = np.concatenate((
		test_x,
		test_x[:,continus] ** 2,
		test_x[:,continus] ** 3,
	),axis=1)


x = (train_x - np.mean(train_x))/(np.std(train_x)+1e-20)
x = np.concatenate((np.ones((x.shape[0],1)),x),axis = 1)
x_test = (test_x - np.mean(train_x))/(np.std(train_x)+1e-20)
x_test = np.concatenate((np.ones((test_x.shape[0],1)),x_test),axis = 1)


w = np.zeros((train_x.shape[1]+1,1))
w_lr = 0
lr = 0.05
epoch = 3000


for e in range(1,epoch+1):
	predict = sigmoid(np.dot(x,w))
	error = train_y - predict

	grad = -np.dot(x.T,error)

	w_lr = w_lr + grad **2
	lr =  0.05 / np.sqrt(w_lr)
	w = w - lr * grad

	if e % 100 == 0:
		loss =  -np.mean(train_y * np.log(predict + 1e-20) + (1 - train_y) * np.log(1 - predict +1e-20))
		p = predict
		p[predict < 0.5] = 0.0
		p[predict >= 0.5] = 1.0
		acc = np.mean(1 - np.abs(train_y - p))
		print('[Epoch {:5d}] - training loss: {:.5f}, accuracy: {:.5f}'.format(e, loss, acc))


y_test = sigmoid(np.dot(x_test,w)) 
print(y_test)


with open(outputFile, 'w') as fout:
	print('id,label', file=fout)
	for (i, v) in enumerate(y_test.flatten()):
		print('{},{}'.format(i+1, 1 if v >= 0.5 else 0), file=fout)
