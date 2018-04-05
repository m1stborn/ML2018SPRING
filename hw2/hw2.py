#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys


train_xFile = sys.argv[3]
train_yFile = sys.argv[4]
testFile = sys.argv[5]
outputFile = sys.argv[6]
rawdata = sys.argv[1]
rawtest = sys.argv[2]

'''
train_xFile = "X_train"
train_yFile = "Y_train"
testFile = "X_test"
''' 
'''
train_xFile = "train_X"
train_yFile = "train_Y"
testFile = "test_X"

outputFile = "output.csv"
'''
train_x = pd.read_csv(train_xFile,encoding='big5').as_matrix().astype('float')
train_y = pd.read_csv(train_yFile,encoding='big5',header = None).as_matrix().astype('float')

test_x = pd.read_csv(testFile,encoding='big5').as_matrix().astype('float')
out_index = [3593, 4568, 5385, 6590, 7741, 8897, 12530, 12973, 13201, 15376, 15567, 15966, 17039, 18270, 20091, 20176, 21475, 23273, 25149, 26492, 26681, 29187, 31512, 32214]

'''
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
'''
def sigmoid(x):
	return 	1 / (1 + np.exp(-x))
'''	
x = {}
for j in range(0,train_x.shape[0]-1):
	for i in range(0,len(train_x[j])-1):
		if train_x[j,i] > 2:
			x[i] = 1
print(x)
'''
con = [0,10,78,79,80]
continus = [0, 10 ,78 ,79 ,80,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142]
# print(train_x[out_index,])
# print(train_x.shape)
# print(w.shape)

train_x = np.concatenate((
		train_x,
		train_x[:,con] ** 2,
		train_x[:,con] ** 3,
		train_x[:,con] ** 4,
		train_x[:,con] ** 5,
	),axis=1)

test_x = np.concatenate((
		test_x,
		test_x[:,con] ** 2,
		test_x[:,con] ** 3,
		test_x[:,con] ** 4,
		test_x[:,con] ** 5,
	),axis=1)

'''
a = {}
for j in range(0,train_x.shape[0]-1):
	for i in range(0,len(train_x[j])-1):
		if train_x[j,i] > 2:
			a[i] = 1
print(a)
'''
mean = np.mean(train_x[:,continus],axis = 0)
std = np.std(train_x[:,continus],axis = 0)

train_x[:,continus] = (train_x[:,continus] - mean)/(std + 1e-20)
test_x[:,continus] = (test_x[:,continus] - mean)/(std + 1e-20)
x = np.concatenate((np.ones((train_x.shape[0],1)),train_x),axis = 1)
x_test = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis = 1)

'''
out_index = [3593, 4568, 5385, 6590, 7741, 8897, 12530, 12973, 13201, 15376, 15567, 15966, 17039, 18270, 20091, 20176, 21475, 23273, 25149, 26492, 26681, 29187, 31512, 32214]
x = np.delete(x,out_index,0)
train_y = np.delete(train_y,out_index,0)

print(len(x))
print(len(train_y))
'''
# x = (train_x - np.mean(train_x,axis = 0))/(np.std(train_x,axis = 0)+1e-20)
# x = np.concatenate((np.ones((x.shape[0],1)),x),axis = 1)
# x_test = (test_x - np.mean(train_x,axis = 0))/(np.std(train_x,axis = 0)+1e-20)
# x_test = np.concatenate((np.ones((test_x.shape[0],1)),x_test),axis = 1)


w = np.zeros((train_x.shape[1]+1,1))
w_lr = 0
lr = 0.01
epoch = 5000
outlier = []

big = []
small = []

# j = np.ones((x.shape[1],1))


for e in range(1,epoch+1):
	predict = sigmoid(np.dot(x,w))
	error = train_y - predict

	grad = -np.dot(x.T,error)

	w_lr = w_lr + grad **2
	lr =  0.05 / np.sqrt(w_lr)
	# w = w - lr * grad -lr*0.001*j
	w = w - lr * grad

	if e % 100 == 0:
		loss =  -np.mean(train_y * np.log(predict + 1e-20) + (1 - train_y) * np.log(1 - predict +1e-20))
		p = predict
		p[predict < 0.5] = 0.0
		p[predict >= 0.5] = 1.0
		acc = np.mean(1 - np.abs(train_y - p))
		print('[Epoch {:5d}] - training loss: {:.5f}, accuracy: {:.5f}'.format(e, loss, acc))

predict = sigmoid(np.dot(x,w))
# # print(predict)
# for i in range(len(p)):
# 	if predict[i] > 0.99 and train_y[i] == 0:
# 	# if predict[i] >= 0.99 :
# 		# outlier.append(i)
# 		big.append(i)
# 	elif predict[i] <= 0.01 and train_y[i] == 1:
# 	# elif predict[i] <= 0.01 :
# 		# outlier.append(i)
# 		small.append(i)
# print(len(big))
# print(outlier)
# print(len(small))
y_test = sigmoid(np.dot(x_test,w))  
# print(y_test)

with open(outputFile, 'w') as fout:
	print('id,label', file=fout)
	for (i, v) in enumerate(y_test.flatten()):
		print('{},{}'.format(i+1, 1 if v >= 0.5 else 0), file=fout)
