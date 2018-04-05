#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
'''
train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]
'''

'''
train_xFile = "X_train"
train_yFile = "Y_train"
testFile = "X_test"
''' 

train_xFile = "train_X"
train_yFile = "train_Y"
testFile = "test_X"

outputFile = "output.csv"

train_x = pd.read_csv(train_xFile,encoding='big5').as_matrix().astype('float')
train_y = pd.read_csv(train_yFile,encoding='big5',header = None).as_matrix().astype('float')

train_xl = train_x


test_x = pd.read_csv(testFile,encoding='big5').as_matrix().astype('float')

test_xl = test_x

continus = [0,10,78,79,80]

mean = np.mean(train_x[:,continus],axis = 0)
std = np.std(train_x[:,continus],axis = 0)

train_x[:,continus] = (train_x[:,continus] - mean)/(std + 1e-20)
test_x[:,continus] = (test_x[:,continus] - mean)/(std + 1e-20)

# out_index = [3593, 4568, 5385, 6590, 7741, 8897, 12530, 12973, 13201, 15376, 15567, 15966, 17039, 18270, 20091, 20176, 21475, 23273, 25149, 26492, 26681, 29187, 31512, 32214]
# train_x = np.delete(train_x,out_index,0)
# train_y = np.delete(train_y,out_index,0)


######################################################
print("part1:")
clf = SVC()
clf.fit(train_x,train_y.flatten())

pred = clf.predict(train_x)
result = (train_y.flatten()==pred)
print(np.mean(result))
y_test =  clf.predict(test_x)
######################################################
print("part2:")
clf2  = RandomForestClassifier(max_depth=17)

clf2.fit(train_x,train_y.flatten())

pred2 = clf2.predict(train_x)
result2 = (train_y.flatten()==pred2)
print(np.mean(result2))

y_test2 =  clf2.predict(test_x)
# y_test = np.array(y_test)
# print(len(test_x))
# print(len(y_test))
######################################################
print("part3:")

def sigmoid(x):
	return 	1 / (1 + np.exp(-x))



out_index = [3593, 4568, 5385, 6590, 7741, 8897, 12530, 12973, 13201, 15376, 15567, 15966, 17039, 18270, 20091, 20176, 21475, 23273, 25149, 26492, 26681, 29187, 31512, 32214]
train_x = np.delete(train_xl,out_index,0)
train_y = np.delete(train_y,out_index,0)

con = [0,10,78,79,80]
continus = [0, 10 ,78 ,79 ,80,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142]


train_x = np.concatenate((
		train_x,
		train_x[:,con] ** 2,
		train_x[:,con] ** 3,
		train_x[:,con] ** 4,
		train_x[:,con] ** 5,
	),axis=1)

test_xl = np.concatenate((
		test_xl,
		test_xl[:,con] ** 2,
		test_xl[:,con] ** 3,
		test_xl[:,con] ** 4,
		test_xl[:,con] ** 5,
	),axis=1)

mean = np.mean(train_x[:,continus],axis = 0)
std = np.std(train_x[:,continus],axis = 0)

train_x[:,continus] = (train_x[:,continus] - mean)/(std + 1e-20)
test_xl[:,continus] = (test_xl[:,continus] - mean)/(std + 1e-20)
x = np.concatenate((np.ones((train_x.shape[0],1)),train_x),axis = 1)
x_test = np.concatenate((np.ones((test_xl.shape[0],1)),test_xl),axis = 1)

w = np.zeros((train_x.shape[1]+1,1))
w_lr = 0
lr = 0.05
epoch = 5000

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


y_test3 = sigmoid(np.dot(x_test,w))
y_test3 = y_test3.flatten()
y_test3 = y_test3.tolist()
y_vote = []
for i in range(len(x_test)):
	if np.mean([y_test[i],y_test2[i],y_test3[i]]) > 0.5:
	# if np.mean([y_test3[i]]) > 0.5:
		y_vote.append(1)
	else:
		y_vote.append(0)

with open(outputFile, 'w') as fout:
	print('id,label', file=fout)
	# for (i, v) in enumerate(y_test.flatten()):
	for i in range(len(y_vote)):
		print('{},{}'.format(i+1, int(y_vote[i])), file=fout)
