#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys


train_xFile = sys.argv[1]
train_yFile = sys.argv[2]
testFile = sys.argv[3]
outputFile = sys.argv[4]

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

'''
print(train_x.shape)
print(train_y)
train_x = train_x[:20000,:]
train_y = train_y[:20000]
print(train_x.shape)
print(train_y)
'''

test_x = pd.read_csv(testFile,encoding='big5').as_matrix().astype('float')

continus = [0,10,78,79,80]

mean = np.mean(train_x[:,continus],axis = 0)
std = np.std(train_x[:,continus],axis = 0)

train_x[:,continus] = (train_x[:,continus] - mean)/(std + 1e-20)
test_x[:,continus] = (test_x[:,continus] - mean)/(std + 1e-20)

def sigmoid(x):
	# return 	np.clip((1 / (1 + np.exp(-x))),0.00000000000001,0.99999999999999)
	return 1/(1+np.exp(-x))

size = train_x.shape[0]
feature = train_x.shape[1]
mu1 = np.zeros((feature,))
mu2 = np.zeros((feature,))
ctn1 = 0
ctn2 = 0
for  i in range(size):
	if train_y[i] == 1:
		mu1 += train_x[i]
		ctn1 += 1
	else:
		mu2 += train_x[i]
		ctn2 += 1
mu1  /= ctn1
mu2 /= ctn2

sigma1 = np.zeros((feature,feature))
sigma2 = np.zeros((feature,feature))

for i in range(train_x.shape[0]):
	if train_y[i] == 1:
		sigma1 += np.dot(np.transpose([train_x[i]-mu1]),[(train_x[i]-mu1)])
	else:	
		sigma2 += np.dot(np.transpose([train_x[i]-mu2]),[(train_x[i]-mu2)])
sigma1 /= ctn1
sigma2 /= ctn2

shared_sigma = (ctn1/size)*sigma1 + (ctn2/size)*sigma2


def predict(test_x,mu1,mu2,shared_sigma,n1,n2):
	sigma_inverse = np.linalg.pinv(shared_sigma)
	w = np.dot((mu1-mu2),sigma_inverse)
	x = test_x.T
	b = (-0.5)*np.dot(np.dot([mu1],sigma_inverse),mu1)+(0.5)*np.dot(np.dot([mu2],sigma_inverse),mu2)+np.log(n1/n2)
	a = np.dot(w,x)
	# print(np.dot(w,x))
	# print(b)
	# print(a)
	y = sigmoid(a)
	# return y
	return  [1 if p >= 0.9 else 0 for p in y]

pred = predict(train_x,mu1,mu2,shared_sigma,ctn1,ctn2)
# pred = np.around(1.0-pred)
result = (train_y.flatten()==pred)
fail = []
for i in range(len(pred)):
	if pred[i] != train_y[i]:
		fail.append(i)
print(len(fail)/size)
print(np.mean(result))

y_test = predict(test_x,mu1,mu2,shared_sigma,ctn1,ctn2)

with open(outputFile, 'w') as fout:
	print('id,label', file=fout)
	# for (i, v) in enumerate(y_test.flatten()):
	for (i, v) in enumerate(y_test):
		print('{},{}'.format(i+1, 1 if v == 1 else 0), file=fout)
		# print('{},{}'.format(), file=fout)

