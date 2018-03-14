#-*- coding: big5 -*-
import numpy as np
import pandas as pd

# df = np.genfromtxt('train.csv',delimiter=',')
# print(df[1,3])
# print(len(df))
# print(df[0])
# X=[df[,3]=='PM2.5']
# x=[]
# y=[]
# for i in range(1,len(df)):
# 	if i%18==10:
# 		x.append(df[i,3:26])
# 	else:
# 		y.append(df[i,3:26])		
# print(len(x))
# print(len(y))
# print(x[0:5])
# print(y[0:5])
# print(x[0][1])

df = pd.read_csv('train.csv',encoding='big5').as_matrix()
data = df[:, 3:]
# print(data[0])
data[data=='NR'] = 0.0
data = data.astype('float')
# print(data.shape[0])
# print(data.shape[1])



x = []
y = []
for i in range(0, data.shape[0], 18*20):
		# i: start of each month
		days = np.vsplit(data[i:i+18*20], 20) # shape: 20 * (18, 24)
		concat = np.concatenate(days, axis=1) # shape: (18, 480)

		for j in range(0, concat.shape[1]-9): #shape[1] = 480
			x.append(concat[:, j:j+9].flatten())
			y.append([concat[9, j+9]])
x = np.array(x)
y = np.array(y)			
# print(x.shape[0])
# print(x.shape[1])
# print(y.shape[0])
# print(y.shape[1])
# print(np.zeros((x.shape[1],1)))

df_test = pd.read_csv('test.csv',header=None,encoding='big5').as_matrix()
data_test = df_test[:,2:]
data_test[data_test == 'NR'] = 0.0
data_test = data_test.astype('float')
data_test = np.vsplit(data_test,data_test.shape[0]/18)
x_test = []
for i in data_test:
	x_test.append(i.flatten())
x_test = np.array(x_test)	

b = 0.0
w = np.ones((x.shape[1] , 1))
b_lr = 0.0
w_lr = np.zeros((x.shape[1] , 1))
lr = 0.5
epoch = 35000

# x = (x - np.min(x,axis=0)) / (np.max(x,axis=0) - np.min(x,axis=0))

for e in range(epoch):
	error = y - b - np.dot(x,w)

	b_gard = -np.sum(error)/x.shape[1]
	w_gard = -np.dot(x.T,error)/x.shape[1]
	# b_gard = -2*np.sum(error)
	# w_gard = -2*np.dot(x.T,error)

	b_lr = b_lr + b_gard ** 2
	w_lr = w_lr + w_gard ** 2

	loss = np.mean(np.square(error))

	b = b - lr/np.sqrt(b_lr) * b_gard
	w = w - lr/np.sqrt(w_lr) * w_gard
	if e % 1000 == 0:
		print('[Epoch {}]: loss: {}'.format(e, loss))

# print(b)
# print(w)
y_test = np.dot(x_test,w)+b_gard
# print(y_test)

with open('submit.csv', 'w') as f:
	print('id,value', file=f)
	for (i, p) in enumerate(y_test) :
		print('id_{},{}'.format(i, p[0]), file=f)
