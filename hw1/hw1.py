#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys
# import matplotlib.pyplot as plt

train = sys.argv[1]
test = sys.argv[2]
output = sys.argv[3]

df = pd.read_csv(train,encoding='big5').as_matrix()
data = df[:, 3:]
data[data=='NR'] = 0.0
data = data.astype('float')

x = []
y = []
hours = 9
for i in range(0, data.shape[0], 18*20):
		# i: start of each month
		days = np.vsplit(data[i:i+18*20], 20) # shape: 20 * (18, 24)
		concat = np.concatenate(days, axis=1) # shape: (18, 480)

		for j in range(0, concat.shape[1]-hours): #shape[1] = 480
			x.append(concat[:, j:j+hours].flatten())
			y.append([concat[9, j+hours]])
x = np.array(x)
y = np.array(y)


df_test = pd.read_csv(test,header=None,encoding='big5').as_matrix()
data_test = df_test[:,2:]
data_test[data_test == 'NR'] = 0.0
data_test = data_test.astype('float')
data_test = np.vsplit(data_test[:,9-hours:],data_test.shape[0]/18)
x_test = []

for i in data_test:
	x_test.append(i.flatten())
x_test = np.array(x_test)

attrs = ['AMB', 'CH4', 'CO', 'NMHC', 'NO', 'NO2','NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH','SO2', 'THC', 'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR']
attr_range = {}

for i, attr in enumerate(attrs):
	attr_range[attr] = list(range(hours*i, hours*i+hours))

select_attr = ['PM10', 'PM2.5', 'O3', 'CO', 'SO2', 'RAINFALL']
select_range = []
for attr in select_attr:
	select_range += attr_range[attr]

x = x[:,select_range]
x_test = x_test[:,select_range]

out_index = []

# for e in range(0,len(x)-1):
# 	for i in range(len(x[e])-1):
# 		if abs(x[e,i] - np.mean(x[:,i])) > 13 * np.std(x[:,i]):
# 			out_index.append(e)
# 			break

# x = np.delete(x,out_index,0)
# y = np.delete(y,out_index,0)

# j = np.ones((x.shape[1] , 1))


def adagrad(x,y,x_test,lr = 1,epoch = 35000):
	print('lr:{} epoch:{}'.format(lr,epoch))
	b = 0.0
	w = np.ones((x.shape[1] , 1))
	b_lr = 0.0
	w_lr = np.zeros((x.shape[1] , 1))
	
	# loss = []

	for e in range(epoch):
		error = y - b - np.dot(x,w)

		b_gard = -np.sum(error)
		w_gard = -np.dot(x.T,error)

		b_lr = b_lr + b_gard ** 2
		w_lr = w_lr + w_gard ** 2 

		
		loss1 = np.sqrt(np.mean(np.square(error))) 
		
		b = b - lr/np.sqrt(b_lr) * b_gard
		w = w - lr/np.sqrt(w_lr) * w_gard 
		if e % 1000 == 0:
			print('[Epoch {}]: loss1: {}'.format(e, loss1))
		# loss.append(loss1)
	return	np.dot(x_test,w) + b
	# return loss
# y_test = adagrad(x,y,x_test,lr=1,epoch = 35000)

y_test = adagrad(x,y,x_test,lr=1,epoch = 35000)
# y_test2 = adagrad(x,y,x_test,lr=0.05,epoch = 8000)
# y_test3 = adagrad(x,y,x_test,lr=0.025,epoch = 8000)
# y_test4 = adagrad(x,y,x_test,lr=0.0125,epoch = 8000)	


# plt.plot(y_test)
# plt.plot(y_test2)
# plt.plot(y_test3)
# plt.plot(y_test4)
# plt.ylim(0,5000)
# plt.xlim(0,1500)
# plt.legend( labels = ['lr=0.1', 'lr=0.05','lr=0.025','lr=0.0125'], loc = 'best')
# plt.show()

with open(output, 'w') as f:
	print('id,value', file=f)
	for (i, p) in enumerate(y_test) :
		print('id_{},{}'.format(i, p[0]), file=f)
