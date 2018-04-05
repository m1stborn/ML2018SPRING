#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

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

test_x = pd.read_csv(testFile,encoding='big5').as_matrix().astype('float')

# out_index = [3593, 4568, 5385, 6590, 7741, 8897, 12530, 12973, 13201, 15376, 15567, 15966, 17039, 18270, 20091, 20176, 21475, 23273, 25149, 26492, 26681, 29187, 31512, 32214]
# train_x = np.delete(train_x,out_index,0)
# train_y = np.delete(train_y,out_index,0)

continus = [0,10,78,79,80]
mean = np.mean(train_x[:,continus],axis = 0)
std = np.std(train_x[:,continus],axis = 0)

train_x[:,continus] = (train_x[:,continus] - mean)/(std + 1e-20)
test_x[:,continus] = (test_x[:,continus] - mean)/(std + 1e-20)


deep = int(sys.argv[1])
score = []

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(train_x):
	x_train, x_test = train_x[train_index], train_x[test_index]
	y_train, y_test = train_y[train_index], train_y[test_index]

	clf  = RandomForestClassifier(max_depth=deep)
	clf.fit(x_train,y_train.flatten())
	pred = clf.predict(x_test)
	result = (y_test.flatten()==pred)
	# print(np.mean(result))
	score.append(np.mean(result))

print(np.mean(score))

'''
# deep = int(sys.argv[1])
deep = 17
print(deep)
clf  = RandomForestClassifier(max_depth=deep)

clf.fit(train_x,train_y.flatten())

pred = clf.predict(train_x)
result = (train_y.flatten()==pred)
print(np.mean(result))

y_test =  clf.predict(test_x)
# y_test = np.array(y_test)
# print(len(test_x))
# print(len(y_test))

with open(outputFile, 'w') as fout:
	print('id,label', file=fout)
	# for (i, v) in enumerate(y_test.flatten()):
	for i in range(len(y_test)):
		print('{},{}'.format(i+1, int(y_test[i])), file=fout)
'''