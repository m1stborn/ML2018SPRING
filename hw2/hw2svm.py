#-*- coding: big5 -*-
import numpy as np
import pandas as pd
import sys
from sklearn.svm import SVC
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

continus = [0,10,78,79,80]

mean = np.mean(train_x[:,continus],axis = 0)
std = np.std(train_x[:,continus],axis = 0)

train_x[:,continus] = (train_x[:,continus] - mean)/(std + 1e-20)
test_x[:,continus] = (test_x[:,continus] - mean)/(std + 1e-20)


clf = SVC()
clf.fit(train_x,train_y.flatten())

pred = clf.predict(train_x)
result = (train_y.flatten()==pred)
print(np.mean(result))

y_test = clf.predict(test_x)

with open(outputFile, 'w') as fout:
	print('id,label', file=fout)
	# for (i, v) in enumerate(y_test.flatten()):
	for i in range(len(y_test)):
		print('{},{}'.format(i+1, int(y_test[i])), file=fout)

