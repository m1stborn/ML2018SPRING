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

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
