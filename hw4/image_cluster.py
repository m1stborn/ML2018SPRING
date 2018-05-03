import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.layers import Input, Dense
from keras.models import Model,load_model
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt 


# imgFile = "image.npy"
# testFile = "test_case.csv"
# outputFile = "model_32#6_check.csv"

imgFile = sys.argv[1]
testFile = sys.arge[2]
outputFile = sys.argv[3]



image_x = np.load(imgFile)
image_x = image_x
image_x = image_x/255.0

encoder = load_model("model_32#6")

encoded_imgs = encoder.predict(image_x)


print(encoded_imgs.shape)
# print(int(sys.argv[1]))

cluster = KMeans(n_clusters=2).fit(encoded_imgs)
# cluster.fit(encoded_imgs)
# print(cluster.labels_[:10])

test_case = pd.read_csv(testFile)

predict = []
for a, b in zip(test_case["image1_index"],test_case["image2_index"]):
	if cluster.labels_[a] == cluster.labels_[b]:
		predict.append(1)
	else:
		predict.append(0)



with open(outputFile, 'w') as f:
	print('ID,Ans', file=f)
	print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(predict)]), file=f)

print(np.mean(predict))
print(len(predict))		


