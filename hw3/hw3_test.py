import numpy as np
import pandas as pd
import sys
from keras.models import Sequential, load_model
import csv

# testFile = "test.csv"
# outputFile  = "output.csv"

testFile = sys.argv[1]
outputFile = sys.argv[2]

test_df = pd.read_csv(testFile)
x_test = np.array( [ list(map(float, test_df["feature"][i].split())) for i in range(len(test_df)) ] )
x_test = x_test.astype('float32')
x_test/=255
x_test = x_test.reshape( -1, 48, 48, 1)


modelEns = load_model("modelEns")
# model = load_model("model_v1.0")
# model = load_model("./model3/model-00168-0.67633.h5")
# model = load_model("./model5/model-00253-0.67433.h5")

predict = modelEns.predict(x_test)
predict = np.argmax(predict, axis=-1)

# model1 = load_model("./model/model-00253-0.67233.h5")
# model2 = load_model("./model/model-00146-0.67200.h5")
# model3 = load_model("./model/model-00132-0.66567.h5")
# predict = model.predict_classes(x_test)
# prob = 0.0
# prob = model1.predict(x_test)
# prob += model2.predict(x_test)
# prob += model3.predict(x_test)
# predict = np.argmax(prob, axis=-1)


# print(prob.shape)
# print(predict.shape)

# with open(outputFile, 'w') as f:
# 	csv_writer = csv.writer(f)
# 	csv_writer.writerow(['id', 'label'])
# 	for i in range(len(x_test)):
#  		csv_writer.writerow([i]+[predict[i]])
with open(outputFile, 'w') as f:
	print('id,label', file=f)
	print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(predict)]), file=f)