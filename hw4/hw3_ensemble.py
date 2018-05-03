import numpy as np
import pandas as pd
import sys
from keras.models import Sequential, load_model
import csv
from keras.models import Input
from keras import layers
from keras.models import Model


testFile = "test.csv"
outputFile  = "output.csv"

test_df = pd.read_csv(testFile)
x_test = np.array( [ list(map(float, test_df["feature"][i].split())) for i in range(len(test_df)) ] )
x_test/=255
x_test = x_test.reshape( -1, 48, 48, 1)

model1 = load_model("./model/model-00253-0.67233.h5")
model2 = load_model("./model/model-00146-0.67200.h5")
# model5 = load_model("./model/model-00132-0.66567.h5")
model3 = load_model("./model3/model-00168-0.67633.h5")
model4 = load_model("./model4/model-00252-0.68433.h5")
model5 = load_model("./model4/model-00248-0.68267.h5")
model6 = load_model("./model3/model-00166-0.66733.h5")
model7 = load_model("./model5/model-00158-0.67400.h5")
model8 = load_model("./model5/model-00157-0.67367.h5")
def ensembleModels(models, model_input):
	# collect outputs of models in a list
	yModels=[model(model_input) for model in models] 
	# averaging outputs
	yAvg=layers.average(yModels) 
	# build model from same input and avg output
	modelEns = Model(inputs=model_input, outputs=yAvg,    name='ensemble')  
   
	return modelEns

models = [model1,model2,model3,model4,model5,model6,model7,model8]
# models = [model1,model3,model4,model7]
model_input = Input(shape=models[0].input_shape[1:])

modelEns = ensembleModels(models, model_input)
# model.summary()
modelEns.compile()
modelEns.save("./model/modelEns")

modelEns = load_model("./model/modelEns")

modelEns.summary()

predict = modelEns.predict(x_test)
predict = np.argmax(predict, axis=-1)
# print(y.shape)
with open(outputFile, 'w') as f:
	print('id,label', file=f)
	print('\n'.join(['{},{}'.format(i, p) for (i, p) in enumerate(predict)]), file=f)


train_df = pd.read_csv("train.csv")

valid_df = train_df.iloc[:3000]
train_df = train_df.iloc[3000:]

x_valid = np.array( [ list(map(float, valid_df["feature"].iloc[i].split())) for i in range(len(valid_df)) ] )
x_valid = x_valid.astype('float32')
x_valid = x_valid.reshape( -1, 48, 48, 1)
x_valid /= 255

y_valid = np.array( valid_df["label"] )

predict = modelEns.predict(x_valid)

predict = np.argmax(predict, axis=-1)

result = (y_valid.flatten()==predict)

print(np.mean(result))