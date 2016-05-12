from __future__ import print_function
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

print('Loading Data...')
data = pd.read_csv('./data/train.csv', header=None)
Xtran = np.ascontiguousarray(data.drop(0,1).values)
Ytran = np.ascontiguousarray(data[0].values)

data = pd.read_csv('./data/validation_1.csv', header=None)
Xval1 = np.ascontiguousarray(data.drop(0,1).values)
Yval1 = np.ascontiguousarray(data[0].values)

data = pd.read_csv('./data/validation_2.csv', header=None)
Xval2 = np.ascontiguousarray(data.drop(0,1).values)
Yval2 = np.ascontiguousarray(data[0].values)

data = pd.read_csv('./data/validation_3.csv', header=None)
Xval3 = np.ascontiguousarray(data.drop(0,1).values)
Yval3 = np.ascontiguousarray(data[0].values)

data = pd.read_csv('./data/testfile.csv', header=None)
Xtest = np.ascontiguousarray(data.drop(0,1).values)


print("Loading Model...")
gnb = GaussianNB()

pkl_handle = open('gnb.pkl','rb')
gnb = pickle.load(pkl_handle)
pkl_handle.close()


print("Predicting...")
print("Accuracy on Train Data          : ", gnb.score(Xtran,Ytran))
print("Accuracy on Validatation Data 1 : ", gnb.score(Xval1,Yval1))
print("Accuracy on Validatation Data 2 : ", gnb.score(Xval2,Yval2))
print("Accuracy on Validatation Data 3 : ", gnb.score(Xval3,Yval3))


prediction = gnb.predict(Xtest)

f = open('prediction-nb.csv','w')
f.write('ID,TARGET\n')
for i in range(len(prediction)):
	f.write(str(i)+','+str(prediction[i])+'\n')

