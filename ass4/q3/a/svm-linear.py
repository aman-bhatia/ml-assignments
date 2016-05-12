from __future__ import print_function
import pickle
import numpy as np
import pandas as pd
from sklearn import svm

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


print("Training...")
clf = svm.LinearSVC(C=100.0,tol=1e-7,dual=False)

clf.fit(Xtran, Ytran)
pkl_handle = open('clf.pkl','wb')
pickle.dump(clf,pkl_handle)
pkl_handle.close()

# pkl_handle = open('clf.pkl','rb')
# clf = pickle.load(pkl_handle)
# pkl_handle.close()


print("Predicting...")
print("Accuracy on Validatation Data 1 : ", clf.score(Xval1,Yval1))
print("Accuracy on Validatation Data 2 : ", clf.score(Xval2,Yval2))
print("Accuracy on Validatation Data 3 : ", clf.score(Xval3,Yval3))

prediction = clf.predict(Xtest)

f = open('prediction.csv','w')
f.write('ID,TARGET\n')
for i in range(len(prediction)):
	f.write(str(i)+','+str(prediction[i])+'\n')


