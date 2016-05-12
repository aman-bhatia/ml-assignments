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
Xtran = np.r_[Xtran,np.ascontiguousarray(data.drop(0,1).values)]
Ytran = np.r_[Ytran,np.ascontiguousarray(data[0].values)]

data = pd.read_csv('./data/validation_2.csv', header=None)
Xtran = np.r_[Xtran,np.ascontiguousarray(data.drop(0,1).values)]
Ytran = np.r_[Ytran,np.ascontiguousarray(data[0].values)]

data = pd.read_csv('./data/validation_3.csv', header=None)
Xtran = np.r_[Xtran,np.ascontiguousarray(data.drop(0,1).values)]
Ytran = np.r_[Ytran,np.ascontiguousarray(data[0].values)]


print("Training...")
clf = svm.SVC(gamma=0.02)
clf.fit(Xtran, Ytran)

print("Saving Model...")
pkl_handle = open('clf.pkl','wb')
pickle.dump(clf,pkl_handle)
pkl_handle.close()

print("Done!")