from svmutil import *
import numpy as np

print('Loading Train Data...')
data = []
with open('train.data') as f:
	data = [line.split(',') for line in f]

m = len(data)
n = len(data[0])-1
[X_temp,Y_temp] = np.split(np.array(data),[n],1)

X = [list(x) for x in np.ndarray.astype(X_temp,float)]
Y = [1 if y[0]=='ad.\n' else -1 for y in Y_temp]


print('Training SVM...')
prob = svm_problem(Y,X)
svm = svm_train(prob,'-t 0 -c 1 -g 2.5e-4')


print('Loading Test Data...')
testdata = []
with open('test.data') as f:
	testdata = [line.split(',') for line in f]

m_test = len(testdata)
n_test = len(testdata[0])-1
[X_test_temp,Y_test_temp] = np.split(np.array(testdata),[n],1)

X_test = [list(x) for x in np.ndarray.astype(X_test_temp,float)]
Y_test = [1 if y[0]=='ad.\n' else -1 for y in Y_test_temp]


print('Predicting...')
svm_predict(Y_test,X_test,svm)