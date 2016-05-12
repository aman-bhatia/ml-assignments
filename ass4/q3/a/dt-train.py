from __future__ import print_function
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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
dtree = DecisionTreeClassifier(criterion="entropy", max_depth=21)

dtree.fit(Xtran,Ytran)
pkl_handle = open('dtree.pkl','wb')
pickle.dump(dtree,pkl_handle)
pkl_handle.close()




#############################################################################
# Uncomment the code below to plot graph of accuracies vs depth of the tree #
#############################################################################

# var_depth = 10

# tran_accs = np.zeros(var_depth)
# val1_accs = np.zeros(var_depth)
# val2_accs = np.zeros(var_depth)
# val3_accs = np.zeros(var_depth)

# for i in range(var_depth):
# 	depth=i+1
# 	print("depth=",depth)
# 	dtree = DecisionTreeClassifier(criterion="entropy", max_features="auto", max_depth=depth)
# 	dtree.fit(Xtran,Ytran)

# 	tran_accs[i] = dtree.score(Xtran,Ytran)
# 	val1_accs[i] = dtree.score(Xval1,Yval1)
# 	val2_accs[i] = dtree.score(Xval2,Yval2)
# 	val3_accs[i] = dtree.score(Xval3,Yval3)


# def plot_graph():
# 	global tran_accs,val1_accs,val2_accs,val3_accs
# 	plt.grid(True)
# 	plt.plot(np.array(range(var_depth)),tran_accs,'-r',label="Tran Acc")
# 	plt.plot(np.array(range(var_depth)),val1_accs,'-b',label="Val1 Acc")
# 	plt.plot(np.array(range(var_depth)),val2_accs,'-g',label="Val2 Acc")
# 	plt.plot(np.array(range(var_depth)),val3_accs,'-y',label="Val3 Acc")
# 	plt.legend(loc='upper left',shadow=True)
# 	plt.show()


# plot_graph()



