import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# This function encodes a symbolic categorical attribute (eg: female/male) as a set of numerical one-versus-all features (one-hot-encoding)
def one_hot_encode_categorical(Xtrn,Xval,Xtst):
	lenc = LabelEncoder()
	catvar = Xtrn.columns[Xtrn.dtypes==object]
	oenc = OneHotEncoder(categorical_features=(Xtrn.dtypes==object),sparse=False)

	# Convert from, say, male/female to 0/1 (refer online for more details)
	for var in catvar:
		lenc.fit( pd.concat( [Xtrn[var],Xval[var],Xtst[var]] ) )
		Xtrn[var] = lenc.transform(Xtrn[var])
		Xval[var] = lenc.transform(Xval[var])
		Xtst[var] = lenc.transform(Xtst[var])

	# one-hot-encoding of 0-(k-1) valued k-categorical attribute
	# oenc.fit( pd.concat( [Xtrn,Xval,Xtst] ) )
	# Xtrn = pd.DataFrame(oenc.transform(Xtrn))
	# Xval = pd.DataFrame(oenc.transform(Xval))
	# Xtst = pd.DataFrame(oenc.transform(Xtst))
	return (Xtrn,Xval,Xtst)

# Read training data and partition into features and target label
data = pd.read_csv("train.csv")
Xtrn = data.drop("Survived",1)
Ytrn = data["Survived"]

# Read validation data and partition into features and target label
data = pd.read_csv("validation.csv")
Xval = data.drop("Survived",1)
Yval = data["Survived"]

# Read test data and partition into features and target label
data = pd.read_csv("test.csv")
Xtst = data.drop("Survived",1)
Ytst = data["Survived"]

# convert a symbolic categorical attribute (eg: female/male) to set of numerical one-versus-all features (one-hot-encoding)
Xtrn,Xval,Xtst = one_hot_encode_categorical(Xtrn,Xval,Xtst)


# from sklearn.externals.six import StringIO  
# from sklearn import tree
# import pydot
# dot_data = StringIO()
# tree.export_graphviz(dtree, out_file=dot_data) 
# graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
# graph.write_pdf("dtree.pdf") 

# function score runs prediction on data, and outputs accuracy. 
# If you need predicted labels, use "predict" function

var_depth = 13

train_accs=np.zeros(var_depth)
val_accs=np.zeros(var_depth)
test_accs=np.zeros(var_depth)

# depth=4
# min_samples_split=7


for i in range(var_depth):
	depth = i+1
	dtree = DecisionTreeClassifier(criterion="entropy",max_depth=depth,min_samples_split=7)
	dtree.fit(Xtrn,Ytrn)

	train_accs[i] = dtree.score(Xtrn,Ytrn)
	val_accs[i] = dtree.score(Xval,Yval)
	test_accs[i] = dtree.score(Xtst,Ytst)


print "Train Accs : ", train_accs*100
print "Valid Accs : ", val_accs*100
print "Test Accs  : ", test_accs*100

def plot_graph():
	global train_accs,val_accs,test_accs
	plt.grid(True)
	plt.xlabel("Depth of tree")
	plt.ylabel("Accuracy")
	plt.title("Accuracies of train, validation and test data vs Depth of decision tree")
	plt.plot(np.array(range(var_depth)),train_accs,'-r',label='Train Data Accuracy')
	plt.plot(np.array(range(var_depth)),val_accs,'-b',label='Validation Data Accuracy')
	plt.plot(np.array(range(var_depth)),test_accs,'-g',label='Test Data Accuracy')
	plt.legend(loc='upper left', shadow=True)
	plt.show()

plot_graph()