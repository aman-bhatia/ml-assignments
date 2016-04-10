import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

disc_atts = ['Pclass','Sex','Embarked','Cabin_a']
num_atts =  ['Age','SibSp','Parch','Ticket','Fare','Cabin_b']

poss_vals = {}
poss_vals['Pclass'] = [0,1,2,3]
poss_vals['Sex'] = ['0','male','female']
poss_vals['Embarked'] = ['0','C','Q','S']
poss_vals['Cabin_a'] = ['0','A','B','C','D','E','F','G','T']
poss_vals['Age'] = [0,1]
poss_vals['SibSp'] = [0,1]
poss_vals['Parch'] = [0,1]
poss_vals['Ticket'] = [0,1]
poss_vals['Fare'] = [0,1]
poss_vals['Cabin_b'] = [0,1]

print('Loading Data...')

pkl_handle = open('data.pkl','rb')
train_data, val_data , test_data = pickle.load(pkl_handle)
pkl_handle.close()

print('Data Loading Completed\n\n')

def entropy(data):
	y = data['Survived']
	p0 = (1+len(y[y==0])) / (2+len(y))
	p1 = 1-p0
	return (-1)*(p0*math.log(p0,2) + p1*math.log(p1,2))


def entropy_given_att(data,att):
	splited_data = split(data,att)
	return sum(len(d)*entropy(d) for d in splited_data)/len(data)


def split(data,att):
	ret = []
	for i in range(len(poss_vals[att])):
		d = data[data[att] == i]
		d = d.drop(att,1)
		ret.append(d)
	return ret


def bestFeatureToSplit(data):
	cur_entropy = entropy(data)
	min_entropy = 2
	best_att = None
	for att in data.columns:
		if (att == 'Survived'):
			continue
		new_entropy = entropy_given_att(data,att)
		if (min_entropy >= new_entropy):
			if (min_entropy == new_entropy):
				new_index = np.where(np.array(data.columns)==att)[0][0]
				min_index = np.where(np.array(data.columns)==best_att)[0][0]
				if (new_index < min_index):
					min_entropy = new_entropy
					best_att = att
			else:
				min_entropy = new_entropy
				best_att = att
	return best_att


def majorityClass(data):
	y = data['Survived']
	return (len(y[y==0]) < len(y[y==1]))


def growTree(data):
	if (len(data)==0):
		return np.array([None,0,np.array([])])
	maj_class = majorityClass(data)
	if (len(data.columns)==1):
		return np.array([None,maj_class,np.array([])])
	y = data['Survived']
	if (len(y[y==y.iloc[0]]) == len(y)):
		return np.array([None,y.iloc[0],np.array([])])
	best_att = bestFeatureToSplit(data)
	splited_data = split(data,best_att)
	return np.array([best_att, maj_class ,np.array([growTree(d) for d in splited_data])])


def predict(tree,data_val,depth):
	while(tree[0]!=None and depth!=0):
		depth-=1
		tree = tree[2][int(data_val[tree[0]])]
	return tree[1]


def accuracy(tree,data,depth):
	acc = 0
	for i,r in data.iterrows():
		acc += (r['Survived']==predict(tree,r,depth))
	return (acc/len(data))*100


def numNodes(tree,depth):
	if (tree[0]==None or depth==0):
		return 1
	num_nodes=1
	depth-=1
	for i in range(len(tree[2])):
		num_nodes += numNodes(tree[2][i],depth)
	return num_nodes


print("Making Decision Tree")
tree = growTree(train_data)
print("Making Decision Tree Completed!\n")

print('Predicting...')
trainAcc = accuracy(tree,train_data,11)
valAcc = accuracy(tree,val_data,11)
testAcc = accuracy(tree,test_data,11)
print('Prediction Completed!\n')


print('===========================')
print('|Accuracies Before Pruning|')
print('===========================\n')
print("Train data      : ", trainAcc)
print("Validation data : ", valAcc)
print("Test data       : ", testAcc)
print('\n\n')

num_node_array=[numNodes(tree,100)]
train_accs=[trainAcc]
val_accs=[valAcc]
test_accs=[testAcc]


def prune(node):
	global tree, valAcc, num_node_array, train_accs, val_accs,test_accs
	if (node[0]==None):
		return False
	for child in node[2]:
		if (prune(child)):
			num_node_array.append(numNodes(tree,100))
			train_accs.append(accuracy(tree,train_data,100))
			val_accs.append(accuracy(tree,val_data,100))
			test_accs.append(accuracy(tree,test_data,100))

	nodecopy0 = node[0]
	node[0] = None
	newAcc = accuracy(tree,val_data,100)
	if (newAcc >= valAcc):
		valAcc = newAcc
		node[2]=np.array([])
		return True
	else:
		node[0] = nodecopy0
		return False


print('Pruning...')
prune(tree)
print('Pruning Completed')

num_node_array=np.array(num_node_array)
train_accs = np.array(train_accs)
val_accs = np.array(val_accs)
test_accs = np.array(test_accs)

print('==========================')
print('|Accuracies After Pruning|')
print('==========================\n')
print("Train data      : ", train_accs[-1])
print("Validation data : ", val_accs[-1])
print("Test data       : ", test_accs[-1])

print('\n\n')

def plot_graph():
	global num_node_array,train_accs,val_accs,test_accs
	plt.grid(True)
	plt.xlabel("Number of nodes in tree")
	plt.ylabel("Accuracy")
	plt.title("Accuracies of train, validation and test data vs Number of nodes in decision tree")
	plt.plot(num_node_array,train_accs,'-r',label='Train Data Accuracy')
	plt.plot(num_node_array,val_accs,'-b',label='Validation Data Accuracy')
	plt.plot(num_node_array,test_accs,'-g',label='Test Data Accuracy')
	plt.legend(loc='center right', shadow=True)
	plt.show()


plot_graph()
