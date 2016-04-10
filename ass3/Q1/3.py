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

print('Loading Data...')
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('validation.csv')
test_data = pd.read_csv('test.csv')

print('Data Loading Completed\n\n')

def preProcess(data):
	for att in disc_atts:
		for i in range(len(poss_vals[att])):
			data[att].loc[data[att]==poss_vals[att][i]] = i

print('Preprocessing Data...')
preProcess(train_data)
preProcess(val_data)
preProcess(test_data)
print('Preprocessing Completed!\n')

def entropy(data):
	y = data['Survived']
	p0 = (1+len(y[y==0])) / float(2+len(y))
	p1 = 1-p0
	return (-1)*(p0*math.log(p0,2) + p1*math.log(p1,2))


def entropy_given_att(data,att):
	splited_data = split(data,att)
	return sum(len(d)*entropy(d) for d in splited_data)/len(data)


def split(data,att):
	ret = []
	if (att in num_atts):
		med = np.median(data[att])
		d0 = data[data[att] <= med]
		d1 = data[data[att] > med]
		ret.append(d0)
		ret.append(d1)	
	else:
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
	for att in num_atts:
		if (att in data.columns):
			med = np.median(data[att])
			if (len(data[data[att]<=med]) == len(data)):
				data=data.drop(att,1)
	best_att = bestFeatureToSplit(data)
	splited_data = split(data,best_att)
	if (best_att in num_atts):
		return np.array([best_att, maj_class ,np.array([growTree(d) for d in splited_data]),np.median(data[best_att])])
	else:
		return np.array([best_att, maj_class ,np.array([growTree(d) for d in splited_data])])

def predict(tree,data_val,depth):
	while(tree[0]!=None and depth!=0):
		depth-=1
		if (tree[0] in num_atts):
			med = tree[3]
			tree = tree[2][not (data_val[tree[0]] <= med)]
		else:
			tree = tree[2][int(data_val[tree[0]])]
	return tree[1]


def accuracy(tree,data,depth):
	acc = 0
	for i,r in data.iterrows():
		acc += (r['Survived']==predict(tree,r,depth))
	return (float(acc)/len(data))*100


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


# import pydot

# g = pydot.Dot(graph_type='graph')

# def makeGraph(node,i):
# 	if (node[0]==None):
# 		return
# 	j = i+1
# 	for child in node[2]:
# 		edge = pydot.Edge(str(node[0]) + '_' + str(i),str(child[0]) + '_' + str(j))
# 		g.add_edge(edge)
# 		j+=1
# 	j=i+1
# 	for child in node[2]:
# 		makeGraph(child,j)
# 		j+=1


# makeGraph(tree,1)
# g.write_png("tree_part3.png")

# max split on age(~ 100 times)


# tree depth on train data = 15
tree_depth = 15

num_node_array=np.zeros(tree_depth)
train_accs=np.zeros(tree_depth)
val_accs=np.zeros(tree_depth)
test_accs=np.zeros(tree_depth)

print('Predicting...')
for i in range(tree_depth):
	num_node_array[i] = numNodes(tree,i)
	train_accs[i] = accuracy(tree,train_data,i)
	val_accs[i] = accuracy(tree,val_data,i)
	test_accs[i] = accuracy(tree,test_data,i)
print('Prediction Completed!\n')


def plot_graph():
	global num_node_array,train_accs,val_accs,test_accs
	plt.grid(True)
	plt.xlabel("Number of nodes in tree")
	plt.ylabel("Accuracy")
	plt.title("Accuracies of train, validation and test data vs Number of nodes in decision tree")
	plt.plot(num_node_array,train_accs,'-r',label='Train Data Accuracy')
	plt.plot(num_node_array,val_accs,'-b',label='Validation Data Accuracy')
	plt.plot(num_node_array,test_accs,'-g',label='Test Data Accuracy')
	plt.legend(loc='lower right', shadow=True)
	plt.show()


print('============')
print('|Accuracies|')
print('============\n')
print("Train data      : ", train_accs[-1])
print("Validation data : ", val_accs[-1])
print("Test data       : ", test_accs[-1])
print('\n\n')

plot_graph()
