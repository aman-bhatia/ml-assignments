import math
import random
import numpy as np

data = []
allwords = set([])
groups = set([])

# intialize words
with open('20ng-rec_talk.txt','r') as f:
	for line in f:
		line=line.split()
		groups.add(line[0])
		del line[0]
		allwords.update(set(line))


class article:
	def __init__(self,line):
		self.group=line[0]
		del line[0]
		self.dic={}
		for w in line:
			if w not in self.dic.keys():
				self.dic[w]=0
			self.dic[w]+=1

with open('20ng-rec_talk.txt','r') as f:
	for line in f:
		line=line.split()
		data.append(article(line))


data=np.array(data)
allwords=np.array(list(allwords))
groups=list(groups)

print('Len of data : ',len(data))
print('Len of all words : ',len(allwords))
print('Len of groups : ',len(groups))


def train(data,words):

	fi_y=np.zeros(len(groups))
	fi_j_given_y = {}
	for w in words:
		fi_j_given_y[w] = np.ones(len(groups))	# initialized to one because of laplace smoothing

	for d in data:
		i = groups.index(d.group)
		fi_y[i]+=1
		for w in d.dic.keys():
			fi_j_given_y[w][i] += d.dic[w]

	fi_y = np.log(fi_y/len(data))
	total=np.zeros(len(groups))
	for w in words:
		total += fi_j_given_y[w]
	for w in words:
		fi_j_given_y[w] = np.log(fi_j_given_y[w]/total)

	return fi_y,fi_j_given_y


def predict(art,fi_y,fi_j_given_y,words):
	prob = np.array(fi_y)
	for w in words:
		prob += fi_j_given_y[w]

	return groups[prob.argmax()]


conf_matrix=np.ndarray((len(groups),len(groups)))
conf_matrix = conf_matrix - conf_matrix

def test(data,fi_y,fi_j_given_y,words):
	global conf_matrix
	acc=0
	words=set(words)
	for d in data:
		inter = np.array(list(set(d.dic.keys()).intersection(words)))
		predicted_group = predict(d,fi_y,fi_j_given_y,inter)
		conf_matrix[groups.index(d.group)][groups.index(predicted_group)] += 1
		if (d.group == predicted_group):
			acc+=1
	return acc*100/len(data)


splits = np.array(range(len(data)))
random.shuffle(splits)
assert(1446==len(data)/5)
splits=splits.reshape((5,1446))

iteration=1
for test_split in splits:
	train_split=[]
	for split in splits:
		if (not np.array_equal(split,test_split)):
			train_split=np.append(train_split,split)
	
	train_split=np.asarray(train_split,int)
	test_split=np.asarray(test_split,int)
	
	train_data=data[train_split]
	test_data=data[test_split]
	
	words=set([])
	for d in train_data:
		words.update(set(d.dic.keys()))
	words=np.array(list(words))

	fi_y,fi_j_given_y = train(train_data,words)
	acc = test(test_data,fi_y,fi_j_given_y,words)
	print('\nIter ',iteration,' : Accuracy = ', acc)
	iteration+=1

print('\n\n')
print('=================')
print('Confusion Matrix')
print('=================\n')
for i in range(len(groups)):
	name = str(groups[i])
	while(len(name)<25):
		name+=' '
	print(name,' : ',conf_matrix[i])


diag_index = 0
non_diag_index = (0,1)
for i in range(len(groups)):
	for j in range(len(groups)):
		if (i==j and conf_matrix[i][i] > conf_matrix[diag_index][diag_index]):
			diag_index = i
		elif(i!=j and conf_matrix[i][j] > conf_matrix[non_diag_index[0]][non_diag_index[1]]):
			non_diag_index = (i,j)

print('\n\n')
print("Newsgroup with highest diagonal entry : ",groups[diag_index])
print("Most confused new groups : ",groups[non_diag_index[0]],' and ',groups[non_diag_index[1]])
