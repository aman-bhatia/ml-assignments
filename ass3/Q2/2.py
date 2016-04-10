import math
import random
import numpy as np

data = []
allwords = set([])
groups = set([])

# intialize words
with open('20ng-rec_talk.txt','rb') as f:
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

with open('20ng-rec_talk.txt','rb') as f:
	for line in f:
		line=line.split()
		data.append(article(line))


data=np.array(data)
allwords=np.array(list(allwords))
groups=list(groups)

print('Len of data : ',len(data))
print('Len of all words : ',len(allwords))
print('Len of groups : ',len(groups))


def predict():
	return groups[random.randint(0,7)]


splits = np.array(range(len(data)))
random.shuffle(splits)
assert(1446==len(data)/5)
splits=splits.reshape((5,1446))

iteration=1
for test_split in splits:
	
	test_data=data[test_split]
	
	acc=0
	for d in test_data:
		if (d.group == predict()):
			acc+=1
	# print("Prediction Completed!")
	
	print('\nIter ',iteration,' : Accuracy = ', acc*100/len(test_data))
	iteration+=1














