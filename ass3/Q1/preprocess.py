import pandas as pd
import numpy as np
import pickle
import math

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
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('validation.csv')
test_data = pd.read_csv('test.csv')

print('Data Loading Completed\n\n')

def preProcess(data):
	for att in disc_atts:
		for i in range(len(poss_vals[att])):
			data[att].loc[data[att]==poss_vals[att][i]] = i

	for att in num_atts:
		med = np.median(data[att])
		data[att].loc[data[att] <= med] = 0
		data[att].loc[data[att] > med] = 1


print('Preprocessing Data...')
preProcess(train_data)
preProcess(val_data)
preProcess(test_data)
print('Preprocessing Completed')


pkl_handle = open('data.pkl','wb')
pickle.dump((train_data,val_data,test_data),pkl_handle)
pkl_handle.close()
print("Data saved in file data.pkl")