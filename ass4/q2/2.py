from __future__ import print_function
import numpy as np
import pandas as pd
from math import log
import copy

trn = pd.read_csv('../data/data_EM/train-m1.data',delimiter=' ')
tst = pd.read_csv('../data/data_EM/test.data',delimiter=' ')


for att in trn.columns:
	trn[att][trn[att] == '?'] = 2
	trn[att] = trn[att].astype(int)


class Table():
	def __init__(self,name,parents):
		self.name=name
		self.num_parents = len(parents)
		self.parents = parents
		self.tab = np.zeros((2**self.num_parents,2))

	def norm(self):
		for i in range(len(self.tab)):
			self.tab[i] /= sum(self.tab[i])

	def printTable(self):
		print('==============================')
		print('Table ',self.name)
		print(self.tab)
		print('==============================\n\n')


def initTables():
	H = Table('H',[])
	for i in [0,1]:
		H.tab[0][i] = len(trn[trn.H==i])

	B = Table('B',[H])
	for i in [0,1]:
		for j in [0,1]:
			B.tab[i][j] = len(trn[trn.H==i][trn.B==j])

	L = Table('L',[H])
	for i in [0,1]:
		for j in [0,1]:
			L.tab[i][j] = len(trn[trn.H==i][trn.L==j])

	F = Table('F',[B,L])
	for i in [0,1,2,3]:
		for j in [0,1]:
			F.tab[i][j] = len(trn[trn.B==(i//2)][trn.L==(i%2)][trn.F==j])		
				
	X = Table('X',[L])
	for i in [0,1]:
		for j in [0,1]:
			X.tab[i][j] = len(trn[trn.L==i][trn.X==j])

	ret = (H,B,L,F,X)
	for t in ret:
		t.norm()
	return ret

H,B,L,F,X = initTables()

def logLikelihood():
	ret = 0 
	for idx,d in tst.iterrows():
		ret += log(H.tab[0][d.H]) + log(B.tab[d.H][d.B]) + log(L.tab[d.H][d.L]) + log(F.tab[2*d.B + d.L][d.F]) + log(X.tab[d.L][d.X])
	return ret



print("Initial tables :-")
for t in [H,B,L,F,X]:
	t.printTable()
# print('Initial Log likelihood = ',logLikelihood(H,B,L,F,X))


def getWeight(d):
	return (H.tab[0][d[0]])*(B.tab[d[0]][d[1]])*(L.tab[d[0]][d[2]])*(F.tab[2*d[1] + d[2]][d[4]])*(X.tab[d[2]][d[3]])

def initNewTables():
	H = Table('H',[])
	for i in [0,1]:
		H.tab[0][i] = sum(trn[trn.H==i]['weights'])

	B = Table('B',[H])
	for i in [0,1]:
		for j in [0,1]:
			B.tab[i][j] = sum(trn[trn.H==i][trn.B==j]['weights'])

	L = Table('L',[H])
	for i in [0,1]:
		for j in [0,1]:
			L.tab[i][j] = sum(trn[trn.H==i][trn.L==j]['weights'])

	F = Table('F',[B,L])
	for i in [0,1,2,3]:
		for j in [0,1]:
			F.tab[i][j] = sum(trn[trn.B==(i//2)][trn.L==(i%2)][trn.F==j]['weights'])		
				
	X = Table('X',[L])
	for i in [0,1]:
		for j in [0,1]:
			X.tab[i][j] = sum(trn[trn.L==i][trn.X==j]['weights'])

	ret = (H,B,L,F,X)
	for t in ret:
		t.norm()
	return ret


data = trn.values
num_ex = len(data)
data = np.c_[data,np.zeros(len(data))]


converge = False
prev_error = 0
while not converge:
	# intialize new data i.e. make new data with 2 replaced as 0 and 1
	new_data = []
	for i in range(num_ex):
		indexOf2 = np.where(data[i]==2)[0][0]
		
		data[i][indexOf2] = 0
		w0 = getWeight(data[i])
		new_data.append(copy.copy(data[i]))
		
		data[i][indexOf2] = 1
		w1 = getWeight(data[i])
		new_data.append(copy.copy(data[i]))
		
		data[i][indexOf2] = 2
		new_data[-2][-1] = w0/(w0+w1)
		new_data[-1][-1] = w1/(w0+w1)
		
	trn = pd.DataFrame(new_data)
	trn.columns = ['H', 'B', 'L', 'X', 'F','weights']
	
	Hn,Bn,Ln,Fn,Xn = initNewTables()

	def getError():
		ret = 0
		ret += np.sum(abs(Hn.tab - H.tab))
		ret += np.sum(abs(Bn.tab - B.tab))
		ret += np.sum(abs(Ln.tab - L.tab))
		ret += np.sum(abs(Fn.tab - F.tab))
		ret += np.sum(abs(Xn.tab - X.tab))
		return ret

	error = getError()
	H,B,L,F,N = Hn,Bn,Ln,Fn,Xn
		
	print("Error : ",error)
	# print("New log likelihood = ",logLikelihood())
	if (abs(error-prev_error) < 0.00001):
		break
	prev_error = error



print("New Tables :-")
for t in [Hn,Bn,Ln,Fn,Xn]:
	t.printTable()
print("New log likelihood = ",logLikelihood())
