from __future__ import print_function
import numpy as np
import pandas as pd
from math import log

trn = pd.read_csv('../data/data_EM/train.data',delimiter=' ')
tst = pd.read_csv('../data/data_EM/test.data',delimiter=' ')


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

for t in [H,B,L,F,X]:
	t.printTable()

print('Log likelihood = ',logLikelihood())