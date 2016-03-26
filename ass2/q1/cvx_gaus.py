# parameters
C = 1
threshold = 10**(-4)
gamma = 2.5 * (10**(-4))


import sys
import cvxopt
import numpy as np
import pickle

print('Loading Train Data...')
data = []
with open('train.data') as f:
	data = [line.split(',') for line in f]

m = len(data)
n = len(data[0])-1
[X_temp,Y_temp] = np.split(np.array(data),[n],1)

X = np.ndarray.astype(X_temp,float)
Y = np.array([1 if y[0]=='ad.\n' else -1 for y in Y_temp])

lin_kernel = lambda x,y : np.dot(x,y)
gaus_kernel = lambda x,y : np.exp( (- np.linalg.norm(x-y)**2)*gamma)

print('Starting...')


# K = np.zeros((m,m))
# for i in range(m):
# 	for j in range(m):
# 		K[i,j] = gaus_kernel(X[i], X[j])

# pkl_handle = open('K_gaus.pkl','wb')
# pickle.dump(K,pkl_handle)
# pkl_handle.close()

pkl_handle = open('K_gaus.pkl','rb')
K = pickle.load(pkl_handle)
pkl_handle.close()

'''
print('Making matrices...')
P = 1.0*cvxopt.matrix(np.outer(Y,Y) * K)
q = 1.0*cvxopt.matrix(np.ones(m) * -1)

A = 1.0*cvxopt.matrix(Y, (1,m))
b = cvxopt.matrix(0.0)

G1 = -1.0*np.diag(np.ones(m))
G2 = 1.0*np.diag(np.ones(m))
G = cvxopt.matrix(np.r_[G1, G2])

h1 = np.zeros(m)
h2 = np.ones(m) * C
h = cvxopt.matrix(1.0*np.r_[h1,h2])

# Solve QP problem
print('Solving...')
solution = cvxopt.solvers.qp(P, q, G, h, A, b)

sol_handle = open('sol_gaus.pkl','wb')
pickle.dump(solution,sol_handle)
sol_handle.close()
'''
sol_handle = open('sol_gaus.pkl','rb')
solution = pickle.load(sol_handle)
sol_handle.close()


# alpha's
alphas = np.array(solution['x'])

# support vectors
sv_temp = alphas>threshold
sv = alphas[alphas>threshold]
print("Number of Support Vectors : ",len(sv))

# save support vectors
sv_file = open('support_vector_cvx_gaus.txt','wb')
temp = ''
for i in range(m):
	if (sv_temp[i]):
		temp += (str(X[i])+'\n')
sv_file.write(temp)
sv_file.close()

# intercept
b1 = [sum(alphas[j] * Y[j] * K[j,i] for j in range(m)) if (Y[i]==1 and sv_temp[i]) else sys.maxint for i in range(m)]
b2 = [sum(alphas[j] * Y[j] * K[j,i] for j in range(m)) if (Y[i]==-1 and sv_temp[i]) else -sys.maxint for i in range(m)]
b = (-0.5) * (min(b1) + max(b2))
b = b[0]
print('Intercept : ', b)



print('Loading Test Data...')
testdata = []
with open('test.data') as f:
	testdata = [line.split(',') for line in f]

m_test = len(testdata)
n_test = len(testdata[0])-1
[X_test_temp,Y_test_temp] = np.split(np.array(testdata),[n],1)

X_test = np.ndarray.astype(X_test_temp,float)
Y_test = np.array([1 if y[0]=='ad.\n' else -1 for y in Y_test_temp])

acc=0
for i in range(m_test):
	res = sum(alphas[j] * Y[j] * gaus_kernel(X[j],X_test[i]) for j in range(m)) + b
	if (res < 0 and Y_test[i] == -1):
		acc += 1
	elif (res >= 0 and Y_test[i] == 1):
		acc += 1

print('Accuracy : ', acc*100.0/m_test)