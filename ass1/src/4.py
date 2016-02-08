#parameters
epsilon = 10**(-8)
iterations = 100

import math
import numpy as np
import matplotlib.pyplot as plt


# xs and ys store the data
xs = []
ys = []

with open('../data/q4x.dat') as f:
	xs = [[eval(t) for t in line.split()] for line in f]

with open('../data/q4y.dat') as f:
	fun = lambda x : 1 if x=='Alaska' else 0
	ys = [fun(line.strip()) for line in f]


xs = np.array(xs)
ys = np.array(ys)
m = len(xs)
n = len(xs[0])

theta = np.zeros(n)

# returns mu_x for x=0 and x=1
def mu(x):
	return np.mat(sum((ys[i]==x)*xs[i] for i in range(m))/sum((ys[i]==x)*1 for i in range(m)))


# returns sigma(assuming sigma0 = sigma1) if nothing is passed, else return sigma_x for x=0 and x=1
def sigma(x=2):
	mu0 = mu(0)
	mu1 = mu(1)
	mu_ = lambda x : mu0 if x==0 else mu1
	if (x==0):
		return np.mat(sum((ys[i]==0)*(np.mat(xs[i] - mu0).T*np.mat(xs[i] - mu0)) for i in range(m)))/sum((ys[i]==0)*1 for i in range(m))
	elif(x==1):
		return np.mat(sum((ys[i]==1)*(np.mat(xs[i] - mu1).T*np.mat(xs[i] - mu1)) for i in range(m)))/sum((ys[i]==1)*1 for i in range(m))
	else:
		return np.mat(sum(np.mat(xs[i] - mu_(ys[i])).T*np.mat(xs[i] - mu_(ys[i])) for i in range(m)))/m

mu0 = mu(0)
mu1 = mu(1)
sig  = sigma()
sig0 = sigma(0)
sig1 = sigma(1)


# returns the value of x2 at the boundary for given value of x1
def boundary(x1,mu0,mu1,sig0,sig1):
	sig0_i = sig0.I
	sig1_i = sig1.I
	A = sig0_i - sig1_i
	B = (-2)*(sig0_i*(mu0.T) - sig1_i*(mu1.T))
	C = mu0*sig0_i*(mu0.T) - mu1*sig1_i*(mu1.T)
	a = A[1,1]
	b = B[1] + x1*(A[0,1]+A[1,0])
	c = A[0,0]*x1*x1 + B[0]*x1 + C[0,0]
	if (a==0):
		return (-c/b)
	else:
		return (-b + math.sqrt(b**2 - 4*a*c))/(2*a)


def plot_graph():
	plt.grid(True)
	temp0 = [[],[]]
	temp1 = [[],[]]
	plt.xlabel("X2")
	plt.ylabel("X1")
	plt.title("Gaussian Discrmimant Analysis")
	for i in range(m):
		if (ys[i]==0):
			temp0[0].append(xs[i][0])
			temp0[1].append(xs[i][1])
		elif (ys[i]==1):
			temp1[0].append(xs[i][0])
			temp1[1].append(xs[i][1])
	plt.plot(temp0[0],temp0[1],'g^',label='Canada')
	plt.plot(temp1[0],temp1[1],'ro',label='Alaska')
	x1 = np.linspace(min([x[0] for x in xs]) , max([x[0] for x in xs]),12)
	x2 = [boundary(x,mu0,mu1,sig,sig)[0,0] for x in x1]
	plt.plot(x1,x2,label="Sigma0 == Sigma1")
	x2 = [boundary(x,mu0,mu1,sig0,sig1)[0,0] for x in x1]
	plt.plot(x1,x2,label="Sigma0 != Sigma1")
	plt.legend()
	plt.show()


print("mu0 : ",mu0)
print("mu1 : ",mu1)
print("sigma : ", sig)
print("sigma0 : ", sig0)
print("sigma1 : ", sig1)
plot_graph()