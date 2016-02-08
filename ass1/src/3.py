#parameters
epsilon = 10**(-8)
iterations = 100

import math
import numpy as np
import matplotlib.pyplot as plt


# xs and ys store the data
xs = []
ys = []

with open('../data/q2x.dat') as f:
	xs = [[eval(t) for t in line.split()]+[1] for line in f]

with open('../data/q2y.dat') as f:
	ys = [eval(line) for line in f]


xs = np.array(xs)
ys = np.array(ys)
m = len(xs)
n = len(xs[0])

theta = np.zeros(n)


# returns the inverse of the hessian matrix
def hessian_inv(theta):
	hess = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			hess[i][j] = sum( -(xs[k][i]*xs[k][j]*h_theta(np.array(xs[k]),theta)*(1-h_theta(np.array(xs[k]),theta))) for k in range(m))
	return np.mat(hess).I


# returns the gradient of the function
def gradient(theta):
	return np.mat([[sum( (xs[k][i]*(ys[k] - h_theta(np.array(xs[k]),theta))) for k in range(m))] for i in range(n)])


# returns th0 + x1*th1 + x2*th2 + ...
def h_theta(x,theta):
	return 1/(1 + math.e**(-theta.dot(x)))


# main loop of newton raphson
for i in range(iterations):
	increment = (-1)*np.array((hessian_inv(theta)*gradient(theta)).flatten())[0]

	temp = np.array([(abs(inc)>epsilon) for inc in increment])
	if (not temp.any()):
		break

	theta = theta + increment
	print(i,"\ttheta : ",theta)


def plot_graph():
	plt.grid(True)
	temp0 = [[],[]]
	temp1 = [[],[]]
	plt.xlabel("X2")
	plt.ylabel("X1")
	plt.title("Logistic Regression")
	for i in range(m):
		if (ys[i]==0):
			temp0[0].append(xs[i][0])
			temp0[1].append(xs[i][1])
		elif (ys[i]==1):
			temp1[0].append(xs[i][0])
			temp1[1].append(xs[i][1])
	plt.plot(temp0[0],temp0[1],'ro',label='Label=0')
	plt.plot(temp1[0],temp1[1],'g^',label='Label=1')
	plt.legend()
	x2 = [min([x[0] for x in xs])-0.2 , max([x[0] for x in xs])+0.2]
	x1 = [(-theta[2] - theta[0]*x)/theta[1] for x in x2]
	plt.plot(x2,x1,'b-')
	plt.show()

plot_graph()
