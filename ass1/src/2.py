#parameters
tau = 0.2

import math
import numpy as np
import matplotlib.pyplot as plt

# xs and ys store the data
xs = []
ys = []

with open('../data/q3x.dat') as f:
	xs = [[eval(line),1] for line in f]

with open('../data/q3y.dat') as f:
	ys = [eval(line) for line in f]


xs = np.array(xs)
ys = np.array(ys)
m = len(xs)
n = len(xs[0])


# returns th0 + x1*th1 + x2*th2 + ...
def h_theta(x,theta):
	return theta.dot(x)

# returns the analytical solution for the data
def analytical_unweighted():
	X = np.mat(xs)
	Y = np.mat([[y] for y in ys])
	return np.array((((X.T*X).I)*(X.T*Y)).flatten())[0]


# returns the weighted analytical solution for the data at the query point 'query'
def analytical_weighted(query):
	W = np.mat(np.diag([math.e**((-0.5)*((query - x[0])/tau)**2) for x in xs]))
	X = np.mat(xs)
	Y = np.mat([[y] for y in ys])
	return np.array((((X.T*W*X).I)*(X.T*W*Y)).flatten())[0]


# plots the analytical unweighted solution as well as weighted analytical solution for the query points given
def plot_graph(query_points):
	plt.grid(True)
	plt.plot([x[0] for x in xs],ys,'ro')
	
	plt.xlabel("X Values")
	plt.ylabel("Y Values")
	plt.title("Weighted Linear Regression")

	theta = analytical_unweighted()
	print(theta)
	x = [min([x[0] for x in xs])-1 , max([x[0] for x in xs])+1]
	y = [h_theta(np.array([t,1]),theta) for t in x]
	plt.plot(x,y,'b-')
	
	for q in query_points:
		theta = analytical_weighted(q)
		x = [q-1,q+1]
		y = [h_theta(np.array([t,1]),theta) for t in x]
		plt.plot(x,y)

	plt.show()


# for each point x in this array, we will plot the graph in range (x-1,x+1)
query_points = np.linspace(min([x[0] for x in xs]) , max([x[0] for x in xs]),10)
plot_graph(query_points)
