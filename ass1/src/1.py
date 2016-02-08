#parameters
rate = 0.5
epsilon = 10**(-8)
iterations = 5000

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# xs and ys store the data
xs = []
ys = []

with open('../data/q1x.dat') as f:
	xs = [eval(line) for line in f]

with open('../data/q1y.dat') as f:
	ys = [eval(line) for line in f]


# normalize xs values
mean_x = np.mean(xs)
var_x  = np.var(xs)
var_x = math.sqrt(var_x)
xs = [[(x - mean_x)/var_x ,1] for x in xs]


xs = np.array(xs)
ys = np.array(ys)
m = len(xs)
n = len(xs[0])


theta = np.zeros(n)

# theta_vals store the values of theta in course of iterations
theta_vals = [theta]


# returns th0 + x1*th1 + x2*th2 + ...
def h_theta(x,theta):
	return theta.dot(x)

# return error metric J_theta
def j_theta(theta):
	return (sum((ys[i] - h_theta(xs[i],theta))**2 for i in range(m)))/(2*m)

# main loop of gradient descent
for i in range(iterations):
	h_theta_array = [h_theta(x,theta) for x in xs]
	increment = (rate/m) * ((ys - h_theta_array).dot(xs))

	# check for convergence
	temp = np.array([(abs(inc)>epsilon) for inc in increment])
	if (not temp.any()):
		break

	theta = theta + increment
	theta_vals.append(theta)
	print(i,"\ttheta : ",theta,"\tError : ",j_theta(theta))


# plot data points and hypothesis function learned
def plot_graph():
	plt.grid(True)
	plt.xlabel("X Values")
	plt.ylabel("Y Values")
	plt.title("Linear Regression")
	plt.plot([x[0] for x in xs],ys,'ro')
	x = [min([x[0] for x in xs])-0.2 , max([x[0] for x in xs])+0.2]
	y = [h_theta(np.array([t,1]),theta) for t in x]
	plt.plot(x,y,'b-')
	plt.show()

# plots the J_theta function
def plot_error():
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	X = np.arange(min([t_val[0] for t_val in theta_vals])-1 , max([t_val[0] for t_val in theta_vals])+1, 0.2)
	Y = np.arange(min([t_val[1] for t_val in theta_vals])-1 , max([t_val[1] for t_val in theta_vals])+1, 0.2)
	X,Y = np.meshgrid(X, Y)
	Z = np.array([j_theta(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
	ax.plot_surface(X,Y,Z)
	plt.ion()
	plt.show()
	plt.xlabel("X Values")
	plt.ylabel("Y Values")
	plt.title("Error Metric")
	for t_val in theta_vals:
		ax.plot([t_val[0]],[t_val[1]],j_theta(t_val), 'ro')
		plt.draw()
		print("\ttheta : ",theta,"\t\t\tError : ",j_theta(theta))
		plt.pause(0.2)


# plots the contours of error function
def plot_contours():
	X = np.arange(min([t_val[0] for t_val in theta_vals])-1 , max([t_val[0] for t_val in theta_vals])+1, 0.2)
	Y = np.arange(min([t_val[1] for t_val in theta_vals])-1 , max([t_val[1] for t_val in theta_vals])+1, 0.2)
	X,Y = np.meshgrid(X, Y)
	Z = np.array([j_theta(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
	plt.ion()
	plt.show()
	plt.title("Contours")
	plt.xlabel("X Values")
	plt.ylabel("Y Values")
	plt.contour(X,Y,Z,30)
	for t_val in theta_vals:
		plt.plot([t_val[0]],[t_val[1]], 'ro')
		plt.draw()
		plt.pause(0.2)


plot_graph()
plot_error()
plot_contours()