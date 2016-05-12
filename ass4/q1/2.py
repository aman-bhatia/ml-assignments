import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode


k=4
iterations = 30

print("Loading Data...")
with open('../data/kmeans_data/digitdata.txt') as f:
	indices = f.readline().split()
	def fun(x):
		return int(x[6:-1])
	fun_ = np.vectorize(fun)
	indices = fun_(indices)

	data = []
	for line in f:
		data.append(np.asarray(line.split()[1:] , int))

with open('../data/kmeans_data/digitlabels.txt') as f:
	f.readline()
	labels = []
	for line in f:
		labels.append(int(line.split()[-1]))


data = np.array(data)
labels = np.array(labels)
m = len(data)

print("Data Loaded Successfully!")

mu=[]
for i in range(k):
	mu.append(data[random.randint(0,m-1)])

mu = np.array(mu)
c = np.zeros(m)
goodness_array = []
error_array = []

print("Running k-means...")
for it in range(iterations):
	print("Iter : ",it)

	# assign clusters
	for i in range(m):
		c[i] = np.argmin([np.linalg.norm(data[i] - mu[j]) for j in range(k)])
	
	prev_mu = np.copy(mu)

	# compute goodness
	goodness_array.append(sum(np.linalg.norm(data[i] - mu[c[i]]) for i in range(k)))

	#compute error
	predicted_labels = np.zeros(m)
	for i in range(k):
		temp = []
		for j in range(m):
			if (c[j]==i):
				temp.append(labels[j])
		new_label = mode(temp)[0][0]
		for j in range(m):
			if (c[j]==i):
				predicted_labels[j] = new_label
		

	error_array.append(np.sum(predicted_labels!=labels)/m)

	# modify centroids
	for j in range(k):
		mu[j] = sum((c[i]==j)*data[i] for i in range(m))/sum((c[i]==j) for i in range(m))
	
	if ((abs(prev_mu-mu)<2).all()):
		break

print("k-means completed")

def plot_graph(x,y,x_label,y_label,title):
	plt.grid(True)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.plot(x,y,'-r')
	plt.show()


plot_graph(np.array(range(len(goodness_array))),goodness_array, "Number of iterations", "Goodness" , "Number of iterations vs Goodness")
plot_graph(np.array(range(len(error_array))),error_array, "Number of iterations", "Misclassified examples/Total examples" , "Number of iterations vs error")

