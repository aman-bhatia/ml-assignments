import sys
import numpy as np
import scipy.io as sio
from matplotlib import cm as cm
from matplotlib import pyplot as plt

with open('../data/kmeans_data/digitdata.txt') as f:
	indices = f.readline().split()
	def fun(x):
		return int(x[6:-1])
	fun_ = np.vectorize(fun)
	indices = fun_(indices)

	data = []
	for line in f:
		data.append(np.asarray(line.split()[1:] , int))

data = np.array(data)

try:
	img_data_compressed = data[int(sys.argv[1])] 
except:
	print('\nUsage : python3 visualize.py <example_index>\n')
	exit(0)


img_data = np.zeros(784)
for i in range(len(indices)):
	img_data[indices[i]] = img_data_compressed[i]

img_data = img_data.reshape(28,28)
plt.imshow(img_data,cmap=cm.Greys_r)
plt.show()