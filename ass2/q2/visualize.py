import sys
import numpy as np
import scipy.io as sio
from matplotlib import cm as cm
from matplotlib import pyplot as plt

data = sio.loadmat('mnist_all.mat')

try:
	img_data = data[sys.argv[1]][int(sys.argv[2])] 
except:
	print('\nUsage : python3 visualize.py <data_class(train0/train1/.../test9)> <example_index>\n')
	exit(0)

img_data = img_data.reshape(28,28)
plt.imshow(img_data,cmap=cm.Greys_r)
plt.show()