import sys
import math
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# load data file
print('Loading Data...')
data = sio.loadmat('mnist_all.mat')
trainset = []
for i in range(10):
	label = np.zeros(10)
	label[i] = 1
	traini = list(zip(data['train'+str(i)], [label]*len(data['train'+str(i)])))
	trainset += traini

testset = []
for i in range(10):
	label = np.zeros(10)
	label[i] = 1
	testi = list(zip(data['test'+str(i)], [label]*len(data['test'+str(i)])))
	testset += testi

trainset = np.array(trainset)
testset = np.array(testset)

print('Dataset Loaded...')


# given an array, returns sigmoid applied to each element of the array
def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))

# define data structure for layers
class Layer :
	def __init__(self, name, num_units, num_inputs_to_each_unit):
		self.name = name
		self.num_units = num_units
		self.num_inputs = num_inputs_to_each_unit
		self.weights = np.random.random_sample((num_units,num_inputs_to_each_unit))*0.4 - 0.2

	def calc_output(self,input_data):
		# if layer is input layer then output is just input
		if (self.name == '0'):
			self.output = np.append(input_data/255.0,[1])
		else:
			assert (len(input_data) == self.num_inputs)
			net_output = np.inner(self.weights, input_data)
			self.output = sigmoid(net_output)
		return self.output


# define data structure for neural network
class Nnet :
	def __init__(self,num_units_in_layers):
		self.num_layers = len(num_units_in_layers)
		self.layers = []

		num_units_in_layers[0] += 1
		for i in range(self.num_layers):
			if (i==0):
				self.layers.append(Layer(str(i),num_units_in_layers[i],1))
			else:
				self.layers.append(Layer(str(i),num_units_in_layers[i],num_units_in_layers[i-1]))

	def feed_forward(self,input_data):
		for i in range(self.num_layers):
			input_data = self.layers[i].calc_output(input_data)


	def back_propagate(self,labels,rate):
		i = self.num_layers - 1
		delta_l = []	# delta of the layer
		while(i>0):
			if (i == self.num_layers-1):
				delta_l = (-1)*(labels - self.layers[i].output) * (self.layers[i].output) * (1 - self.layers[i].output)
				increment = rate * np.outer(delta_l , self.layers[i-1].output)
			else:
				delta_l = np.dot(delta_l , self.layers[i+1].weights) * (self.layers[i].output) * (1 - self.layers[i].output)
				increment = rate * np.outer(delta_l , self.layers[i-1].output)

			self.layers[i].weights = self.layers[i].weights - increment
			i-=1
		
		error = sum(0.5*(t**2) for t in (labels - self.layers[self.num_layers-1].output))
		return error

	def test(self,test_data):
		acc = 0
		for te_data in test_data:
				self.feed_forward(te_data[0])
				output = self.layers[self.num_layers-1].output
				if (output.argmax() == te_data[1].argmax()):
					acc += 1
		print ('\tAccuracy : ',acc,'/',len(test_data), '[',acc*100.0/len(test_data),']')


	def train(self,train_data,test_data,iterations):
		print('Training data...')
		order = np.array(range(len(train_data)))
		prev_error = 0
		for i in range(iterations):
			random.shuffle(order)
			print('Iteration : ' , i,' of ',iterations)
			rate = 1/math.sqrt(i+1)
			error = 0
			for j in order:
				self.feed_forward(train_data[j][0])
				error += self.back_propagate(train_data[j][1],rate)
			print('\tError : ' , error)
			if (abs(prev_error - error) < 1):
				return
			prev_error = error
			self.test(test_data)



# init nnet
nnet = Nnet([784,100,10])
nnet.train(trainset,testset,30)
print("Trained Succesfully...")
print('Testing')
nnet.test(testset)