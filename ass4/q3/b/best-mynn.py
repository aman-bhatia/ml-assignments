from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import pandas as pd

import lasagne


def load_dataset():

	data = pd.read_csv('./data/train.csv', header=None)
	X_train = np.ascontiguousarray(data.drop(0,1).values)
	y_train = np.ascontiguousarray(data[0].values)

	data = pd.read_csv('./data/validation_1.csv', header=None)
	X_train = np.r_[X_train,np.ascontiguousarray(data.drop(0,1).values)]
	y_train = np.r_[y_train,np.ascontiguousarray(data[0].values)]

	data = pd.read_csv('./data/validation_2.csv', header=None)
	X_train = np.r_[X_train,np.ascontiguousarray(data.drop(0,1).values)]
	y_train = np.r_[y_train,np.ascontiguousarray(data[0].values)]

	data = pd.read_csv('./data/validation_3.csv', header=None)
	X_train = np.r_[X_train,np.ascontiguousarray(data.drop(0,1).values)]
	y_train = np.r_[y_train,np.ascontiguousarray(data[0].values)]

	data = pd.read_csv('./data/validation_3.csv', header=None)
	X_val = np.ascontiguousarray(data.drop(0,1).values)
	y_val = np.ascontiguousarray(data[0].values)

	data = pd.read_csv('./data/testfile.csv', header=None)
	X_test = np.ascontiguousarray(data.drop(0,1).values)

	# cast these arrays to int32 for theano compatibility
	y_train = y_train.astype(dtype=np.int32,copy=False)
	y_val = y_val.astype(dtype=np.int32,copy=False)

	return X_train, y_train, X_val, y_val, X_test



def build_mlp(input_var=None):

	# Input layer, specifying the expected input shape of the network
	# (unspecified batchsize, 146 values) and
	# linking it to the given Theano variable `input_var`, if any:
	l_in = lasagne.layers.InputLayer(shape=(None, 146),
									 input_var=input_var)

	# Apply 20% dropout to the input data:
	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

	# Add a fully-connected layer of 200 units, using the linear rectifier, and
	# initializing weights with Glorot's scheme (which is the default anyway):
	l_hid1 = lasagne.layers.DenseLayer(
			l_in_drop, num_units=200,
			nonlinearity=lasagne.nonlinearities.leaky_rectify)
			# W=lasagne.init.Normal(0.1,0.0))

	# We'll now add dropout of 20%:
	l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.2)

	# Another 200-unit layer:
	l_hid2 = lasagne.layers.DenseLayer(
			l_hid1_drop, num_units=200,
			nonlinearity=lasagne.nonlinearities.leaky_rectify)

	# 20% dropout again:
	l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.2)

	# Finally, we'll add the fully-connected output layer, of 2 softmax units:
	l_out = lasagne.layers.DenseLayer(
			l_hid2_drop, num_units=2,
			nonlinearity=lasagne.nonlinearities.softmax)

	# Each layer is linked to its incoming layer(s), so we only need to pass
	# the output layer to give access to a network in Lasagne:
	return l_out


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]



def main(num_epochs=150):
	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test = load_dataset()

	#normalize data
	for i in range(len(X_train[0])):
		X_train[:,i] = (X_train[:,i] - X_train[:,i].mean()) / X_train[:,i].std()
		X_val[:,i] = (X_val[:,i] - X_val[:,i].mean()) / X_val[:,i].std()
		X_test[:,i] = (X_test[:,i] - X_test[:,i].mean()) / X_test[:,i].std()

	# Prepare Theano variables for inputs and targets
	input_var = T.matrix('inputs')
	target_var = T.ivector('targets')

	# Create neural network model
	print("Building model and compiling functions...")
	network = build_mlp(input_var)


	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	# We could add some weight decay as well here, see lasagne.regularization.

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate=0.01,momentum=0.9)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
					  dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	#predict on test data function
	test_fn = theano.function([input_var],test_prediction)

	eighty77 = False
	eighty78 = False
	eighty79 = False
	eighty = False
	eighty1 = False
	eighty2 = False
	eighty3 = False
	eighty4 = False
	eighty5 = False
	eighty6 = False
	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, y_train, 200, shuffle=True):
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 200, shuffle=False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		if ((not eighty77 ) and (val_acc / val_batches * 100) > 77):
			eighty77 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-77.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty78 ) and (val_acc / val_batches * 100) > 78):
			eighty78 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-78.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty79 ) and (val_acc / val_batches * 100) > 79):
			eighty79 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-79.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty ) and (val_acc / val_batches * 100) > 80):
			eighty = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-80.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty1 ) and (val_acc / val_batches * 100) > 81):
			eighty1 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-81.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty2 ) and (val_acc / val_batches * 100) > 82):
			eighty2 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-82.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty3 ) and (val_acc / val_batches * 100) > 83):
			eighty3 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-83.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty4 ) and (val_acc / val_batches * 100) > 84):
			eighty4 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-84.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty5 ) and (val_acc / val_batches * 100) > 85):
			eighty5 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-85.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		if ((not eighty6 ) and (val_acc / val_batches * 100) > 86):
			eighty6 = True
			predicted_values = test_fn(X_test)
			f = open('prediction-nn-86.csv','w')
			f.write('ID,TARGET\n')
			for i in range(len(predicted_values)):
				f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(
			val_acc / val_batches * 100))


	# predicted_values = test_fn(X_test)

	# f = open('prediction-nn.csv','w')
	# f.write('ID,TARGET\n')
	# for i in range(len(predicted_values)):
	# 	f.write(str(i)+','+str(np.argmax(predicted_values[i]))+'\n')

	# Optionally, you could now dump the network weights to a file like this:
	# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
	
	# And load them again later on like this:
	# with np.load('model.npz') as f:
	#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	# lasagne.layers.set_all_param_values(network, param_values)


main()