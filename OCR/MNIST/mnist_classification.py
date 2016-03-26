#!/usr/bin/python
#Use the Virtual env inside ML

import numpy as np
from theano import *
import theano.tensor as T
import math
from LoadDataset import LoadDataset

#Storing the dataset in shared theano variable
def create_shared(data_xy, borrow=True):
	data_x, data_y = data_xy
	shared_x = theano.shared(numpy.asarray(data_x,
				dtype=theano.config.floatX),
				borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y,
				dtype=theano.config.floatX),
				borrow=borrow)
	return shared_x, shared_y

def encode_labels(labels, max_index):
    """Encode the labels into binary vectors."""
    encoded = np.zeros((labels.shape[0], max_index + 1))
    
    # Fill in the ones at the right indices.
    for i in xrange(labels.shape[0]):
        encoded[i, labels[i]] = 1
    return encoded


def accuracy(predicted, actual):
    total = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        total += 1
        if p == a:
            correct += 1
    print correct/total
    return correct / total



if __name__ == '__main__':

	load = LoadDataset()
	load_x, load_y = [], []
	for sets in load.return_dataset():
		load_x.append(create_shared(sets)[0])
		load_y.append(create_shared(sets)[1])

	weight3_shape = (10, 40)
	bias3_shape = 10

	weight1_shape = (40, 784)
	bias1_shape = 40

	weight2_shape = (40, 40)
	bias2_shape = 40

	#Creating Shared variables for weight and bias
	w2 = shared(np.random.random(weight2_shape) - 0.5, name="w2")
	b2 = shared(np.random.random(bias2_shape) - 0.5, name="b2")
	w3 = shared(np.random.random(weight3_shape) - 0.5, name="w3")
	b3 = shared(np.random.random(bias3_shape) - 0.5, name="b3")
	w1 = shared(np.random.random(weight1_shape) - 0.5, name="w1")
	b1 = shared(np.random.random(bias1_shape) - 0.5, name="b1")

	inp = T.dmatrix("inp")
	labels = T.dmatrix("labels")

	#The hidden layer computation
	hidden1 = T.nnet.sigmoid(inp.dot(w1.transpose()) + b1)
	hidden2 = T.nnet.sigmoid(hidden1.dot(w2.transpose()) + b2)

	#The softmax computation
	output = T.nnet.softmax(hidden2.dot(w3.transpose()) + b3)
	predict = T.argmax(output, axis=1)
	
	#The cost function to be minimized
	cost = T.nnet.binary_crossentropy(output, labels).mean()
	print"doesn't pass"
	predict_function = function([inp], predict)
	print "passes"

	alpha = T.dscalar('alpha')
	#weights = [hidden_w_shared, w_shared, hidden_b_shared, b_shared]
	weights = [w1, w2, w3, b1, b2, b3]
	updates = [(w, w - alpha * grad(cost, w)) for w in weights]
	train = function([inp, labels, alpha], cost, updates=updates)

	temp_alpha = 10.0
	old_cost = 1000
	current_cost = 1
	# The modulus of difference between costs should be between 0.005
	while True:
		for j in range(load_x[0]):
			old_cost = current_cost
			current_cost = train(load_x[0][j], load_y[0][j], temp_alpha)

		print current_cost
		print "Cost diff.." + current_cost - old_cost

		if old_cost - current_cost < 0.001:
			break

	print ("test set prediction")
	prediction = predict_function(test_set[0])
	accuracy(prediction, test_set[1])

	print ("CV set prediction")
	prediction = predict_function(valid_set[0])
	accuracy(prediction, valid_set[1])
