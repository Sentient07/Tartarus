#!/usr/bin/python
#Logistic regression
#Use the Virtual env inside ML

import numpy as np
from theano import *
import theano.tensor as T
import cPickle, gzip



def encode_labels(labels, max_index):
    """Encode the labels into binary vectors."""
    # Allocate the output labels, all zeros.
    encoded = np.zeros((labels.shape[0], max_index + 1))
    
    # Fill in the ones at the right indices.
    for i in xrange(labels.shape[0]):
        encoded[i, labels[i]] = 1
    return encoded




#Storing the dataset in shared theano variable
def shared_dataset(data_xy, borrow=True):
	data_x, data_y = data_xy
	shared_x = theano.shared(numpy.asarray(data_x,
				dtype=theano.config.floatX),
				borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y,
				dtype=theano.config.floatX),
				borrow=borrow)
	return shared_x, T.cast(shared_y, 'int32')


def errors(self, y):

	return T.mean(T.neq(self.y_pred, y))


def accuracy(predicted, actual):
    total = 0.0
    correct = 0.0
    for p, a in zip(predicted, actual):
        total += 1
        if p == a:
            correct += 1
    return correct / total

if __name__ == '__main__':

	#loading dataset
	with gzip.open('mnist.pkl.gz', 'rb') as f:
		train_set, valid_set, test_set = cPickle.load(f)

    #Storing dataset in variable
	inp_train_set, oup_train_set = shared_dataset(train_set)
	inp_test_set, oup_test_set = shared_dataset(test_set)
	inp_cv_set, oup_cv_set = shared_dataset(valid_set)

	#split to batches
	n_train_batches = inp_train_set.get_value(borrow=True).shape[0] / 500
	n_valid_batches = inp_test_set.get_value(borrow=True).shape[0] / 500
	n_test_batches = inp_cv_set.get_value(borrow=True).shape[0] / 500

	weights = (10, 784) #784 pixels, 10 possibility
	bias = 10

	#using shared function because it's a convention
	w_shared = shared(np.random.random(weights)-0.5, name="w_shared")
	b_shared = shared(np.random.random(bias)-0.5, name="b_shared")
	#little confusion between usage of dmatrix and dvector
	inp = T.dmatrix("inp")
	labels = T.dmatrix("labels")
	#Encode labels

	#The softmax computation
	output = T.nnet.softmax(inp.dot(w_shared.transpose()) + b_shared)
	predict_y = T.argmax(output, axis=1)
	predict_func = function([inp], predict_y)
	cost = T.nnet.binary_crossentropy(output, labels).mean()
	cost_compute = function([inp, labels], cost)
	weight_grad = grad(cost, w_shared)
	bias_grad = grad(cost, b_shared)

	alpha = T.dscalar('alpha')
	updates = [(w_shared, w_shared - alpha * weight_grad),
	           (b_shared, b_shared - alpha * bias_grad)]

	train = function([inp, labels, alpha], cost, updates=updates)
	labeled = encode_labels(train_set[1], 9)
	temp_alpha = 10.0
	costs = []
	while True:
	    costs.append(float(train(train_set[0], labeled, temp_alpha)))
	    
	    if len(costs) % 10 == 0:
	        print 'Epoch', len(costs), 'with cost', costs[-1], 'and alpha', temp_alpha
	    if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
	        if temp_alpha < 0.2:
	            break
	        else:
	            temp_alpha = temp_alpha / 1.5

	compute_prediction = predict_func(test_set)
	accuracy(prediction, test_set[1])
	prediction = predict_func(test_set[0])
