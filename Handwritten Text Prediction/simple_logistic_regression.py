#!/usr/bin/python
#Logistic regression
#Use the Virtual env inside ML

import numpy as np
from theano import *
import theano.tensor as T
import cPickle, gzip


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

def encode_labels(labels, max_index):
    """Encode the labels into binary vectors."""
    # Allocate the output labels, all zeros.
    encoded = np.zeros((labels.shape[0], max_index + 1))
    
    # Fill in the ones at the right indices.
    for i in xrange(labels.shape[0]):
        encoded[i, labels[i]] = 1
    return encoded


def errors(self, y):

	return T.mean(T.neq(self.y_pred, y))


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

	weights = (10, 50)
	bias = 10

	hidden_weights = (50, 784)
	hidden_bias = 50


	#using shared function because it's a convenient
	w_shared = shared(np.random.random(weights), name="w_shared")
	b_shared = shared(np.random.random(bias), name="b_shared")
	hidden_w_shared = shared(np.random.random(hidden_weights), name="hidden_w_shared")
	hidden_b_shared = shared(np.random.random(hidden_bias), name="hidden_b_shared")


	#little confusion between usage of dmatrix and dvector
	inp = T.dmatrix("inp")
	labels = T.dmatrix("labels")
	#Encode labels

	#The hidden layer computation
	hidden = T.nnet.sigmoid(inp.dot(hidden_w_shared.transpose()) + hidden_b_shared)

	#The softmax computation
	output = T.nnet.softmax(hidden.dot(w_shared.transpose()) + b_shared)
	predict_y = T.argmax(output, axis=1)
	
	cost = T.nnet.binary_crossentropy(output, labels).mean()
	cost_compute = function([inp, labels], cost)

	#Regularization
	cost = cost + 0.0001 * ((w_shared * w_shared).sum() + (hidden_w_shared * hidden_w_shared).sum()
			+ (b_shared * b_shared).sum() + (hidden_b_shared * hidden_b_shared).sum())

	predict_func = function([inp], predict_y)

	#output 
	weight_grad = grad(cost, w_shared)
	bias_grad = grad(cost, b_shared)

	#hidden layer
	hidden_weight_grad = grad(cost, hidden_w_shared)
	hidden_bias_grad = grad(cost, hidden_b_shared)


	alpha = T.dscalar('alpha')

	updates = [(w_shared, w_shared - alpha * weight_grad),
				(hidden_w_shared, hidden_w_shared - alpha * hidden_weight_grad),
	           (b_shared, b_shared - alpha * bias_grad),
	           (hidden_b_shared, hidden_b_shared - alpha * hidden_bias_grad)]

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
 	
 	print "len(costs)" 
 	print len(costs) 
	compute_prediction = predict_func(test_set[0])
	accuracy(compute_prediction, test_set[1])
	