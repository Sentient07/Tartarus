#!/usr/bin/python
#Logistic regression
#Use the Virtual env inside ML

import numpy as np
from theano import *
import theano.tensor as T
import cPickle, gzip
import math

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
	inp_test_set_cv_set, oup_cv_set = shared_dataset(valid_set)

	#split to batches
	n_train_batches = inp_train_set.get_value(borrow=True).shape[0] / 500
	n_valid_batches = inp_test_set.get_value(borrow=True).shape[0] / 500
	n_test_batches = inp_test_set_cv_set.get_value(borrow=True).shape[0] / 500

	weights = (10, 40)
	bias = 10

	hidden1_weights = (40, 784)
	hidden1_bias = 40

	hidden2_weights = (40, 40)
	hidden2_bias = 40


	#using shared function because it's a convenient
	w2 = shared(np.random.random(weights) - 0.5, name="w2")
	b2 = shared(np.random.random(bias) - 0.5, name="b2")
	w3 = shared(np.random.random(hidden2_weights) - 0.5, name="w3")
	b3 = shared(np.random.random(hidden2_bias) - 0.5, name="b3")
	w1 = shared(np.random.random(hidden1_weights) - 0.5, name="w1")
	b1 = shared(np.random.random(hidden1_bias) - 0.5, name="b1")


	#little confusion between usage of dmatrix and dvector
	inp = T.dmatrix("inp")
	labels = T.dmatrix("labels")
	#Encode labels

	#The hidden layer computation
	hidden1 = T.nnet.sigmoid(inp.dot(w1.transpose()) + b1)
	print hidden1.shape

	hidden2 = T.nnet.sigmoid(hidden1.dot(w3.transpose()) + b3)
	#The softmax computation
	output = T.nnet.softmax(hidden2.dot(w2.transpose()) + b2)
	predict = T.argmax(output, axis=1)
	
	cost = T.nnet.binary_crossentropy(output, labels).mean()

	#Regularization
	cost = cost + 0.0001 * ((w1 * w1).sum() + 
		(w2 * w2).sum()+ 
		(w3 * w3).sum() + 
		(b1 * b1).sum() +
		(b2 * b2).sum() +
		(b3 * b3).sum())



	predict_function = function([inp], predict)

	alpha = T.dscalar('alpha')
	#weights = [hidden_w_shared, w_shared, hidden_b_shared, b_shared]
	weights = [w1, w2, w3, b1, b2, b3]
	updates = [(w, w - alpha * grad(cost, w)) for w in weights]
	train = function([inp, labels, alpha], cost, updates=updates)

	temp_alpha = 10.0
	labeled = encode_labels(train_set[1], 9)
	old_cost = 1000
	current_cost = 100
	while ((old_cost - current_cost > 0.00005) or (old_cost - current_cost < -0.00005)):
		old_cost = current_cost
		current_cost = float(train(train_set[0], labeled, temp_alpha))

	print ("test set prediction")
	prediction = predict_function(test_set[0])
	accuracy(prediction, test_set[1])

	print ("CV set prediction")
	prediction = predict_function(valid_set[0])
	accuracy(prediction, valid_set[1])



