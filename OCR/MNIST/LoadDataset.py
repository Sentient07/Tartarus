#!/usr/bin/python
#For loading and returning the dataset.


class LoadDataset():

	def return_dataset():
		with gzip.open('mnist.pkl.gz', 'rb') as f:
			train_set, valid_set, test_set = cPickle.load(f)