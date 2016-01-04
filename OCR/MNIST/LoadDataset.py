#!/usr/bin/python
#For loading and returning the dataset.
import gzip, cPickle

class LoadDataset():

	def return_dataset(self):
		with gzip.open('mnist.pkl.gz', 'rb') as f:
			train_set, valid_set, test_set = cPickle.load(f)
		list1 = [train_set, valid_set, test_set] 
		for sets in list1 :
			yield sets