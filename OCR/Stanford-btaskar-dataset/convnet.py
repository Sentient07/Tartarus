#!/usr/bin/python
#Logistic regression
#Use the Virtual env inside ML

import numpy as np
from theano import *
import theano.tensor as T
import cPickle, gzip
import math
import lasagne
from LoadDataset import LoadDataset
