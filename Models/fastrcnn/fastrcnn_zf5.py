
# Fast RCNN over ZF Net
# author : sentient07

"""
Instructions : 

This requires the ROIPool branch of theano and lasagne, from https://www.github.com/sentient07/Theano
and https://github.com/Sentient07/Lasagne respectively.

For setting up the ROI, you need the roidb. Along with that, you need iou metric to evaluate overlap.
The code for that is in a separate repository. This contains only the model.
"""

import theano
import theano.tensor as tensor
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer, RoIPoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer, get_output, get_all_params
from lasagne.init import Normal
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import sgd
import numpy as np
import time

def zf_rpn(inp, roidb):
    net = {}
    # Replace this InputLayer with ROIData Layer
    # And train over another network to obtain 
    # CaffeNet model or the RPN, Regional Proposal Network

    net['input'] = InputLayer((None, 3, 224, 224), input_var=inp)
    net['conv1'] = Conv2DLayer(net['input'], 96, 11, stride=4, pad=5)
    net['pool1'] = MaxPool2DLayer(net['conv1'], 3, stride=2, pad=1)
    net['norm1'] = LocalResponseNormalization2DLayer(net['pool1'])
    net['conv2'] = Conv2DLayer(net['norm1'], 256, 5, stride=2, pad=2, W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['pool2'] = MaxPool2DLayer(net['conv2'], 3, stride=2, pad=1)
    net['norm2'] = LocalResponseNormalization2DLayer(net['pool2'])
    net['conv3'] = Conv2DLayer(net['pool2'], 384, 3, stride=1, pad=1, W=lasagne.init.Uniform())
    net['conv4'] = Conv2DLayer(net['conv3'], 384, 3, stride=1, pad=1, W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['conv5'] = Conv2DLayer(net['conv4'], 256, 3, stride=1, pad=1, W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['roipool5'] = RoIPoolLayer(net['conv5'], 6, 6, 0.0625, roidb)
    net['fc6'] = DenseLayer(net['roipool5'], num_units=4096, W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['fc6_drop'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_drop'], num_units=4096, W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['fc7_drop'] = DropoutLayer(net['fc7'], p=0.5)
    net['cls_dense'] = DenseLayer(net['fc7_drop'], num_units=21, W=lasagne.init.Normal(), nonlinearity=None)
    net['cls_score'] = NonlinearityLayer(net['cls_dense'], softmax)
    net['bbox_pred'] = DenseLayer(net['fc7_drop'], num_units=84, W=lasagne.init.Normal(std=0.001), nonlinearity=None)
    return net
