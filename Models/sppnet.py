# Implementation of Sppnet
# Link to Paper : https://arxiv.org/pdf/1406.4729.pdf
# Author : sentient07
# For using the Spatial pooling layer, refer to: https://github.com/Theano/Theano/pull/5395

from __future__ import absolute_import

import theano
import theano.tensor as tensor
import numpy as np
import lasagne
from lasagne.layers import (batch_norm, Conv2DLayer, Deconv2DLayer,
                            DropoutLayer, ReshapeLayer, InputLayer, DenseLayer, NonlinearityLayer, 
                            FlattenLayer, LocalResponseNormalization2DLayer)
from lasagne.layers.pool import MaxPool2DLayer, Upscale2DLayer
from lasagne.nonlinearities import softmax

# SPPnet over ZF-5 model
def build_model(inp):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=inp)
    net['conv1'] = Conv2DLayer(net['input'], 96, 7, stride=2, pad=(1, 1), W=lasagne.init.Uniform())
    net['norm1'] = LocalResponseNormalization2DLayer(net['conv1'], alpha=0.00005)
    net['pool1'] = MaxPool2DLayer(net['norm1'], 3, stride=2)
    net['conv2'] = Conv2DLayer(net['pool1'], 256, 5, stride=2, pad=(0, 0), W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['norm2'] = LocalResponseNormalization2DLayer(net['conv2'], alpha=0.00005)
    net['pool2'] = MaxPool2DLayer(net['norm2'], 3, stride=2)
    net['conv3'] = Conv2DLayer(net['pool2'], 384, 3, stride=1, pad=(1, 1), W=lasagne.init.Uniform())
    net['conv4'] = Conv2DLayer(net['conv3'], 384, 3, stride=1, pad=(1, 1), W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['conv5'] = Conv2DLayer(net['conv4'], 256, 3, stride=1, pad=(1, 1), W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['p5_spm6'] = FlattenLayer(MaxPool2DLayer(net['conv5'], 3, stride=2))
    net['p5_spm3'] = FlattenLayer(MaxPool2DLayer(net['conv5'], 5, stride=4))
    net['p5_spm2'] = FlattenLayer(MaxPool2DLayer(net['conv5'], 7, stride=7))
    net['p5_spm1'] = MaxPool2DLayer(net['conv5'], 13, stride=13)
    net['p5_spm1_f'] = FlattenLayer(net['p5_spm1'])
    net['conc_spm'] = ConcatLayer([net['p5_spm6'], net['p5_spm3'], net['p5_spm2'], net['p5_spm1_f']])
    net['fc6'] = DenseLayer(net['conc_spm'], num_units=4096, W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['fc6_drop'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_drop'], num_units=4096, W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['fc7_drop'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['fc6_drop'], num_units=1000, W=lasagne.init.Uniform(), b=lasagne.init.Constant(1.))
    net['out'] = NonlinearityLayer(net['fc8'], softmax)
    return net
