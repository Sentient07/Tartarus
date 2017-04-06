# Implementation of DCGAN(Deep Conv Generative Adversarial Network)
# Link to the paper : https://arxiv.org/pdf/1511.06434.pdf
# __author__ : @sentient07

from __future__ import absolute_import
import theano
import lasagne
from lasagne.layers import (Conv2DLayer, MaxPool2DLayer, BatchNormLayer, DenseLayer,
                            InputLayer, Deconv2DLayer, NonlinearityLayer)
from lasagne.init import Normal, Uniform
from lasagne.nonlinearities import tanh, LeakyRectify, sigmoid

def build_generator(inp):
    net['inp'] = InputLayer((None, 100), input_var=inp)
    net['dconv1'] = Deconv2DLayer(net['inp'], 512, 4)
    net['bn1'] = BatchNormLayer(net['dconv1'])
    net['relu1'] = NonlinearityLayer(net['bn1'])
    net['dconv2'] = Deconv2DLayer(net['relu1'], 256, 4, stride=2, pad=1)
    net['bn2'] = BatchNormLayer(net['dconv2'])
    net['relu2'] = NonlinearityLayer(net['bn2'])
    net['dconv3'] = Deconv2DLayer(net['relu2'], 128, 4, stride=2, pad=1)
    net['bn3'] = BatchNormLayer(net['dconv3'])
    net['relu3'] = NonlinearityLayer(net['bn3'])
    net['dconv4'] = Deconv2DLayer(net['relu3'], 64, 4, stride=2, pad=1)
    net['bn4'] = BatchNormLayer(net['dconv4'])
    net['relu4'] = NonlinearityLayer(net['bn4'])
    net['dconv5'] = Deconv2DLayer(net['relu4'], 3, 4, stride=2, pad=1, nonlinearity=tanh)
    return net

def build_desc(inp):
    net['inp'] = InputLayer((None, 3, 64, 64), input_var=inp)
    net['conv1'] = Conv2DLayer(net['inp'], 64, 4, stride=2, pad=1, nonlinearity=LeakyRectify(leakiness=0.2))
    net['conv2'] = Conv2DLayer(net['conv1'], 128, 4, stride=2, pad=1)
    net['bn2'] = BatchNormLayer(net['conv2'])
    net['lrelu2'] = NonlinearityLayer(net['bn2'], nonlinearity=LeakyRectify(leakiness=0.2))
    net['conv3'] = Conv2DLayer(net['lrelu2'], 256, 4, stride=2, pad=1)
    net['bn3'] = BatchNormLayer(net['conv3'])
    net['lrelu3'] = NonlinearityLayer(net['bn3'], nonlinearity=LeakyRectify(leakiness=0.2))
    net['conv4'] = Conv2DLayer(net['lrelu3'], 512, 4, stride=2, pad=1)
    net['bn4'] = BatchNormLayer(net['conv4'])
    net['lrelu4'] = NonlinearityLayer(net['bn4'], nonlinearity=LeakyRectify(leakiness=0.2))
    net['conv5'] = Conv2DLayer(net['lrelu4'], 1, 4, nonlinearity=sigmoid)
    return net
