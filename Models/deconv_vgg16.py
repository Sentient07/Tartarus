# Implementation of Deconv network for semantic Segmentation paper
# Link to Paper : https://arxiv.org/pdf/1505.04366.pdf
# Author : sentient07
# This just contains the model

from __future__ import absolute_import

import theano
import theano.tensor as tensor
import numpy as np
import lasagne
from lasagne.layers import (batch_norm, Conv2DLayer, Deconv2DLayer,
                            DropoutLayer, ReshapeLayer, InputLayer, DenseLayer, NonlinearityLayer)
from lasagne.layers.pool import MaxPool2DLayer, Upscale2DLayer
from lasagne.nonlinearities import softmax

def build_model(inp):
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), input_var=input_var)
    net['conv1_1'] = batch_norm(Conv2DLayer(net['input'], 64, 3, pad=1, flip_filters=False))
    net['conv1_2'] = batch_norm(Conv2DLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False))
    net['pool1'] = MaxPool2DLayer(net['conv1_2'], 2)
    net['conv2_1'] = batch_norm(Conv2DLayer(net['pool1'], 128, 3, pad=1, flip_filters=False))
    net['conv2_2'] = batch_norm(Conv2DLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False))
    net['pool2'] = MaxPool2DLayer(net['conv2_2'], 2)
    net['conv3_1'] = batch_norm(Conv2DLayer(net['pool2'], 256, 3, pad=1, flip_filters=False))
    net['conv3_2'] = batch_norm(Conv2DLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False))
    net['conv3_3'] = batch_norm(Conv2DLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False))
    net['pool3'] = MaxPool2DLayer(net['conv3_3'], 2)
    net['conv4_1'] = batch_norm(Conv2DLayer(net['pool3'], 512, 3, pad=1, flip_filters=False))
    net['conv4_2'] = batch_norm(Conv2DLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False))
    net['conv4_3'] = batch_norm(Conv2DLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False))
    net['pool4'] = MaxPool2DLayer(net['conv4_3'], 2)
    net['conv5_1'] = batch_norm(Conv2DLayer(net['pool4'], 512, 3, pad=1, flip_filters=False))
    net['conv5_2'] = batch_norm(Conv2DLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False))
    net['conv5_3'] = batch_norm(Conv2DLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False))
    net['pool5'] = MaxPool2DLayer(net['conv5_3'], 2)
    net['fc6'] = batch_norm(DenseLayer(net['pool5'], num_units=4096))
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = batch_norm(DenseLayer(net['fc6_dropout'], num_units=4096))
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['reshape'] = ReshapeLayer(net['fc7_dropout'], (-1, 4096, 1, 1))
    net['deconv6'] = batch_norm(Deconv2DLayer(net['reshape'], 512, 7))
    net['unpool5'] = Upscale2DLayer(net['deconv6'], 2)
    net['deconv5_3'] = batch_norm(Deconv2DLayer(net['unpool5'], 512, 3, crop=1, flip_filters=False))
    net['deconv5_2'] = batch_norm(Deconv2DLayer(net['deconv5_3'], 512, 3, crop=1, flip_filters=False))
    net['deconv5_1'] = batch_norm(Deconv2DLayer(net['deconv5_2'], 512, 3, crop=1, flip_filters=False))
    net['unpool4'] = Upscale2DLayer(net['deconv5_1'], 2)
    net['deconv4_3'] = batch_norm(Deconv2DLayer(net['unpool4'], 512, 3, crop=1, flip_filters=False))
    net['deconv4_2'] = batch_norm(Deconv2DLayer(net['deconv4_3'], 512, 3, crop=1, flip_filters=False))
    net['deconv4_1'] = batch_norm(Deconv2DLayer(net['deconv4_2'], 512, 3, crop=1, flip_filters=False))
    net['unpool3'] = Upscale2DLayer(net['deconv4_1'], 2)
    net['deconv3_3'] = batch_norm(Deconv2DLayer(net['unpool3'], 256, 3, crop=1, flip_filters=False))
    net['deconv3_2'] = batch_norm(Deconv2DLayer(net['deconv3_3'], 256, 3, crop=1, flip_filters=False))
    net['deconv3_1'] = batch_norm(Deconv2DLayer(net['deconv3_2'], 256, 3, crop=1, flip_filters=False))
    net['unpool2'] = Upscale2DLayer(net['deconv3_1'], 2)
    net['deconv2_2'] = batch_norm(Deconv2DLayer(net['unpool2'], 128, 3, crop=1, flip_filters=False))
    net['deconv2_1'] = batch_norm(Deconv2DLayer(net['deconv2_2'], 128, 3, crop=1, flip_filters=False))
    net['unpool1'] = Upscale2DLayer(net['deconv2_1'], 2)
    net['deconv1_1'] = batch_norm(Deconv2DLayer(net['unpool1'], 64, 3, crop=1, flip_filters=False))
    net['deconv1_2'] = batch_norm(Deconv2DLayer(net['deconv1_1'], 64, 3, crop=1, flip_filters=False))
    net['out'] = NonlinearityLayer(DenseLayer(net['deconv1_2'], 21), softmax)
    return net
