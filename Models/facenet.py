# Model implementation of Facenet
# This is a ZFNet based model with 1 X 1 convolutions
# Link to Paper : https://arxiv.org/pdf/1503.03832.pdf


import lasagne
import theano
from lasagne.layers import (Conv2DLayer, MaxPool2DLayer, BatchNormLayer, DenseLayer,
                            InputLayer, Deconv2DLayer, NonlinearityLayer, get_output, get_output_shape, batch_norm,
                            NINLayer, GlobalPoolLayer, LocalResponseNormalization2DLayer, DropoutLayer, ReshapeLayer)

def build_facenet(input)
    net = InputLayer((None, 220, 220, 3), input_var=input)
    net = Conv2DLayer(net, 64, 7, stride=2, pad=2)
    net = MaxPool2DLayer(net, 2)
    net = LocalResponseNormalization2DLayer(net)
    net = Conv2DLayer(net, 64, 1, stride=1)
    net = LocalResponseNormalization2DLayer(Conv2DLayer(net, 192, 3, stride=1, pad=1))
    net = MaxPool2DLayer(net, 2)
    net = Conv2DLayer(net, 192, 1, stride=1)
    net = Conv2DLayer(net, 384, 3, stride=1, pad=1)
    net = MaxPool2DLayer(net, 2)
    net = Conv2DLayer(net, 384, 1, stride=1)
    net = Conv2DLayer(net, 256, 3, stride=1, pad=1)
    net = Conv2DLayer(net, 256, 1, stride=1)
    net = Conv2DLayer(net, 256, 3, stride=1, pad=1)
    net = Conv2DLayer(net, 256, 1, stride=1)
    net = Conv2DLayer(net, 256, 3, stride=1, pad=1)
    net = MaxPool2DLayer(net, 2)
    net =  DropoutLayer(DenseLayer(net, 4096), p=0.2)
    net = DropoutLayer(DenseLayer(net, 4096), p=0.2)
    net = ReshapeLayer(DenseLayer(net, 128), (1, 1, 128))
    return net
