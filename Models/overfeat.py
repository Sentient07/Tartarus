# Implementation of Overfeat.
# This was the first deep learning code I wrote, hence it's in this repo,
# despite being one of the most common model

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer

def build_overfeat(inp):
    net['inp'] = InputLayer((None, 3, 231, 231), input_var=inp)
    net['conv1'] = Conv2DLayer(net['inp'], 96, 11, stride=4, pad='valid')
    net['pool1'] = MaxPool2DLayer(net['conv1'], 2)
    net['conv2'] = Conv2DLayer(net['pool1'], 256, 5, pad='valid')
    net['pool2'] = MaxPool2DLayer(net['conv2'], 2)
    net['conv3'] = Conv2DLayer(net['pool2'], 512, 3, pad='same')
    net['conv4'] = Conv2DLayer(net['conv3'], 1024, 3, pad='same')
    net['conv5'] = Conv2DLayer(net['conv4'], 1024, 3, pad='same')
    net['pool3'] = MaxPool2DLayer(net['conv5'], 2)
    net['fc6'] = DenseLayer(net['pool3'], 3072)
    net['fc7'] = DenseLayer(net['fc6'], 4096)
    net['fc8'] = DenseLayer(net['fc7'], 1000, nonlinearity=None)
    return net