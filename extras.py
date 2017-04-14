import theano
import theano.tensor as tensor
import numpy as np
import cv2

# Histogram normalization so that difference in brigntness/lighting in an image won't affect the result
def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm

def log_sum_exp(x, axis=1):
    m = tensor.max(x, axis=axis)
    return m+tensor.log(tensor.sum(tensor.exp(x-m.dimshuffle(0,'x')), axis=axis))

def combined_alt_loss_function(l_gen, l_des):
    return -0.5 * tensor.mean(l_desc) + 0.5 * tensor.mean(tensor.nnet.softplus(l_desc)) + 0.5 * tensor.mean(tensor.nnet.softplus(l_gen))