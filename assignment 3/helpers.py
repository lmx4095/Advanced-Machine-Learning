import numpy as np
from time import time

import os
import theano
import theano.tensor as T
theano.config.optimization=None
import numpy as np
import scipy
import matplotlib.pyplot as pyplot
from skimage.filters import gabor_kernel
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros, ceil, ones
from scipy.misc import imsave

seed = 42
np_rng = np.random.RandomState(seed)
# an adam implementation from https://github.com/skaae/
def adam(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001,
         epsilon=1e-8, gamma=1-1e-7):   
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """
    updates = []
    all_grads = theano.grad(loss, all_params)

    i = theano.shared(np.float32(1))  
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)  
    learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1)

    for param_i, g in zip(all_params, all_grads):
        m = theano.shared(
            np.zeros(param_i.get_value().shape, dtype='float32'))
        v = theano.shared(
            np.zeros(param_i.get_value().shape, dtype='float32'))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m) 
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_i_t = param_i - (learning_rate_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t) )
    updates.append((i, i_t))
    return updates

# code from https://github.com/Newmu/dcgan_code.git

def grayscale_grid_vis(X, xxx_todo_changeme, save_path=None):
    (nh, nw) = xxx_todo_changeme
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = n//nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
    if save_path is not None:
        imsave(save_path, img)
    return img

def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            X = X*g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a)*0. + a*u
            s = (1. - a)*1. + a*s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X*g + b
    else:
        raise NotImplementedError
    return X

from sklearn import utils as skutils
def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], str):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)

def transform(X,nc=1,npx=28):
    return (np.float32(X)/255.).reshape(-1, nc, npx, npx)

def inverse_transform(X,npx=28):
    X = X.reshape(-1, npx, npx)
    return X  
#end of code from from https://github.com/Newmu/dcgan_code.git

def gray_plot(im,d=1,d1=1,new_figure=True):
    if new_figure:
        pyplot.figure(figsize=[d,d1])
    pyplot.imshow(im,interpolation='None',cmap='Greys')
    
def show_examples(x,square=True,h=9,w=9):
    N = x.shape[0]
    if square:
        d = int(ceil(np.sqrt(N)))
        d1 = int(ceil(N/d))
    else:
        d = N
        d1 = 1
        
    im = ones([1+d1*(h+1),1+d*(w+1)])
    for i in range(d1):
        for j in range(d):
            c = i*d + j
            if c<N:
                im[1+i*(h+1):(i+1)*(h+1),1+j*(w+1):(j+1)*(w+1)] = x[c,:].reshape([h,w])
    gray_plot(im,d,d1,True)
    
def load_mnist(dataset="training", digits=np.arange(10), path="data"):
    if dataset == "training":
        fname_img = os.path.join('E:\\SILS Course\\COMP 790\\assign 3\\hw3', 'train-images-idx3-ubyte')
        fname_lbl = os.path.join('E:\\SILS Course\\COMP 790\\assign 3\\hw3', 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join('E:\\SILS Course\\COMP 790\\assign 3\\hw3', 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join('E:\\SILS Course\\COMP 790\\assign 3\\hw3', 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels
