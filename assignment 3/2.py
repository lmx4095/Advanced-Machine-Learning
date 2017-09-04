import matplotlib.pyplot as pyplot
import numpy as np
from time import time

import theano

theano.config.optimization = None
import theano.tensor as T
import numpy as np
import scipy

from skimage.filters import gabor_kernel
from helpers import *

images, _ = load_mnist()


def gabor_patch(d, lam, theta, sigma, phase):
    x = (np.linspace(1, d, d) / d) - 0.5
    freq = d / lam
    phase = phase * 2 * np.pi
    xs, ys = np.meshgrid(x, x)
    theta = (theta / 360) * 2 * np.pi
    xr = xs * np.cos(theta)
    yr = ys * np.sin(theta)
    mask = np.sin(((xr + yr) * freq * 2 * np.pi) + phase)
    gauss = np.exp(-1 / (2 * (sigma / d) ** 2) * ((xs ** 2) + (ys ** 2)))

    return mask * gauss


img = images[1, :, :] / 255
img = img - np.mean(img.flatten())
img = img / np.sqrt(np.sum(img.flatten() ** 2.0))

f = 9
d = img.shape[0]
r = gabor_patch(f, 0.1, 1 / 4 * np.pi, 1.5, 0.01) + 0.01

# really slow
W = np.zeros([d * d, (d - f + 1) * (d - f + 1)])
for i in range(d - f + 1):
    for j in range(d - f + 1):
        for k in range(f):
            for l in range(f):
                W[(i + k) * d + j + l, i * (d - f + 1) + j] = r[k, l]

beta_really_slow = np.dot(img.flatten(), W).reshape((d - f + 1, d - f + 1))

# slow
beta_slow = np.zeros((d - f + 1, d - f + 1))
for i in range(d - f + 1):
    for j in range(d - f + 1):
        patch = img[i:i + f, j:j + f]
        beta_slow[i, j] = np.dot(patch.flatten(), r.flatten())

# fast
beta = scipy.signal.convolve2d(img, np.flipud(np.fliplr(r)), mode='valid')
pyplot.figure(figsize=(5, 20))

pyplot.subplot(1, 3, 1)
gray_plot(img, new_figure=False)
pyplot.title('Image')

pyplot.subplot(1, 3, 2)
gray_plot(r, new_figure=False)
pyplot.title('Filter')

pyplot.subplot(1, 3, 3)
gray_plot(beta, new_figure=False)
pyplot.title('Response')
np.sum((beta - beta_slow) ** 2.0), np.sum((beta - beta_really_slow) ** 2.0)

import theano
import theano.tensor as T
import theano.tensor.signal.conv as conv
import theano.tensor.nnet.abstract_conv as abstract_conv


def make_shared(name, value):
    if type(value) is np.ndarray:
        value = value.astype('float32')
    else:
        value = np.float32(value)
    return theano.shared(name=name, value=value)


def trconv(output, filters, output_shape, filter_size, subsample=(1, 1), border_mode=(0, 0)):
    f1, f2 = (filter_size[0], filter_size[1])
    a1 = 1
    a2 = 1
    o_prime1 = subsample[0] * (output_shape[2] - 1) + a1 + f1 - 2 * border_mode[0]
    o_prime2 = subsample[1] * (output_shape[3] - 1) + a2 + f2 - 2 * border_mode[1]
    input_shape = (None, None, o_prime1, o_prime2)
    input = abstract_conv.conv2d_grad_wrt_inputs(
        output, filters, input_shape=input_shape, filter_shape=None,
        subsample=subsample, border_mode=border_mode)
    return input


img4 = img.astype('float32').reshape(1, 1, d, d)
xv = T.dtensor4('x')
xv = xv.astype('float32')
r4 = r.reshape(1, 1, f, f)
rv = make_shared('r', value=r4)
hv = T.nnet.conv2d(xv, rv, input_shape=img4.shape, filter_shape=r4.shape,
                   filter_flip=False, border_mode=(0, 0))

# tranposed convolution computed via gradient of
# convolution
# this operation creates a reconstruction, y,
# from filter responses in hv.
sv = make_shared('s', value=r.reshape((1, 1, f, f)))
print(img4.shape)
print(r4.shape)
# y = deconv(hv,sv,input_shape=img4.shape,filter_shape=r4.shape)
y = trconv(hv, sv, output_shape=(None, None, d - f, d - f), filter_size=(f, f))
recon_cost = T.mean(y)
weight_decay = 0.1 * T.mean(sv.flatten() ** 2.0)
cost = recon_cost + weight_decay
gsv = 0.5*((xv-y)**2)
train = theano.function(inputs=[xv], outputs=cost, updates=[(sv, sv - gsv)])

for it in range(1000):
    train(img4)

pyplot.figure(figsize=(5, 20))
yrecon = y.eval({xv: img4}).reshape((d, d))
pyplot.subplot(1, 3, 1)
gray_plot(img, new_figure=False)
pyplot.title('Original image')

pyplot.subplot(1, 3, 2)
gray_plot(yrecon, new_figure=False)
pyplot.title('Reconstruction')

pyplot.subplot(1, 3, 3)
gray_plot(img - yrecon, new_figure=False)
pyplot.title('Error')
print('image norm:', np.sum(img ** 2.0), 'Error norm:', np.sum((img - yrecon) ** 2.0))