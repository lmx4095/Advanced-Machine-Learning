import numpy as np
import matplotlib.pyplot as pyplot
import pylab
faces = np.load('freyfaces.npy',encoding='latin1')
from collections import OrderedDict
def show_examples(x,square=True):
    N = x.shape[0]
    if square:
        d = int(np.ceil(np.sqrt(N)))
        d1 = int(np.ceil(N/d))
    else:
        d = N
        d1 = 1
    im = np.zeros([d1*28,d*20])
    for i in range(d1):
        for j in range(d):
            c = i*d + j
            if c<N:
                im[i*28:(i+1)*28,j*20:(j+1)*20] = x[c,:].reshape([28,20])
    pyplot.figure(figsize=[d,d1])
    pyplot.imshow(im,interpolation=None,cmap='Greys')
    pylab.show()

import theano
import theano.tensor as T
import numpy as np
theano.config.floatX='float32'
theano.config.optimization='fastrun'
theano.config.exception_verbosity='high'

def sampler(mu, sigma, seed=1):
    srng = T.shared_randomstreams.RandomStreams(seed=seed)
    eps = srng.normal(mu.shape)
    z = mu + sigma* eps
    return z

def make_shared(name,value):
    if value is np.ndarray:
        value = value.astype('float32')
    else:
        value = np.float32(value)
    return theano.shared(name=name,value=value.astype('float32'))

# an adam implementation from https://github.com/skaae/

def adam(loss, all_params, learning_rate=0.0002, beta1=0.1, beta2=0.001, epsilon=1e-8, gamma=1-1e-7):
    updates = []
    all_grads = theano.grad(loss, all_params)

    i = theano.shared(np.float32(1))
    i_t = i + 1.
    fix1 = 1. - (1. - beta1) ** i_t
    fix2 = 1. - (1. - beta2) ** i_t
    beta1_t = 1 - (1 - beta1) * gamma ** (i_t - 1)
    learning_rate_t = learning_rate * (T.sqrt(fix2) / fix1)

    for param_i, g in zip(all_params, all_grads):
        m = theano.shared(np.zeros(param_i.get_value().shape, dtype='float32'))
        v = theano.shared(np.zeros(param_i.get_value().shape, dtype='float32'))
        m_t = (beta1_t * g) + ((1. - beta1_t) * m)
        v_t = (beta2 * g ** 2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        param_i_t = param_i - (learning_rate_t * g_t)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((param_i, param_i_t))

    updates.append((i, i_t))
    return updates


X = T.dmatrix('X').astype('float32')
y = T.dvector('y').astype('float32')

w = make_shared(name='w',value=np.zeros(5))
b = make_shared(name='b',value=0)
prediction = T.dot(X,w) + b

cost = 0.5*T.mean((prediction - y)**2.0)

cost2 = cost + 0.1*T.sum(w**2)

gw2,gb2 = T.grad(cost2,[w,b])

train2 = theano.function(inputs=[X,y],outputs=cost,updates=((w, w - 0.05 * gw2),(b, b - 0.05 * gb2)))

b.set_value(0.0)
w.set_value(np.zeros(5,dtype='float32'))

N = 100
d = 5
np.random.seed(1)
data_X = np.random.randn(N,d).astype('float32')
true_w = np.zeros(d).astype('float32')
true_w[0:1] = 1.0
true_b = 5.0
true_y = np.dot(data_X,true_w) + true_b
data_y = true_y + 0.1*np.random.randn(N).astype('float32')

def relu(x):
    return T.switch(x<0, 0, x)

x = T.dmatrix('x').astype('float32')
d = faces.shape[1]
nh = 256
nz = 4

W_enc = make_shared(name='W_enc',value=0.01*np.random.randn(d,nh))
b_enc = make_shared(name='b_enc',value=np.zeros(nh))
W_encmu = make_shared(name='W_encmu',value=0.01*np.random.randn(nh,nz))
b_encmu = make_shared(name='b_encmu',value=np.zeros(nz))
W_encsigma = make_shared(name='W_encsigma',value=0.01*np.random.randn(nh,nz))
b_encsigma = make_shared(name='b_encsigma',value=np.zeros(nz))

h_enc = relu(T.dot(x, W_enc) + b_enc)
mu_enc = T.dot(h_enc,W_encmu) + b_encmu
log_sigma2_enc = T.dot(h_enc, W_encsigma) + b_encsigma

encoder_params = [W_enc,b_enc,W_encmu,b_encmu,W_encsigma,b_encsigma]
sigma_enc = T.exp(0.5*log_sigma2_enc)
z = sampler(mu=mu_enc,sigma=sigma_enc)

W_enc = make_shared(name='W_enc',value=0.01*np.random.randn(d,nh))
b_enc = make_shared(name='b_enc',value=np.zeros(nh))
W_encmu = make_shared(name='W_encmu',value=0.01*np.random.randn(nh,nz))
b_encmu = make_shared(name='b_encmu',value=np.zeros(nz))
W_encsigma = make_shared(name='W_encsigma',value=0.01*np.random.randn(nh,nz))
b_encsigma = make_shared(name='b_encsigma',value=np.zeros(nz))

h_enc = relu(T.dot(x, W_enc) + b_enc)
mu_enc = T.dot(h_enc,W_encmu) + b_encmu
log_sigma2_enc = T.dot(h_enc, W_encsigma) + b_encsigma

encoder_params = [W_enc,b_enc,W_encmu,b_encmu,W_encsigma,b_encsigma]
sigma_enc = T.exp(0.5*log_sigma2_enc)
z = sampler(mu=mu_enc,sigma=sigma_enc)

W_dec = make_shared(name='W_dec',value=0.01*np.random.randn(nz,nh))
b_dec = make_shared(name='b_dec',value=np.zeros(nh))
W_decmu = make_shared(name='W_decmu',value=0.01*np.random.randn(nh,d))
b_decmu = make_shared(name='b_decmu',value=np.zeros(d))
W_decsigma = make_shared(name='W_decsigma',value=0.01*np.random.randn(nh,d))
b_decsigma = make_shared(name='b_decsigma',value=np.zeros(d))
h_dec = relu(T.dot(z, W_dec) + b_dec)
mu_dec = T.dot(h_dec, W_decmu) + b_decmu
log_sigma2_dec = T.dot(h_dec, W_decsigma) + b_decsigma
decoder_params = [W_dec,b_dec,W_decmu,b_decmu,W_decsigma,b_decsigma]
sigma_dec = T.exp(0.5*log_sigma2_dec)

all_params = encoder_params + decoder_params
logpxz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma2_dec) -0.5 * ((x - mu_dec)**2 / T.exp(log_sigma2_dec))).sum(axis=1)
KLD = -0.5 * T.sum(1 + np.log(sigma_enc) - mu_enc**2 - T.exp(log_sigma2_enc), axis=1)
logpx = T.mean(logpxz - KLD)


x_train = faces[0:1500,:].astype('float32')
x_valid = faces[1500:,:].astype('float32')

batch_size = 100
batch_count = x_train.shape[0]//batch_size
batches = np.arange(batch_count)

updates = adam(-logpx,all_params)
likelihood = theano.function([x], logpx)
encode = theano.function([x], z)
decode = theano.function([z], mu_dec)

reconstruct = theano.function([x],mu_dec)
train = theano.function(inputs=[x],outputs=logpx,updates=updates)

np.random.seed(1)
for i in range(200):
    np.random.shuffle(batches)
    for batch in batches:
        minibatch = x_train[batch*batch_size:(batch+1)*batch_size, :]
        train(minibatch)
    if i % 50 == 0:
        print("iter",i,"current likelihood",likelihood(x_valid))

show_examples(reconstruct(x_train[1:1500:30,:]))



