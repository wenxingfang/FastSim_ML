#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: ops.py
description: ancillary ops for [arXiv/1705.02355] 
    (borrowing from [arXiv/1701.05927])
author: Luke de Oliveira (lukedeo@manifold.ai)
"""

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints, activations
from keras.layers import Lambda, ZeroPadding2D, LocallyConnected2D
from keras.layers.merge import concatenate, multiply, subtract
import tensorflow as tf
import numpy as np


def channel_softmax(x):
    e = K.exp(x - K.max(x, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s

def get_mc_info(x, n):
    size = x.shape[1]
    print('size=',size)
    x = x[:,size-n:size]
    return x


def normalize(x, epsilon):
    assert len(x.shape)==4
    x_mean, x_variance = tf.nn.moments(x, axes=[1,2,3], shift=None, keepdims=True, name=None)
    x = (x - x_mean) / tf.sqrt(x_variance + epsilon)
    return x


def scale(x, v):
    return Lambda(lambda _: _ / v)(x)


def inpainting_attention(primary, carryover, constant=-10):

    def _initialize_bias(const=-5):
        def _(shape, dtype=None):
            assert len(shape) == 3, 'must be a 3D shape'
            x = np.zeros(shape)
            x[:, :, -1] = const
            return x
        return _

    x = concatenate([primary, carryover], axis=-1)
    h = ZeroPadding2D((1, 1))(x)
    lcn = LocallyConnected2D(
        filters=2,
        kernel_size=(3, 3),
        bias_initializer=_initialize_bias(constant)
    )

    h = lcn(h)
    weights = Lambda(channel_softmax)(h)

    channel_sum = Lambda(K.sum, arguments={'axis': -1, 'keepdims': True})

    return channel_sum(multiply([x, weights]))


def energy_error(requested_energy, recieved_energy):
    difference = (recieved_energy - requested_energy) / 10000

    over_energized = K.cast(difference > 0., K.floatx())

    too_high = 100 * K.abs(difference)
    too_low = 10 * K.abs(difference)

    return over_energized * too_high + (1 - over_energized) * too_low


def minibatch_discriminator(x):
    """ Computes minibatch discrimination features from input tensor x"""
    print('x shape:',x.shape)
    diffs = K.expand_dims(x, 3) - \
        K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    print('x shape 1:',(K.expand_dims(x, 3)).shape)
    print('x shape 2:',( K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0) ).shape)
    print('diffs shape:',diffs.shape)
    l1_norm = K.sum(K.abs(diffs), axis=2)
    print('l1_norm shape:',l1_norm.shape)
    print('final shape:',K.sum(K.exp(-l1_norm), axis=2).shape)
    return K.sum(K.exp(-l1_norm), axis=2)

def minibatch_discriminator_v1(x):
    print('x shape:',x.shape)
    features = [x]
    mx = tf.roll(x, shift=[1], axis=[0])
    print('mx shape:',mx.shape)
    mmx = subtract([x, mx])
    features.append(mmx)
    return concatenate(features)
    

def minibatch_output_shape(input_shape):
    """ Computes output shape for a minibatch discrimination layer"""
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    return tuple(shape[:2])


def single_layer_energy(x):
    shape = K.get_variable_shape(x)
    #return K.reshape(K.sum(x, axis=range(1, len(shape))), (-1, 1))
    return K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))

def single_layer_relative_diff(x):
    x0  = x[0]
    x1  = x[1]
    x1  = x1 + 0.001
    return np.divide((x0-x1),x1)

def single_layer_relative(x):
    x0  = x[0]
    x1  = x[1]
    x1  = x1 + 0.001
    return np.divide(x0,x1)

def single_layer_relative_diff_too_big(x):
    x0  = x[0]
    x1  = x[1]
    x1  = x1 + 0.001
    x2  = np.abs(np.divide((x0-x1),x1))
    x3 = x2 > 0.2
    x3 = K.cast(x3,dtype='float32') # transfer to 0 and 1
    return x3

def is_High_Z(x):
    x0  = np.abs(x[:,5:6])
    x1 = x0 > 100
    x2 = K.cast(x1,dtype='float32') # transfer to 0 and 1
    return x2

def secondMom(x):
    shape = K.get_variable_shape(x)
    sum_e = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    sum_i = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    sum_i = 0*sum_i
    #sum_i = K.cast(sum_i>99999,dtype='float32')
    sum_j = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    sum_j = 0*sum_j
    secondM = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    secondM = 0*secondM
    for i in range(11):
        for j in range(11):
            sum_i = sum_i + K.reshape(x[:,i,j,0]*i,(-1,1))
            sum_j = sum_j + K.reshape(x[:,i,j,0]*j,(-1,1))
    cen_i = np.divide(sum_i,sum_e)
    cen_j = np.divide(sum_j,sum_e)
    for i in range(11):
        for j in range(11):
            r2 = (cen_i - i)*(cen_i - i) + (cen_j - j)*(cen_j - j)
            secondM = secondM + r2*x[:,i,j,:]
             
    return secondM


def latMom(x):
    shape = K.get_variable_shape(x)
    sum_e = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    sum_i = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    #sum_i = K.cast(sum_i>99999,dtype='float32')
    sum_i = 0*sum_i
    sum_j = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    sum_j = 0*sum_j
    numer = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    numer = 0*numer
    denom = K.reshape(K.sum(x, axis=list(range(1, len(shape)))), (-1, 1))
    denom = 0*denom
    for i in range(11):
        for j in range(11):
            sum_i = sum_i + K.reshape(x[:,i,j,0]*i,(-1,1))
            sum_j = sum_j + K.reshape(x[:,i,j,0]*j,(-1,1))
    cen_i = np.divide(sum_i,sum_e+1e-3)
    cen_j = np.divide(sum_j,sum_e+1e-3)
    print('sum_i=',sum_i.shape,',sum_j=',sum_j.shape,',lat cen_j=',cen_j.shape)
    for i in range(11):
        for j in range(11):
            if i==5 and j==5:continue
            elif i==5 and j==6:continue
            elif i==5 and j==4:continue
            elif i==4 and j==5:continue
            elif i==6 and j==5:continue
            r2 = (cen_i - i)*(cen_i - i) + (cen_j - j)*(cen_j - j)
            #print('r2=',r2.shape)
            #print('r2*x=',(r2*x[:,i,j,0]).shape)
            numer = numer + r2*x[:,i,j,:]
    denom = ((cen_i - 5)*(cen_i - 5) + (cen_j - 5)*(cen_j - 5))*x[:,5,5,:] +   ((cen_i - 4)*(cen_i - 4) + (cen_j - 5)*(cen_j - 5))*x[:,4,5,:] + ((cen_i - 6)*(cen_i - 6) + (cen_j - 5)*(cen_j - 5))*x[:,6,5,:] + ((cen_i - 5)*(cen_i - 5) + (cen_j - 4)*(cen_j - 4))*x[:,5,4,:] + ((cen_i - 5)*(cen_i - 5) + (cen_j - 6)*(cen_j - 6))*x[:,5,6,:] 
    denom = denom + 0.0001
    print('numer=',numer.shape,',denom=',denom.shape)
    return numer/denom


def single_layer_e1x1(x):
    result = x[:,5:6,5:6,:]
    return single_layer_energy(result)

def single_layer_e3x3(x):
    result = x[:,4:7,4:7,:]
    return single_layer_energy(result)
def single_layer_e5x5(x):
    result = x[:,3:8,3:8,:]
    return single_layer_energy(result)

def single_layer_energy_output_shape(input_shape):
    shape = list(input_shape)
    # assert len(shape) == 3
    return (shape[0], 1)

def mini_batch_mean(x):
    #x1 = np.full((x.shape[0], 1), K.mean(x) ,dtype=np.float32)
    #x1 = np.copy(x)
    #x1[:,] = K.mean(x)
    x1 = x - x + K.mean(x)
    return x1
def mini_batch_std(x):
    #x1 = np.full((x.shape[0], 1), K.std(x) ,dtype=np.float32)
    #x1 = np.copy(x)
    #x1[:,] = K.std(x)
    x1 = x - x + K.std(x)
    return x1

def calculate_energy(x):
    return Lambda(single_layer_energy, single_layer_energy_output_shape)(x)
def calculate_e3x3(x):
    return Lambda(single_layer_e3x3, single_layer_energy_output_shape)(x)
def calculate_e5x5(x):
    return Lambda(single_layer_e5x5, single_layer_energy_output_shape)(x)
def calculate_e1x1(x):
    return Lambda(single_layer_e1x1, single_layer_energy_output_shape)(x)

def calculate_energy_diff(x_e,x_mom):
    return K.abs(x_e - x_mom)

def threshold_indicator(x, thresh):
    return K.cast(x > thresh, K.floatx())


def sparsity_level(x):
    _shape = K.get_variable_shape(x)
    shape = K.shape(x)
    total = K.cast(K.prod(shape[1:]), K.floatx())
    return K.reshape(K.sum(
        K.cast(x > 0.0, K.floatx()), axis=list(range(1, len(_shape)))
    ), (-1, 1)) / total


def sparsity_output_shape(input_shape):
    shape = list(input_shape)
    return (shape[0], 1)


class MyDense2D(Layer):

    def __init__(self, output_dim1, output_dim2, **kwargs):
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        super(MyDense2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel', 
                                      #shape=(input_shape[1], self.output_dim1, self.output_dim2),
                                      shape=(self.output_dim1, input_dim, self.output_dim2),
                                      initializer='uniform',
                                      trainable=True)
        super(MyDense2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        print('x',x.shape)
        print('kernel',self.kernel.shape)
        tmp = K.dot(x, self.kernel)
        print('tmp',tmp.shape)
        return K.dot(x, self.kernel)
        #out = K.reshape(K.dot(x, self.kernel), (-1, self.output_dim1, self.output_dim2))
        #return out
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim1, self.output_dim2)

    def get_config(self):
        config = {
            'output_dim1': self.output_dim1,
            'output_dim2': self.output_dim2
        }
        base_config = super(MyDense2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dense3D(Layer):

    """
    A 3D, trainable, dense tensor product layer
    """

    def __init__(self, first_dim,
                 last_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense3D, self).__init__(**kwargs)
        self.first_dim = first_dim
        self.last_dim = last_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(self.first_dim, input_dim, self.last_dim),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.first_dim, self.last_dim),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, mask=None):
        out = K.reshape(K.dot(inputs, self.kernel), (-1, self.first_dim, self.last_dim))
        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.first_dim, self.last_dim)

    def get_config(self):
        config = {
            'first_dim': self.first_dim,
            'last_dim': self.last_dim,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))
