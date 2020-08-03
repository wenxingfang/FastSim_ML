#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: architectures.py
description: sub-architectures for [arXiv/1705.02355]
author: Luke de Oliveira (lukedeo@manifold.ai)
"""

import keras.backend as K
from keras.initializers import constant
from keras.layers import (Dense, Reshape, Conv2D, Conv3D, LeakyReLU, BatchNormalization, UpSampling2D, UpSampling3D, Cropping2D, LocallyConnected2D, Activation, ZeroPadding2D, Dropout, Lambda, Flatten, AveragePooling3D, ReLU, Cropping3D, ZeroPadding3D, Conv3DTranspose)
from keras.layers.merge import concatenate, multiply, add, subtract
import numpy as np
import tensorflow as tf

from ops import (minibatch_discriminator, minibatch_output_shape,
                 Dense3D, sparsity_level, sparsity_output_shape, minibatch_discriminator_v1, normalize, MyDense2D)


def sparse_softmax(x):
    x = K.relu(x)
    e = K.exp(x - K.max(x, axis=(1, 2, 3), keepdims=True))
    s = K.sum(e, axis=(1, 2, 3), keepdims=True)
    return e / s


def build_generator(x, nb_rows, nb_cols):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        x: a keras Input with shape (None, latent_dim)
        nb_rows: int, number of desired output rows
        nb_cols: int, number of desired output cols

    Returns:
    --------
        a keras tensor with the transformation applied
    """

    x = Dense((nb_rows + 2) * (nb_cols + 2) * 16)(x)
    x = Reshape((nb_rows + 2, nb_cols + 2, 16))(x)
    x = Conv2D(8, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(4, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    '''
    x = LocallyConnected2D(6, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    print('g3')

    x = LocallyConnected2D(1, (2, 2), use_bias=False,
                           kernel_initializer='glorot_normal')(x)
    print('g4')
    '''
    return x


def build_generator_v1(x, nb_rows, nb_cols, template0, template1):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        x: a keras Input with shape (None, latent_dim)
        nb_rows: int, number of desired output rows
        nb_cols: int, number of desired output cols

    Returns:
    --------
        a keras tensor with the transformation applied
    """
    '''
    x = Dense((nb_rows + 2) * (nb_cols + 2) * 16)(x)
    x = Reshape((nb_rows + 2, nb_cols + 2, 16))(x)
    print('g1:',x.shape)
    x = Conv2D(8, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    print('g2:',x.shape)
    x = Conv2D(4, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    print('g3:',x.shape)
    x = Conv2D(1, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    print('g4:',x.shape)
    '''
    x = Dense(256*8*4)(x)
    x = Reshape((4, 8, 256))(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    print('g1:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(64, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    print('g2:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(32, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    print('g3:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(16, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    print('g4:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(1, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU()(x)
    print('g5:',x.shape)
    x = Cropping2D(cropping=((1, 1), (15, 15)))(x)
    print('gg:',x.shape)

    #x = BatchNormalization()(x)
    #print('g4')
    #x = Activation('relu')(x)
    x = multiply([x, template1])
    x = add     ([x, template0])

    '''
    x = LocallyConnected2D(6, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    print('g3')

    x = LocallyConnected2D(1, (2, 2), use_bias=False,
                           kernel_initializer='glorot_normal')(x)
    print('g4')
    '''
    return x

def build_generator_v2(x, nb_rows, nb_cols, template0, template1):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        x: a keras Input with shape (None, latent_dim)
        nb_rows: int, number of desired output rows
        nb_cols: int, number of desired output cols

    Returns:
    --------
        a keras tensor with the transformation applied
    """
    x = Dense(256*12*6)(x)
    x = Reshape((6, 12, 256))(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    print('g1:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(64, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    print('g2:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(32, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    print('g3:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(16, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    print('g4:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(1, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU()(x)
    print('g5:',x.shape)
    x = Cropping2D(cropping=((6, 6), (12, 12)))(x)
    print('gg:',x.shape)
    '''
    def _temp():
        f = open('/hpcfs/juno/junogpu/fangwx/FastSim/data/pmt_id_x_y.txt','r')
        template = []
        for line in f:
            pid,x,y = line.split()
            template.append((x,y))
        f.close()
        return template
    temp = _temp()
    def _modify(x, temp):
        shape = x.shape 
        x_range = shape[1]
        y_range = shape[2]
        for i in range(x_range):
            for j in range(y_range):
                if (i,j) not in temp: x[:,i,j,:] = -1
        return x
    x = Lambda(_modify, arguments={'temp':temp})(x)
    '''
    x = multiply([x, template1])
    x = add     ([x, template0])
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    return x

def build_generator_3D(x, nb_rows, nb_cols, nb_channels):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        x: a keras Input with shape (None, latent_dim)
        nb_rows: int, number of desired output rows
        nb_cols: int, number of desired output cols

    Returns:
    --------
        a keras tensor with the transformation applied
    """
    x = Dense(6*6*6*8)(x)
    x = Reshape((6, 6, 6, 8))(x)
    x = UpSampling3D([5, 5, 5])(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(6, (4, 4, 6), padding='same', kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(6, (3, 3, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(1, (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    x = Cropping3D(cropping=((0, 0), (0, 0), (0, 1)))(x)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    assert x.shape[3] == nb_channels
    return x

def build_generator_3D_v1(x, nb_rows, nb_cols, nb_channels):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        x: a keras Input with shape (None, latent_dim)
        nb_rows: int, number of desired output rows
        nb_cols: int, number of desired output cols

    Returns:
    --------
        a keras tensor with the transformation applied
    """
    #x = Dense(8*8*6*8)(x)
    #x = Reshape((8, 8, 6, 8))(x)
    x = Dense(16*16*6*8)(x)
    x = Reshape((16, 16, 6, 8))(x)
    x = UpSampling3D([2, 2, 5])(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(6, (4, 4, 6), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(6, (3, 3, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(1, (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    x = Cropping3D(cropping=((0, 1), (0, 1), (0, 1)))(x)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    assert x.shape[3] == nb_channels
    return x

def build_generator_3D_v3(x, nb_rows, nb_cols, nb_channels):
    x = Dense(8*8*8*64)(x)
    x = Reshape((8, 8, 8, 64))(x)
    x = Conv3D(64, (7, 7, 7), padding='same', kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU()(x)
    x = Conv3D(64, (7, 7, 7), padding='same', kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU()(x)
    #x = UpSampling3D([2, 2, 2])(x)
    #we will set ‘padding‘ to ‘same’ to ensure the output dimensions are 8×2 ( strides * input shape) as required.
    x = Conv3DTranspose(32, (2,2,2), strides=(2,2,2), padding='same')(x)
    x = Conv3D(32, (5, 5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU()(x)
    x = Conv3D(32, (5, 5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU()(x)
    #x = UpSampling3D([2, 2, 2])(x)
    x = Conv3DTranspose(16, (2,2,2), strides=(2,2,2), padding='same')(x)
    x = Conv3D(16, (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU()(x)
    x = Conv3D(8 , (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU()(x)
    x = Conv3D(4 , (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU()(x)
    x = Conv3D(1 , (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    x = Cropping3D(cropping=((0, 1), (0, 1), (0, 3)))(x)
    #assert x.shape[1] == nb_rows
    #assert x.shape[2] == nb_cols
    #assert x.shape[3] == nb_channels
    return x


def build_generator_3D_v2(x, nb_rows, nb_cols, nb_channels):
    x = Dense(8*8*5*8)(x)
    x = Reshape((8, 8, 5, 8))(x)
    x = Conv3D(8, (3, 3, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(8, (3, 3, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(8, (3, 3, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = UpSampling3D([2, 2, 3])(x)
    x = Conv3D(8, (5, 5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(8, (5, 5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(8, (5, 5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = UpSampling3D([2, 2, 2])(x)
    x = Conv3D(8, (7, 7, 7), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(8, (7, 7, 7), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(6, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(6, (3, 3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(1, (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    x = Cropping3D(cropping=((0, 1), (0, 1), (0, 1)))(x)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    assert x.shape[3] == nb_channels
    return x


def build_generator_3D_EHcal(x, nb_rows_E, nb_cols_E, nb_high_E, nb_rows_H, nb_cols_H, nb_high_H):
    x = Dense(16*16*6*8)(x)
    x = Reshape((16, 16, 6, 8))(x)
    x = UpSampling3D([2, 2, 5])(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(6, (4, 4, 6), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(6, (3, 3, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(1, (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('x shape:',x.shape)
    x = Cropping3D(cropping=((0, 1), (0, 1), (0, 1)))(x)
    assert x.shape[1] == nb_rows_E#31
    assert x.shape[2] == nb_cols_E#31
    assert x.shape[3] == nb_high_E#29
    y = ZeroPadding3D(padding=((5,5), (5,5), (13,13)), data_format=None)(x)
    y = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(y)
    y = ReLU()(y)
    y = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(y)
    y = ReLU()(y)
    y = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(y)
    y = ReLU()(y)
    y = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(y)
    y = ReLU()(y)
    y = Conv3D(6, (4, 4, 6), padding='same', kernel_initializer='glorot_uniform')(y)
    y = ReLU()(y)
    y = Conv3D(6, (3, 3, 5), padding='same', kernel_initializer='glorot_uniform')(y)
    y = ReLU()(y)
    y = Conv3D(1, (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(y)
    y = ReLU()(y)
    print('y shape:',y.shape)
    y = Cropping3D(cropping=((0, 1), (0, 1), (0, 0)))(y)
    assert y.shape[1] == nb_rows_H#40
    assert y.shape[2] == nb_cols_H#40
    assert y.shape[3] == nb_high_H#55
    return x,y



def build_generator_3D_H_v1(x, nb_rows, nb_cols, nb_high):
    #x = Dense(8*8*6*8)(x)
    #x = Reshape((8, 8, 6, 8))(x)
    x = Dense(10*10*11*8)(x)
    x = Reshape((10, 10, 11, 8))(x)
    x = UpSampling3D([4, 4, 5])(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(8, (6, 6, 8), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(6, (4, 4, 6), padding='same', kernel_initializer='glorot_uniform')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3D(6, (3, 3, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv3D(1, (2, 2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    #x = Cropping3D(cropping=((0, 1), (0, 1), (0, 1)))(x)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    assert x.shape[3] == nb_high
    return x


def build_regression(image):
    x = Conv3D(16, (5, 6, 6), padding='same')(image)
    x = LeakyReLU()(x)

    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling3D()(x)
    x = Flatten()(x)

    x = Dense(16, activation='tanh')(x)
    x = Dense(8, activation='tanh')(x)
#    x = Dense(3)(x)
    return x

def build_regression_v1(image, epsilon):
    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv3D(16, (5, 6, 6), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling3D()(x)
    x = Flatten()(x)

#    x = Dense(16, activation='tanh')(x)
#    x = Dense(8, activation='tanh')(x)
#    x = Dense(3)(x)
    return x

def build_discriminator(image, mbd=False, sparsity=False, sparsity_mbd=False):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        image: keras tensor of 4 dimensions (i.e. the output of one calo layer)
        mdb: bool, perform feature level minibatch discrimination
        sparsiry: bool, whether or not to calculate and include sparsity
        sparsity_mdb: bool, perform minibatch discrimination on the sparsity 
            values in a batch

    Returns:
    --------
        a keras tensor of features

    """

    x = Conv2D(16, (2, 2), padding='same')(image)
    x = LeakyReLU()(x)

    #x = ZeroPadding2D((1, 1))(x)
    #x = LocallyConnected2D(16, (3, 3), padding='valid', strides=(1, 2))(x)
    x = Conv2D(8, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    #x = ZeroPadding2D((1, 1))(x)
    #x = LocallyConnected2D(8, (2, 2), padding='valid')(x)
    x = Conv2D(4, (2, 2), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    #x = ZeroPadding2D((1, 1))(x)
    #x = LocallyConnected2D(8, (2, 2), padding='valid', strides=(1, 2))(x)
    #x = Conv2D(8, (2, 2), padding='valid')(x)
    #x = LeakyReLU()(x)
    #x = BatchNormalization()(x)

    x = Flatten()(x)

    if mbd or sparsity or sparsity_mbd:
        minibatch_featurizer = Lambda(minibatch_discriminator,
                                      output_shape=minibatch_output_shape)

        features = [x]

        nb_features = 10
        vspace_dim = 10

        # creates the kernel space for the minibatch discrimination
        if mbd:
            K_x = Dense3D(nb_features, vspace_dim)(x)
            features.append(Activation('tanh')(minibatch_featurizer(K_x)))
            

        if sparsity or sparsity_mbd:
            sparsity_detector = Lambda(sparsity_level, sparsity_output_shape)
            empirical_sparsity = sparsity_detector(image)
            if sparsity:
                features.append(empirical_sparsity)
            if sparsity_mbd:
                K_sparsity = Dense3D(nb_features, vspace_dim)(empirical_sparsity)
                features.append(Activation('tanh')(minibatch_featurizer(K_sparsity)))

        return concatenate(features)
    else:
        return x

def build_discriminator_3D(image, mbd=False, sparsity=False, sparsity_mbd=False):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        image: keras tensor of 4 dimensions (i.e. the output of one calo layer)
        mdb: bool, perform feature level minibatch discrimination
        sparsiry: bool, whether or not to calculate and include sparsity
        sparsity_mdb: bool, perform minibatch discrimination on the sparsity 
            values in a batch

    Returns:
    --------
        a keras tensor of features

    """

    x = Conv3D(16, (5, 6, 6), padding='same')(image)
    x = LeakyReLU()(x)

    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (5, 6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling3D()(x)
    x = Flatten()(x)
    
    if mbd or sparsity or sparsity_mbd:
        #features = [x]
        #mx = tf.roll(x, shift=[1], axis=[0])
        #print('mx shape:',mx.shape)
        #print('x type:',type(x))
        #print('mx type:',type(mx))
        #mmx = subtract([x, mx])
        #print('mmx type:',type(mmx))
        #features.append(mmx)
        #return concatenate(features)
        #return mx 
        minibatch_featurizer = Lambda(minibatch_discriminator_v1)(x)
        return minibatch_featurizer
        
        '''
        minibatch_featurizer = Lambda(minibatch_discriminator,
                                      output_shape=minibatch_output_shape)

        features = [x]

        nb_features = 10
        vspace_dim = 10

        # creates the kernel space for the minibatch discrimination
        if mbd:
            K_x = Dense3D(nb_features, vspace_dim)(x)
            features.append(Activation('tanh')(minibatch_featurizer(K_x)))
            

        if sparsity or sparsity_mbd:
            sparsity_detector = Lambda(sparsity_level, sparsity_output_shape)
            empirical_sparsity = sparsity_detector(image)
            if sparsity:
                features.append(empirical_sparsity)
            if sparsity_mbd:
                K_sparsity = Dense3D(nb_features, vspace_dim)(empirical_sparsity)
                features.append(Activation('tanh')(minibatch_featurizer(K_sparsity)))

        return concatenate(features)
        '''
    else:
        return x

def build_discriminator_3D_v1(image, mbd=False, sparsity=False, sparsity_mbd=False):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        image: keras tensor of 4 dimensions (i.e. the output of one calo layer)
        mdb: bool, perform feature level minibatch discrimination
        sparsiry: bool, whether or not to calculate and include sparsity
        sparsity_mdb: bool, perform minibatch discrimination on the sparsity 
            values in a batch

    Returns:
    --------
        a keras tensor of features

    """

    x = Conv3D(16, (2, 2, 2), padding='same')(image)
    x = LeakyReLU()(x)

    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling3D()(x)
    x = Flatten()(x)
    return x

def build_discriminator_3D_v2(image, epsilon):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        image: keras tensor of 4 dimensions (i.e. the output of one calo layer)
        mdb: bool, perform feature level minibatch discrimination
        sparsiry: bool, whether or not to calculate and include sparsity
        sparsity_mdb: bool, perform minibatch discrimination on the sparsity 
            values in a batch

    Returns:
    --------
        a keras tensor of features

    """

    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv3D(16, (2, 2, 2), padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
#    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
#    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
#    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
#    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling3D()(x)
    x = Flatten()(x)
    return x

def build_discriminator_3D_v3(image, info, epsilon):

    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv3D(16, (2, 2, 2), padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling3D()(x)
    x = Flatten()(x)
    x = concatenate( [x, info] )
    return x

def build_discriminator_3D_v4(image, epsilon):

    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv3D(16, (2, 2, 2), padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling3D()(x)
    x = Flatten()(x)

    K_x = MyDense2D(10, 10)(x)
    print('K_x',K_x.shape)
    minibatch_featurizer = Lambda(minibatch_discriminator)(K_x)
    print('minibatch_featurizer',minibatch_featurizer.shape)
    x = concatenate( [x, minibatch_featurizer] )
    return x




def build_discriminator_3D_H_v0(image, epsilon):

    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv3D(16, (2, 2, 2), padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv3D(8, (3, 3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling3D()(x)
    x = Flatten()(x)
    return x
