#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: architectures.py
description: sub-architectures for [arXiv/1705.02355]
author: Luke de Oliveira (lukedeo@manifold.ai)
"""
from keras.layers.merge import _Merge
import keras.backend as K
from keras.initializers import constant
from keras.layers import (Dense, Reshape, Conv2D, Conv3D, LeakyReLU, BatchNormalization, UpSampling2D, UpSampling3D, Cropping2D, LocallyConnected2D, Activation, ZeroPadding2D, Dropout, Lambda, Flatten, AveragePooling2D, ReLU, Cropping3D, MaxPooling2D, Conv2DTranspose)
from keras.layers.merge import concatenate, multiply, add, subtract
import numpy as np
import tensorflow as tf
import sys
from ops import (minibatch_discriminator, minibatch_output_shape,calculate_energy,calculate_e3x3,calculate_e5x5, single_layer_relative,single_layer_relative_diff_too_big,
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

def build_generator_2D(x, nb_rows, nb_cols):
    x = Dense(20*5*5)(x)
    x = Reshape((5, 5, 20))(x)
    x = UpSampling2D([4, 4])(x)
    x = Conv2D(8, (5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(6, (4, 4), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(6, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(1, (2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    #x = Cropping3D(cropping=((0, 0), (0, 0), (0, 1)))(x)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    #assert x.shape[3] == nb_channels
    return x

def build_generator_2D_v1(x, nb_rows, nb_cols):
    x = Dense(20*4*4)(x)
    x = Reshape((4, 4, 20))(x)
    x = UpSampling2D([3, 3])(x)
    x = Conv2D(8, (2, 2), padding='valid', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (4, 4), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(1, (5, 5), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    #x = Cropping3D(cropping=((0, 0), (0, 0), (0, 1)))(x)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    #assert x.shape[3] == nb_channels
    return x

def build_generator_2D_v2(x, nb_rows, nb_cols):
    x = Dense(16*4*4)(x)
    x = Reshape((4, 4, 16))(x)
    #x = Conv2DTranspose(filters=16, kernel_size=(3,3), strides=(3, 3), padding='valid', data_format='channels_last')(x)
    #print('Conv2DTranspose=',x.shape)
    x = UpSampling2D([3, 3])(x)
    x = Conv2D(16, (4, 4), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(1, (2, 2), padding='valid', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    return x

def build_generator_2D_v2_1(x, nb_rows, nb_cols):
    x = Dense(16*12*12)(x)
    x = Reshape((12, 12, 16))(x)
    x = Conv2D(16, (4, 4), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(1, (2, 2), padding='valid', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    return x

def build_generator_2D_v2_2(x, nb_rows, nb_cols):
    x = Dense(16*4*4)(x)
    x = Reshape((4, 4, 16))(x)
    #we will set ‘padding‘ to ‘same’ to ensure the output dimensions are 12×12 ( strides * input shape) as required.
    x = Conv2DTranspose(16, (3,3), strides=(3,3), padding='same')(x)    
    print('Conv2DTranspose=',x.shape)
    #x = UpSampling2D([3, 3])(x)
    x = Conv2D(16, (4, 4), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(1, (2, 2), padding='valid', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    #assert x.shape[1] == nb_rows
    #assert x.shape[2] == nb_cols
    return x

def build_generator_2D_v3(x, nb_rows, nb_cols):
    x = Dense(16*4*4)(x)
    x = Reshape((4, 4, 16))(x)
    x = UpSampling2D([3, 3])(x)
    x = Conv2D(16, (4, 4), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(8, (3, 3), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(1, (2, 2), padding='same', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    x = Conv2D(1, (2, 2), padding='valid', kernel_initializer='glorot_uniform')(x)
    x = ReLU()(x)
    print('gg:',x.shape)
    #x = Cropping3D(cropping=((0, 0), (0, 0), (0, 1)))(x)
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    #assert x.shape[3] == nb_channels
    return x


def build_regression(image, epsilon):
    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv2D(8 , (2, 2), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Conv2D(16, (2, 2), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (4, 4), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(400, activation='tanh')(x)
    return x

def build_regression_v1(image, epsilon):
    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(8,  (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(8, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(8, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling2D()(x)
    x = Flatten()(x)

#    x = Dense(16, activation='tanh')(x)
#    x = Dense(8, activation='tanh')(x)
#    x = Dense(3)(x)
    return x

def build_regression_v2(image, epsilon):
    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv2D(64, (2, 2), padding='valid')(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (2, 2), padding='valid')(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(10 , activation='relu')(x)
    return x

def build_regression_v3(image, epsilon):
    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv2D(64, (2, 2), padding='valid')(x)
    x = ReLU()(x)
    x = Conv2D(64, (2, 2), strides=(2, 2),padding='valid')(x)
    x = ReLU()(x)
    x = Conv2D(128, (2, 2), padding='valid')(x)
    x = ReLU()(x)
    x = Conv2D(128, (2, 2), strides=(2, 2),padding='valid')(x)
    x = ReLU()(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84 , activation='relu')(x)
    x = Dense(10 , activation='relu')(x)
    return x
 
def build_regression_v4(image, epsilon):
    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv2D(64, (2, 2), padding='valid')(x)
    x = ReLU()(x)
    x = Conv2D(64, (2, 2), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(64, (2, 2), strides=(2, 2),padding='valid')(x)
    x = Conv2D(128, (2, 2), padding='valid')(x)
    x = ReLU()(x)
    x = Conv2D(128, (2, 2), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(128, (2, 2), strides=(2, 2),padding='valid')(x)
    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(100 , activation='relu')(x)
    x = Dense(10 , activation='relu')(x)
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

def build_discriminator_2D(image, epsilon):

    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv2D(16, (2, 2) , padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(8, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(8, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Conv2D(8, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    K_x = MyDense2D(10, 10)(x)
    print('K_x',K_x.shape)
    minibatch_featurizer = Lambda(minibatch_discriminator)(K_x)
    print('minibatch_featurizer',minibatch_featurizer.shape)
    x = concatenate( [x, minibatch_featurizer] )
    #sys.exit()
    return x

def build_discriminator_2D_v1(image, epsilon):

    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    #x1 = Conv2D(1, (12, 1), padding='valid')(x)
    #x1 = Flatten()(x1)
    #x2 = Conv2D(8, (1, 16), padding='valid')(x)
    #x2 = Flatten()(x2)
    x = Conv2D(16, (2, 2) , padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    #x = MaxPooling2D()(x)
    x = Conv2D(64, (2, 2),strides=(2, 2), padding='valid')(x)
    x = Flatten()(x)
    K_x = MyDense2D(10, 10)(x)
    print('K_x',K_x.shape)
    minibatch_featurizer = Lambda(minibatch_discriminator)(K_x)
    print('minibatch_featurizer',minibatch_featurizer.shape)
    x = concatenate( [x, minibatch_featurizer] )
    return x

def build_discriminator_2D_v2(image, info, epsilon):

    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)
    x = Conv2D(16, (2, 2) , padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    K_x = MyDense2D(10, 10)(x)
    print('K_x',K_x.shape)
    minibatch_featurizer = Lambda(minibatch_discriminator)(K_x)
    print('minibatch_featurizer',minibatch_featurizer.shape)
    x = concatenate( [x, minibatch_featurizer] )
    #################################################
    mom_info = Lambda(lambda x:x[:,0:1], name='mom_layer')(info)
    print('mom:',mom_info.shape)
    energies = calculate_energy(image)
    print('energies:',energies.shape)
    energies_3x3 = calculate_e3x3(image)
    print('energies 3x3:',energies_3x3.shape)
    energies_5x5 = calculate_e5x5(image)
    print('energies 5x5:',energies_5x5.shape)
    sparsity = Lambda(sparsity_level)(image)
    print('sparsity:',sparsity.shape)
    #p0 = concatenate([features, input_Info, sparsity])
    #p1 = Dense(1, name='features_output')(p0)
    #en_diff = Lambda(calculate_energy_diff,  arguments={'x_mom':mom_info})(energies)
    en_diff = subtract([energies, mom_info])
    print('en_diff:', en_diff.shape)
    en_ratio = Lambda(single_layer_relative)([energies,mom_info])
    print('en_ratio:', en_ratio.shape)
    too_big = Lambda(single_layer_relative_diff_too_big)([energies,mom_info])
    print('too_big:',too_big.shape)
    #en_mean = Lambda(mini_batch_mean)(energies)
    #print('en_mean:',en_mean.shape)
    #en_std = Lambda(mini_batch_std)(energies)
    #print('en_std:',en_std.shape)
    #en_diff_mean = Lambda(mini_batch_mean)(en_diff)
    #print('en_diff_mean:',en_diff_mean.shape)
    #en_diff_std  = Lambda(mini_batch_std)(en_diff)
    #print('en_diff_std:',en_diff_std.shape)
    #ref_diff_mean = Lambda(mini_batch_mean)(en_ratio)
    #print('ref_diff_mean:',ref_diff_mean.shape)
    #ref_diff_std  = Lambda(mini_batch_std)(en_ratio)
    #print('ref_diff_std:',ref_diff_std.shape)

    en_info  = concatenate([sparsity, mom_info, energies, energies_3x3, energies_5x5, en_diff, en_ratio, too_big])
    #p_e_1 = Dense(10, name='energy_features1')(en_info)
    #p_e_2 = Dense(10, activation='relu', name='energy_features2')(p_e_1)
    #p_e_3 = Dense(1, name='energy_features3')(p_e_2)
    x  = concatenate([x, en_info])
    x  = Dense(1, name='fakereal_output')(x)

    return x

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        BATCH_SIZE = 512
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class RandomWeightedAverage1(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self, inputs):
        BATCH_SIZE = 512
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)

def LS_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def binary_loss(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    print('Making gradients')
    gradients = K.gradients(y_pred, averaged_samples)
    print('len gradients=',len(gradients),',gradients=',gradients)
    gradients  = gradients[0]
    gradients1 = gradients[1]
    print('gradients=',gradients.shape)
    # compute the euclidean norm by squaring ...
    gradients_sqr  = K.square(gradients)
    gradients_sqr1 = K.square(gradients1)
    #   ... summing over the rows ...
    gradients_sqr_sum  = K.sum(gradients_sqr , axis=np.arange(1, len(gradients_sqr.shape)))
    gradients_sqr_sum1 = K.sum(gradients_sqr1, axis=np.arange(1, len(gradients_sqr1.shape)))
    #   ... and sqrt
    gradient_l2_norm  = K.sqrt(gradients_sqr_sum)
    gradient_l2_norm1 = K.sqrt(gradients_sqr_sum1)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty  = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    gradient_penalty1 = gradient_penalty_weight * K.square(1 - gradient_l2_norm1)
    # return the mean as loss over all the batch samples
    #print('gradient_penalty=',gradient_penalty.shape)
    #tmp = K.mean(gradient_penalty)
    #print('K.mean(gradient_penalty)=',tmp.shape)
    #return K.mean(gradient_penalty)
    return K.mean(gradient_penalty) + K.mean(gradient_penalty1)

def Discriminator_Regularizer(y_true, y_pred, real_samples, fake_out, fake_samples, weight):
    BATCH_SIZE = 512 
    D1 = tf.nn.sigmoid(y_pred)
    D2 = tf.nn.sigmoid(fake_out)
    D1 = tf.reshape(D1, [BATCH_SIZE,1])
    D2 = tf.reshape(D2, [BATCH_SIZE,1])
    print('Making gradients')
    grad_D1_logits = K.gradients(y_pred, real_samples)
    print('len grad_D1_logits=',len(grad_D1_logits),',grad_D1_logits=',grad_D1_logits)
    grad_D1_logits_0  = grad_D1_logits[0]
    grad_D1_logits_1  = grad_D1_logits[1]
    print('grad_D1_logits_0=',grad_D1_logits_0.shape, ',grad_D1_logits_1=',grad_D1_logits_1.shape)
    grad_D2_logits    = K.gradients(fake_out, fake_samples)
    grad_D2_logits_0  = grad_D2_logits[0]
    grad_D2_logits_1  = grad_D2_logits[1]
   
    grad_D1_logits_0 = tf.reshape(grad_D1_logits_0, [BATCH_SIZE,-1])
    grad_D1_logits_1 = tf.reshape(grad_D1_logits_1, [BATCH_SIZE,-1])
    grad_D2_logits_0 = tf.reshape(grad_D2_logits_0, [BATCH_SIZE,-1])
    grad_D2_logits_1 = tf.reshape(grad_D2_logits_1, [BATCH_SIZE,-1])
    grad_D1_logits = tf.keras.backend.concatenate((grad_D1_logits_0, grad_D1_logits_1), axis=-1 )
    grad_D2_logits = tf.keras.backend.concatenate((grad_D2_logits_0, grad_D2_logits_1), axis=-1 )

    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [BATCH_SIZE,-1]), axis=1, keep_dims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [BATCH_SIZE,-1]), axis=1, keep_dims=True)
    #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
    print('grad_D1_logits_norm=',grad_D1_logits_norm.shape, ',D1=', D1.shape)
    assert grad_D1_logits_norm.shape == D1.shape
    assert grad_D2_logits_norm.shape == D2.shape
    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2)    , tf.square(grad_D2_logits_norm))
    # see https://arxiv.org/pdf/1705.09367.pdf in page 5 for Regularized Jensen-Shannon GAN
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return weight*disc_regularizer



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
