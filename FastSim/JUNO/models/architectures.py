#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: architectures.py
description: sub-architectures for [arXiv/1705.02355]
author: Luke de Oliveira (lukedeo@manifold.ai)
"""
import keras
import keras.backend as K
from keras.initializers import constant
from keras.layers import (Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, Cropping2D, LocallyConnected2D, Activation, ZeroPadding2D, Dropout, Lambda, Flatten, MaxPooling2D, ReLU, AveragePooling2D, Conv2DTranspose, ZeroPadding1D, Cropping1D)
from keras.layers.merge import concatenate, multiply, add
import numpy as np
import tensorflow as tf

from ops import (minibatch_discriminator, minibatch_output_shape,
                 Dense3D, sparsity_level, sparsity_output_shape, scale, normalize, map_fn, MyDense2D)


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

def build_generator_v2(x, nb_rows, nb_cols, template0, template1, s_value):
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
    x = scale(x, s_value) 
    x = multiply([x, template1])
    x = add     ([x, template0])
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    return x

def build_generator_v3(x, nb_rows, nb_cols, template0, template1, s_value):
    x = Dense(256*12*6)(x)
    x = Reshape((6, 12, 256))(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, (2, 2), padding='same')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    print('g1:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(64, (2, 2), padding='same')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    print('g2:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(32, (2, 2), padding='same')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    print('g3:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(16, (2, 2), padding='same')(x)
#    x = BatchNormalization()(x)
    x = ReLU()(x)
    print('g4:',x.shape)
    x = UpSampling2D()(x)
    x = Conv2D(1, (2, 2), padding='same')(x)
    #x = BatchNormalization()(x)
    #x = LeakyReLU()(x)
    print('g5:',x.shape)
    x = Cropping2D(cropping=((6, 6), (12, 12)))(x)
    print('gg:',x.shape)
    x = scale(x, s_value) 
    x = multiply([x, template1])
    x = add     ([x, template0])
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
    return x

def build_generator_v4(image, nb_rows, nb_cols, template0, template1, s_value):
    x = Lambda(lambda x: 1/(abs(x)+0.01))(image)
    x = Conv2D(128, (2, 2), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(64, (2, 2), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(32, (2, 2), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(16, (2, 2), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(1, (2, 2), padding='same')(x)
    x = scale(x, s_value) 
    x = multiply([x, template1])
    x = add     ([x, template0])
    assert x.shape[1] == nb_rows
    assert x.shape[2] == nb_cols
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


def build_discriminator_v1(image):
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
    x = concatenate([image[0], image[1]])
    x = Conv2D(16, (2, 2), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Conv2D(32, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (4, 4), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (5, 5), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    return x

def build_discriminator_v2(image, info):
    
    print('image 0:',image[0].shape)
    print('image 1:',image[1].shape)
    print('info   :',info.shape)
    x0 = Lambda(map_fn)([image[0],info])
    #x0 = map_fn((image[0], info))
    print('x0:',x0.shape)
    x1 = Lambda(map_fn)([image[1],info])
    #x1 = map_fn((image[1], info))
    print('x1:',x1.shape)
    x = concatenate([x0, x1])
    x = Conv2D(16, (2, 2), padding='valid')(x)
    x = LeakyReLU()(x)
    x = Conv2D(32, (3, 3), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (4, 4), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (5, 5), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    return x

def build_discriminator_v3(image, epsilon):
    
    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)

    x = Conv2D(32, (8, 8), strides=(1,1), padding='same')(x) # 124, 360
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (8, 8), strides=(2,2), padding='same')(x) # 62, 180
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (7, 7), strides=(1,1), padding='same')(x) 
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (7, 7), strides=(2,2), padding='same')(x) # 31, 90
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (6, 6), strides=(1,1), padding='same')(x) 
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (6, 6), strides=(2,2), padding='same')(x) # 16, 45
    print('Dis v3 x=',x.shape)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(256, (5, 5), strides=(1,1), padding='same')(x) 
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(256, (5, 5), strides=(2,2), padding='same')(x) # 8, 23
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(512, (4, 5), strides=(1,1), padding='same')(x) 
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(512, (4, 5), strides=(2,2), padding='same')(x) # 4, 12
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(1024, (2, 3), strides=(1,1), padding='same')(x) 
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(1024, (2, 3), strides=(2,2), padding='same')(x) # 2, 6
    x = LeakyReLU()(x)
    #x = Conv2D(2048, (2, 2), strides=(1,2), padding='same')(x) # 2, 3
    #x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    K_x = MyDense2D(10, 10)(x)
    print('K_x',K_x.shape)
    minibatch_featurizer = Lambda(minibatch_discriminator)(K_x)
    print('minibatch_featurizer',minibatch_featurizer.shape)
    x = concatenate( [x, minibatch_featurizer] )
    return x

def build_discriminator_v4(image, epsilon):
    
    x = Lambda(normalize, arguments={'epsilon':epsilon})(image)

    x = Conv2D(64, (8, 8), strides=(2,2), padding='same')(x) # 62, 180
    x = LeakyReLU()(x)
    x = Conv2D(128, (7, 7), strides=(2,2), padding='same')(x) # 31, 90
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, (6, 6), strides=(2,2), padding='same')(x) # 16, 45
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, (5, 5), strides=(2,2), padding='same')(x) # 8, 23
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(1024, (4, 4), strides=(2,2), padding='same')(x) # 4, 12
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)
    K_x = MyDense2D(10, 10)(x)
    print('K_x',K_x.shape)
    minibatch_featurizer = Lambda(minibatch_discriminator)(K_x)
    print('minibatch_featurizer',minibatch_featurizer.shape)
    x = concatenate( [x, minibatch_featurizer] )
    return x

def build_generator_v5(x, nb_rows, nb_cols):
    x = Dense(1024*2*6)(x)
    x = Reshape((2, 6, 1024))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    #we will set ‘padding‘ to ‘same’ to ensure the output dimensions are 12×12 ( strides * input shape) as required.
    x = Conv2DTranspose(512, (8,8), strides=(2,2), padding='same')(x) # 4 , 12
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(256, (8,8), strides=(2,2), padding='same')(x) # 8, 24
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(128, (6,6), strides=(2,2), padding='same')(x) # 16, 48
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(64, (6,6), strides=(2,2), padding='same')(x) # 32, 96
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(x) # 64, 192
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Cropping2D(cropping=((0, 0), (6, 6)))(x)                    # 64 , 180
    x = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same')(x) # 128, 360
    x = ReLU()(x)
    x = Cropping2D(cropping=((2, 2), (0, 0)))(x)                    # 124, 360
    print('gen v5 x=',x.shape) 
    return x

def build_generator_v6(x, nb_rows, nb_cols):
    x = Dense(1024*2*6)(x)
    x = Reshape((2, 6, 1024))(x)
    #we will set ‘padding‘ to ‘same’ to ensure the output dimensions are 12×12 ( strides * input shape) as required.
    x = Conv2DTranspose(512, (8,8), strides=(2,2), padding='same')(x) # 4 , 12
    x = ReLU()(x)
    x = Conv2DTranspose(256, (8,8), strides=(2,2), padding='same')(x) # 8, 24
    x = ReLU()(x)
    x = Conv2DTranspose(128, (6,6), strides=(2,2), padding='same')(x) # 16, 48
    x = ReLU()(x)
    x = Conv2DTranspose(64, (6,6), strides=(2,2), padding='same')(x) # 32, 96
    x = ReLU()(x)
    x = Conv2DTranspose(32, (4,4), strides=(2,2), padding='same')(x) # 64, 192
    x = ReLU()(x)
    x = Cropping2D(cropping=((0, 0), (6, 6)))(x)                    # 64 , 180
    x = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same')(x) # 128, 360
    x = ReLU()(x)
    x = Cropping2D(cropping=((2, 2), (0, 0)))(x)                    # 124, 360
    print('gen v5 x=',x.shape) 
    return x

def build_generator_v7(x):
    x = Dense(2048,kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None))(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = Dense(8192,kernel_initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=None))(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    #x = Dense(47486,kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None), kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x = Dense(47486,kernel_initializer=keras.initializers.RandomUniform(minval=-0.001, maxval=0.001, seed=None))(x)
    x = ReLU()(x)
    return x

def build_generator_v8(x):
    x = Dense(7*7*512,kernel_initializer=keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None))(x)
    x = Reshape((7, 7, 512))(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same')(x) # 14 , 14
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2DTranspose(128, (6,6), strides=(2,2), padding='same')(x) # 28 , 28
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2DTranspose(64, (8,8), strides=(2,2), padding='same')(x) # 56 , 56
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = Cropping2D(cropping=((0, 1), (0, 1)))(x)                   # 55, 55
    x = Conv2DTranspose(32, (10,10), strides=(2,2), padding='same')(x) # 110,110
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2DTranspose(1, (10,10), strides=(2,2), padding='same')(x) # 220,220
    x = ReLU()(x)
    x = Dropout(0.1)(x)
    x = Cropping2D(cropping=((1, 1), (1, 1)))(x)                   # 218,218
    x = ReLU()(x)
    x = Reshape((-1, 1))(x)                                        # 47524
    x = Cropping1D(cropping=(0, 38))(x)                            # 47486
    x = Flatten()(x)                                               # 47486
    print('gen v8 x=',x.shape)
    return x


def build_generator_v9(x, hid_n ,out_n):
    x = Dense(hid_n, activation='tanh')(x)
    x = Dense(hid_n, activation='tanh')(x)
    x = Dense(hid_n, activation='tanh')(x)
    x = Dense(out_n, activation='relu')(x)
    return x


def build_discriminator_v5(x, epsilon):
    x = Reshape((x.shape[1], 1))(x)
    x = ZeroPadding1D((0,38))(x)
    print('dis v5 x=',x.shape)
    x = Reshape((218, 218, 1))(x)
    x = Lambda(normalize, arguments={'epsilon':epsilon})(x)
    x = Conv2D(64, (8, 8), strides=(2,2), padding='same')(x) # 109
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (7, 7), strides=(2,2), padding='same')(x) # 55
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(256, (6, 6), strides=(2,2), padding='same')(x) # 28
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(512, (5, 5), strides=(2,2), padding='same')(x) # 14
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(1024, (4, 4), strides=(2,2), padding='same')(x) #7
    #x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    K_x = MyDense2D(10, 10)(x)
    print('K_x',K_x.shape)
    minibatch_featurizer = Lambda(minibatch_discriminator)(K_x)
    print('minibatch_featurizer',minibatch_featurizer.shape)
    x = concatenate( [x, minibatch_featurizer] )
    return x

def build_discriminator_v6(x, epsilon):
    x = Dense(12, activation='tanh')(x)
    x = Dense(12, activation='tanh')(x)
    x = Dense(12, activation='tanh')(x)
    #x = Flatten()(x)
    K_x = MyDense2D(5, 5)(x)
    minibatch_featurizer = Lambda(minibatch_discriminator)(K_x)
    x = concatenate( [x, minibatch_featurizer] )
    return x

def build_regression(image):### much better performance for normalized input
    x = Conv2D(64, (3, 3))(image)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D((1,2))(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(512, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = ReLU()(x)
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    #x = Dense(6, 'softmax')(x)
    return x 

def build_regression_v1(image):### try with BatchNormalization()
    x = Conv2D(64, (3, 3))(image)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((1,2))(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = ReLU()(x)
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    #x = Dense(6, 'softmax')(x)
    return x 

def build_regression_v2(image):### 
    x = Conv2D(16, (6, 6), padding='same')(image)
    x = LeakyReLU()(x)
    x = Conv2D(8, (6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(8, (6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(8, (6, 6), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(16, activation='tanh')(x)
    x = Dropout(0.1)(x)
    x = Dense(8, activation='tanh')(x)

    return x 

def build_regression_v3(images, epsilon):### try normalization 
    #x_mean, x_variance = tf.nn.moments(image, axes=[1,2,3], shift=None, keepdims=True, name=None)
    #x = (image - x_mean) / tf.sqrt(x_variance + epsilon)
    x0 = Lambda(normalize, arguments={'epsilon':epsilon})(images[0])
    x1 = Lambda(normalize, arguments={'epsilon':epsilon})(images[1])
    x = concatenate([x0,x1])
    print('x shape',x.shape)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(64, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D((1,2))(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(128, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(256, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(512, (3, 3))(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3))(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = ReLU()(x)
    x = Dense(512)(x)
    x = ReLU()(x)
    x = Dense(256)(x)
    x = ReLU()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    return x 
