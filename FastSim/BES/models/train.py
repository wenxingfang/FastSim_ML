#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: train_v1.py
Add pre-trained regression net
description: main training script for [arXiv/1705.02355]
author: Luke de Oliveira (lukedeo@manifold.ai), 
        Michela Paganini (michela.paganini@yale.edu)
"""

from __future__ import print_function

import argparse
from collections import defaultdict
import logging


import h5py
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #do not use GPU
from six.moves import range
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import sys
import yaml
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
import math

if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)

def binary_crossentropy(target, output):
    output = -target * np.log(output) - (1.0 - target) * np.log(1.0 - output)
    return output

def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run CalGAN training. '
        'Sensible defaults come from [arXiv/1511.06434]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nb-epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')

    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')

    parser.add_argument('--latent-size', action='store', type=int, default=32,
                        help='size of random N(0, 1) latent space to sample')

    parser.add_argument('--disc-lr', action='store', type=float, default=2e-5,
                        help='Adam learning rate for discriminator')

    parser.add_argument('--gen-lr', action='store', type=float, default=2e-4,
                        help='Adam learning rate for generator')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    parser.add_argument('--prog-bar', action='store_true',
                        help='Whether or not to use a progress bar')

    parser.add_argument('--no-attn', action='store_true',
                        help='Whether to turn off the layer to layer attn.')

    parser.add_argument('--debug', action='store_true',
                        help='Whether to run debug level logging')

    parser.add_argument('--d-pfx', action='store',
                        default='params_discriminator_epoch_',
                        help='Default prefix for discriminator network weights')

    parser.add_argument('--g-pfx', action='store',
                        default='params_generator_epoch_',
                        help='Default prefix for generator network weights')

    parser.add_argument('--dataset', action='store', type=str,
                        help='yaml file with particles and HDF5 paths (see '
                        'github.com/hep-lbdl/CaloGAN/blob/master/models/'
                        'particles.yaml)')

    parser.add_argument('--reg-model-in', action='store',type=str,
                        default='',
                        help='input of trained reg model')
    parser.add_argument('--reg-weight-in', action='store',type=str,
                        default='',
                        help='input of trained reg weight')

    parser.add_argument('--gen-model-out', action='store',type=str,
                        default='',
                        help='output of trained gen model')
    parser.add_argument('--gen-weight-out', action='store',type=str,
                        default='',
                        help='output of trained gen weight')
    parser.add_argument('--dis-model-out', action='store',type=str,
                        default='',
                        help='output of trained dis model')
    parser.add_argument('--dis-weight-out', action='store',type=str,
                        default='',
                        help='output of trained dis weight')
    parser.add_argument('--comb-model-out', action='store',type=str,
                        default='',
                        help='output of trained combined model')
    parser.add_argument('--comb-weight-out', action='store',type=str,
                        default='',
                        help='output of trained combined weight')
    parser.add_argument('--gen-out', action='store',type=str,
                        default='',
                        help='output of trained gen model')
    parser.add_argument('--comb-out', action='store',type=str,
                        default='',
                        help='output of trained combined model')
    parser.add_argument('--dis-out', action='store',type=str,
                        default='',
                        help='output of dis model')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()

    # delay the imports so running train.py -h doesn't take 5,234,807 years
    import keras.backend as K
    import tensorflow as tf
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
    session = tf.Session(config=session_conf)
    K.set_session(session)
    from keras.layers import (Activation, AveragePooling2D, Dense, Embedding,
                              Flatten, Input, Lambda, UpSampling2D)
    from keras.layers.merge import add, concatenate, multiply
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.utils.generic_utils import Progbar
    #from keras.utils.vis_utils     import plot_model
    K.set_image_dim_ordering('tf')

    from ops import (minibatch_discriminator, minibatch_output_shape, Dense3D,
                     calculate_energy, scale, inpainting_attention)

    from architectures import build_generator_2D, build_discriminator_2D

    # batch, latent size, and whether or not to be verbose with a progress bar

    if parse_args.debug:
        logger.setLevel(logging.DEBUG)

    # set up all the logging stuff
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s'
        '[%(levelname)s]: %(message)s'
    )
    hander = logging.StreamHandler(sys.stdout)
    hander.setFormatter(formatter)
    logger.addHandler(hander)

    nb_epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    latent_size = parse_args.latent_size
    verbose = parse_args.prog_bar
    no_attn = parse_args.no_attn

    disc_lr = parse_args.disc_lr
    gen_lr = parse_args.gen_lr
    adam_beta_1 = parse_args.adam_beta

    reg_model_in = parse_args.reg_model_in
    reg_weight_in = parse_args.reg_weight_in

    logger.debug('parameter configuration:')

    logger.debug('number of epochs = {}'.format(nb_epochs))
    logger.debug('batch size = {}'.format(batch_size))
    logger.debug('latent size = {}'.format(latent_size))
    logger.debug('progress bar enabled = {}'.format(verbose))
    logger.debug('Using attention = {}'.format(no_attn == False))
    logger.debug('discriminator learning rate = {}'.format(disc_lr))
    logger.debug('generator learning rate = {}'.format(gen_lr))
    logger.debug('Adam $\beta_1$ parameter = {}'.format(adam_beta_1))


    import h5py
    d = h5py.File(parse_args.dataset, 'r')
    first   = d['Barrel_Hit'][:]
    second  = d['Hit_Depth'][:]/100
    mc_info = d['MC_info'][:] 
    d.close()
    ###### do normalization ##############
    mc_info[:,1] = (mc_info[:,1])/180  #dtheta
    mc_info[:,2] = (mc_info[:,2])/10   #dphi
    mc_info[:,3] = (mc_info[:,3])/100  #Z
    first, second, mc_info = shuffle(first, second, mc_info, random_state=0)

    sizes = [ first.shape[1], first.shape[2], 1]
    print("first:",first.shape,",second:",second.shape,",mc info:", mc_info.shape)
    print(sizes)

    logger.info('Building discriminator')

    calorimeter = Input(shape=sizes)
    depth_info  = Input(shape=(1,))
    input_Info  = Input(shape=(4,))


    features =build_discriminator_2D(image=calorimeter,  epsilon=0.001)
    print('features:',features.shape)
    energies = calculate_energy(calorimeter)
    print('energies:',energies.shape)
    p = concatenate([features, depth_info, input_Info, energies])
    '''
        well_too_big,
    '''
    
    fake = Dense(1, activation='sigmoid', name='fakereal_output')(p)
    discriminator_outputs = fake
    discriminator_losses = 'binary_crossentropy'

    discriminator = Model([calorimeter, depth_info, input_Info] , discriminator_outputs, name='discriminator')

    discriminator.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
    )

    logger.info('Building generator')

    latent = Input(shape=(latent_size, ), name='z')
    input_info = Input(shape=(4, ), dtype='float32')
    generator_inputs = [latent, input_info]

    h = concatenate([latent, input_info])
    img_layer  = build_generator_2D(h, 20, 20)
    print('img_layer shape:',img_layer.shape)
    
    flat = Flatten()(img_layer)
    depth_layer = Dense(1, name='depth_output')(flat)
    output_info = Lambda(lambda x: x)(input_info) # same as input
    generator_outputs =  [img_layer, depth_layer, output_info]
    generator = Model(generator_inputs, generator_outputs, name='generator')

    generator.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    print('h3')
######### regression part ##########################
    reg_model = load_model(parse_args.reg_model_in, custom_objects={'tf': tf})
    reg_model.trainable = False
    reg_model.name  = 'regression'
    reg_model.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
###################################
    # build combined model
    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    
    
    gen_depth_info = Lambda(lambda x: x/100)(generator(generator_inputs)[1]) # same as input

    combined_outputs = [discriminator( [generator(generator_inputs)[0], gen_depth_info, generator(generator_inputs)[2]]), reg_model([generator(generator_inputs)[0], gen_depth_info]) ]
    print('h31')
    combined_losses = ['binary_crossentropy', 'mae']

    combined = Model(generator_inputs, combined_outputs, name='combined_model')
    print('h4')
    combined.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss=combined_losses
    )

    logger.info('commencing training')
 
    print('total sample:', first.shape[0])
    #disc_outputs_real = np.ones(batch_size)
    disc_outputs_real = np.ones(batch_size)*0.95
    disc_outputs_fake = np.zeros(batch_size)
    loss_weights      = np.ones(batch_size)
    combined_loss_weights      = [np.ones(batch_size), 1*np.ones(batch_size)]

    for epoch in range(nb_epochs):
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(first.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    logger.info('processed {}/{} batches'.format(index + 1, nb_batches))
                elif index % 10 == 0:
                    logger.debug('processed {}/{} batches'.format(index + 1, nb_batches))

            # generate a new batch of noise
            noise = np.random.normal(0, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch = first  [index * batch_size:(index + 1) * batch_size]
            depth_batch = second [index * batch_size:(index + 1) * batch_size]
            info_batch =  mc_info[index * batch_size:(index + 1) * batch_size]

            sampled_mom      = np.random.uniform( 1   , 1.8, (batch_size, 1))
            sampled_dtheta   = np.random.uniform(35   , 145, (batch_size, 1))/180
            sampled_dphi     = np.random.uniform(-4   , 10 , (batch_size, 1))/10
            sampled_Z        = np.random.uniform(-130 ,130 , (batch_size, 1))/100
            sampled_info     = np.concatenate((sampled_mom, sampled_dtheta, sampled_dphi, sampled_Z),axis=-1)

            generator_inputs = [noise, sampled_info]
            generated_images = generator.predict(generator_inputs, verbose=0)

            real_batch_loss = discriminator.train_on_batch(
                [image_batch, depth_batch, info_batch],
                disc_outputs_real,
                loss_weights
            )
            #print('real_batch_loss=',real_batch_loss)
            fake_batch_loss = discriminator.train_on_batch(
                [generated_images[0], generated_images[1]/100, generated_images[2] ], 
                disc_outputs_fake,
                loss_weights         
            )
            '''
            print('fake_batch_loss=',fake_batch_loss)
            '''
            if index == (nb_batches-1):
                real_pred = discriminator.predict_on_batch([image_batch, depth_batch, info_batch])
                fake_pred = discriminator.predict_on_batch([generated_images[0], generated_images[1]/100, generated_images[2]])
                print('real_pred:\n',real_pred)
                print('fake_pred:\n',fake_pred)
                print('binary_crossentropy real\n:', binary_crossentropy(disc_outputs_real, real_pred))
                print('binary_crossentropy fake\n:', binary_crossentropy(disc_outputs_fake, fake_pred))

            epoch_disc_loss.append(
                (np.array(fake_batch_loss) + np.array(real_batch_loss)) / 2)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, real} labels to say

            gen_losses = []

            # we do this twice simply to match the number of batches per epoch used to
            # train the discriminator
            for _ in range(2):
                noise            = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_mom      = np.random.uniform( 1   , 1.8, (batch_size, 1))
                sampled_dtheta   = np.random.uniform(35   , 145, (batch_size, 1))/180
                sampled_dphi     = np.random.uniform(-4   , 10 , (batch_size, 1))/10
                sampled_Z        = np.random.uniform(-130 ,130 , (batch_size, 1))/100
                sampled_info     = np.concatenate((sampled_mom, sampled_dtheta, sampled_dphi, sampled_Z),axis=-1)

                combined_inputs  = [noise, sampled_info]
                #combined_outputs = [np.ones(batch_size), sampled_info]
                combined_outputs = [np.ones(batch_size)*0.95, sampled_info]
                gen_losses.append(combined.train_on_batch(
                    combined_inputs,
                    combined_outputs,
                    combined_loss_weights
                ))

            epoch_gen_loss.append(np.mean(np.array(gen_losses), axis=0))

        logger.info('Epoch {:3d} Generator loss: {}'.format(
            epoch + 1, np.mean(epoch_gen_loss, axis=0)))
        logger.info('Epoch {:3d} Discriminator loss: {}'.format(
            epoch + 1, np.mean(epoch_disc_loss, axis=0)))

    generator.save(parse_args.gen_out)
    discriminator.save(parse_args.dis_out)
    combined .save(parse_args.comb_out)
    print('done reg training')
