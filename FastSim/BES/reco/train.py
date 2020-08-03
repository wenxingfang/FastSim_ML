#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: train.py
description: main training script for [arXiv/1705.02355]
author: Luke de Oliveira (lukedeo@manifold.ai), 
        Michela Paganini (michela.paganini@yale.edu)
"""

from __future__ import print_function

import argparse
from collections import defaultdict
import logging


import math
import h5py
import numpy as np
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" #do not use GPU
from six.moves import range
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import sys
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/models/')
import yaml


if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)


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

    parser.add_argument('--str-out', action='store',type=str,
                        default='',
                        help='output of trained structure')
    parser.add_argument('--weight-out', action='store',type=str,
                        default='',
                        help='output of trained weight')
    parser.add_argument('--model-out', action='store',type=str,
                        default='',
                        help='output of trained model')


    parser.add_argument('--datafile', action='store', type=str,
                        help='yaml file with particles and HDF5 paths (see '
                        'github.com/hep-lbdl/CaloGAN/blob/master/models/'
                        'particles.yaml)')

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

    from architectures import build_generator_3D, build_discriminator_3D, build_regression, build_regression_v1

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
    d = h5py.File(parse_args.datafile, 'r')
    first   = d['Barrel_Hit'][:]
    second  = d['Hit_Depth'][:]/100
    mc_info = d['MC_info'][:] 
    print("first:",first.shape,",second:", second.shape,",mc info:", mc_info.shape)
    ###### do normalization ##############
    mc_info[:,1] = (mc_info[:,1])/180  #dtheta
    mc_info[:,2] = (mc_info[:,2])/10   #dphi
    mc_info[:,3] = (mc_info[:,3])/100  #Z
    d.close()

    sizes = [ first.shape[1], first.shape[2], 1]
    print(sizes)

    first, second, mc_info = shuffle(first, second, mc_info, random_state=0)

    logger.info('Building regression')

    calorimeter = Input(shape=sizes)
    depth_info  = Input(shape=(1,))

    features = build_regression_v1(image=calorimeter, epsilon=0.001)
    #features = build_regression(image=calorimeter, epsilon=0.001)
    print('features:',features.shape)
    energies = calculate_energy(calorimeter)
    print('energies:',energies.shape)

    p = concatenate([features, energies, depth_info])
    
    reg = Dense(4, name='reg_output')(p)
    regression_outputs = reg
    regression_losses = 'mae'

    regression = Model([calorimeter, depth_info], regression_outputs)

    regression.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=regression_losses
    )


    logger.info('commencing training')
 
    print('total sample:', first.shape[0])
    for epoch in range(nb_epochs):
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(first.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_loss = []

        for index in range(nb_batches):
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    logger.info('processed {}/{} batches'.format(index + 1, nb_batches))
                elif index % 10 == 0:
                    logger.debug('processed {}/{} batches'.format(index + 1, nb_batches))


            image_batch = first  [index * batch_size:(index + 1) * batch_size]
            depth_batch = second [index * batch_size:(index + 1) * batch_size]
            info_batch  = mc_info[index * batch_size:(index + 1) * batch_size]


            reg_outputs_real = info_batch

            loss_weights = np.ones(batch_size)

            real_batch_loss = regression.train_on_batch(
                [image_batch,depth_batch],
                reg_outputs_real,
                loss_weights
            )
            if math.isnan(real_batch_loss): continue
            epoch_loss.append(np.array(real_batch_loss))

        logger.info('Epoch {:3d}  loss: {}'.format( epoch + 1, np.mean(epoch_loss, axis=0)))

        # save weights every epoch
    #regression.save_weights(parse_args.weight_out, overwrite=True)
    #yaml_string = regression.to_yaml()
    #open(parse_args.str_out, 'w').write(yaml_string)
    regression.save(parse_args.model_out)
    print('done')

