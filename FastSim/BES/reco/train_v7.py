#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
"""

from __future__ import print_function

import argparse
from collections import defaultdict
import logging

import ast
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
import h5py
from keras.models import load_model
##########################################################################
#using Mom dtheta, Mom dphi, Pos dz, Pos dphi, array with 121 cell energy#
#try to add more vars and adjust the network 
#add load pre trained model option
#try to use all info to reco mom
##########################################################################


if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)

def load_data(datafile):
    d = h5py.File(datafile, 'r')
    first   = d['Barrel_Hit'][:]
    mc_info = d['MC_info'][:] 
    print("first:",first.shape,"mc info:", mc_info.shape)
    ###### do scale ##############
    mc_info[:,1] = (mc_info[:,1])/5    #M_dtheta
    mc_info[:,2] = (mc_info[:,2])/10   #M_dphi
    mc_info[:,3] = (mc_info[:,3])/2    #P_dz
    mc_info[:,4] = (mc_info[:,4])/2    #P_dphi
    mc_info[:,5] = (mc_info[:,5])/150  #Z
    d.close()
    first, mc_info = shuffle(first, mc_info)
    return first, mc_info

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
                        help='txt file contains h5 file list' )

    parser.add_argument('--load_preTrained_model', action='store', type=ast.literal_eval, default=False,
                        help='load_preTrained_model')
    parser.add_argument('--preTrained_model', action='store', type=str,
                        help='preTrained_model' )
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

    from ops import (minibatch_discriminator, minibatch_output_shape, Dense3D, single_layer_energy ,
                     calculate_energy, scale, inpainting_attention, calculate_e3x3, calculate_e5x5, calculate_e1x1, single_layer_relative)

    from architectures import build_generator_3D, build_discriminator_3D, build_regression, build_regression_v1, build_regression_v2, build_regression_v3, build_regression_v4

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
    load_preTrained_model = parse_args.load_preTrained_model
    preTrained_model = parse_args.preTrained_model

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





    logger.info('Building regression')

    # input image shape
    sizes =[11, 11, 1]
    calorimeter = Input(shape=sizes)

    pos_info = Input(shape=(3,))

    features = build_regression_v4(image=calorimeter, epsilon=0.001)
    #features = build_regression_v3(image=calorimeter, epsilon=0.001)
    #features = build_regression_v2(image=calorimeter, epsilon=0.001)
    #features = build_regression_v1(image=calorimeter, epsilon=0.001)
    #features = build_regression(image=calorimeter, epsilon=0.001)
    print('features:',features.shape)
    energies = calculate_energy(calorimeter)
    print('energies:',energies.shape)
    energies_3x3 = calculate_e3x3(calorimeter)
    print('energies 3x3:',energies_3x3.shape)
    energies_5x5 = calculate_e5x5(calorimeter)
    print('energies 5x5:',energies_5x5.shape)
    energies_1x1 = calculate_e1x1(calorimeter)
    print('energies 1x1:',energies_1x1.shape)
    ratio_e1x1 = Lambda(single_layer_relative)([energies_1x1, energies])
    print('ratio_e1x1:', ratio_e1x1.shape)
    ratio_e3x3 = Lambda(single_layer_relative)([energies_3x3, energies])
    print('ratio_e3x3:', ratio_e3x3.shape)
    ratio_e5x5 = Lambda(single_layer_relative)([energies_5x5, energies])
    print('ratio_e5x5:', ratio_e5x5.shape)

    p_en = concatenate([energies, energies_3x3, energies_5x5, energies_1x1, ratio_e3x3, ratio_e5x5, ratio_e1x1])
    #p_e_1 = Dense(10, name='p_e_1')(p_en)
    #p_e_2 = Dense(10, activation='relu', name='p_e_2')(p_e_1)
    #p_e_3 = Dense(1 , name='p_e_3')(p_e_2)
    
    p1 = concatenate([features, p_en, pos_info])
    p2 = Dense(100, activation='relu', name='p2')(p1)
    p3 = Dense(100, activation='relu', name='p3')(p2)
    p4 = Dense(100, activation='relu', name='p4')(p3)
    reg = Dense(3, name='reg_output')(p4)
    regression_outputs = reg
    #regression_losses = 'mae'
    regression_losses = 'mse'

    #regression = Model(calorimeter, regression_outputs)
    regression = Model([calorimeter, pos_info], regression_outputs)

    regression.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=regression_losses
    )


    if load_preTrained_model:
        regression = load_model(preTrained_model, custom_objects={'tf': tf,'single_layer_energy':single_layer_energy})
    

    logger.info('commencing training')
    f_DataSet = open(parse_args.datafile, 'r')
    Data = []
    Event = []
    Batch = []
    for line in f_DataSet: 
        #(idata, ievent) = line.split()
        idata = line.strip('\n')
        idata = idata.strip(' ')
        if "#" in idata: continue ##skip the commented one
        Data.append(idata)
        print(idata)
        d = h5py.File(str(idata), 'r')
        ievent   = d['Barrel_Hit'].shape[0]
        d.close()
        Event.append(float(ievent))
        Batch.append(int(float(ievent)/batch_size))
    total_event = sum(Event)
    f_DataSet.close() 
    print('total sample:', total_event)
    print('All Batch:', Batch)
 
    for epoch in range(nb_epochs):
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = sum(Batch)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_loss = []

        processed_batch = 0
        for ib in range(len(Batch)):
            first, mc_info = load_data(Data[ib])
            ibatch = Batch[ib]
            for index in range(ibatch):
                if verbose:
                    progress_bar.update(index)
                else:
                    if index % 100 == 0:
                        logger.info('processed {}/{} batches'.format(index + 1, ibatch))
                    elif index % 10 == 0:
                        logger.debug('processed {}/{} batches'.format(index + 1, ibatch))


                image_batch = first  [index * batch_size:(index + 1) * batch_size]
                #info_batch  = mc_info[index * batch_size:(index + 1) * batch_size]
                info_batch  = mc_info[index * batch_size:(index + 1) * batch_size,0:3]
                pos_info_batch= mc_info[index * batch_size:(index + 1) * batch_size,3:6]

                reg_outputs_real = info_batch

                loss_weights = np.ones(batch_size)

                real_batch_loss = regression.train_on_batch(
                    [image_batch, pos_info_batch],
                    reg_outputs_real,
                    loss_weights
                )
                if math.isnan(real_batch_loss): continue
                epoch_loss.append(np.array(real_batch_loss))
            processed_batch = processed_batch + ibatch
            logger.info('processed {}/{} total batches'.format(processed_batch, nb_batches))
        logger.info('Epoch {:3d}  loss: {}'.format( epoch + 1, np.mean(epoch_loss, axis=0)))

        # save weights every epoch
    #regression.save_weights(parse_args.weight_out, overwrite=True)
    #yaml_string = regression.to_yaml()
    #open(parse_args.str_out, 'w').write(yaml_string)
    regression.save(parse_args.model_out)
    print('done')

