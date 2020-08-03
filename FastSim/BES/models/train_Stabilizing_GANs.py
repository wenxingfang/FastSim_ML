#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
https://github.com/rothk/Stabilizing_GANs
"""

from __future__ import print_function

import argparse
from collections import defaultdict
import logging


import ast
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
from functools import partial

##########################################################################
#using Mom dtheta, Mom dphi, Pos dz, Pos dphi, array with 121 cell energy#
#change the input to one dimension, not a list
#try to simulate sum E better 
#try to add mean and std of sum hit E for D
#try to use wgan with gp
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
    parser.add_argument('--LoadTrainedWeight', action='store',type=ast.literal_eval,
                        default=False,
                        help='remove Z info')
    parser.add_argument('--dis_model_weight_in', action='store',type=str,
                        default='',
                        help='dis model in')
    parser.add_argument('--gen_model_weight_in', action='store',type=str,
                        default='',
                        help='gen model in')
    parser.add_argument('--comb_model_weight_in', action='store',type=str,
                        default='',
                        help='comb model in')
    parser.add_argument('--loss_out', action='store',type=str,
                        default='',
                        help='dis loss out')

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
    from keras.layers import (Input, Dense, Reshape, Conv2D, Conv3D, LeakyReLU, BatchNormalization, UpSampling2D, UpSampling3D, Cropping2D, LocallyConnected2D, Activation, ZeroPadding2D, Dropout, Lambda, Flatten, AveragePooling2D, ReLU, Cropping3D, MaxPooling2D)
    from keras.layers.merge import add, concatenate, multiply, subtract
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.utils.generic_utils import Progbar
    #from keras.utils.vis_utils     import plot_model
    K.set_image_dim_ordering('tf')

    from ops import (minibatch_discriminator, minibatch_output_shape, Dense3D,
                     calculate_energy, scale, inpainting_attention, sparsity_level, get_mc_info, single_layer_relative_diff_too_big, calculate_e3x3, calculate_e5x5, single_layer_energy, single_layer_relative_diff, mini_batch_mean, mini_batch_std ,normalize ,MyDense2D, single_layer_relative, is_High_Z)

    from architectures import build_generator_2D, build_discriminator_2D, build_generator_2D_v1, build_generator_2D_v2, build_generator_2D_v2_2, build_discriminator_2D_v2, RandomWeightedAverage, RandomWeightedAverage1, binary_loss, Discriminator_Regularizer 

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
    #removeZ = parse_args.removeZ
    reg_model_in = parse_args.reg_model_in
    reg_weight_in = parse_args.reg_weight_in
    LoadTrainedWeight = parse_args.LoadTrainedWeight

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
    mc_info = d['MC_info'][:] 
    mc_info_v1 = d['MC_info'][:] 
    d.close()
    ###### do normalization ##############
    mc_info[:,1] = (mc_info[:,1])/5    #M_dtheta
    mc_info[:,2] = (mc_info[:,2])/10   #M_dphi
    mc_info[:,3] = (mc_info[:,3])/2    #P_dz
    mc_info[:,4] = (mc_info[:,4])/2    #P_dphi
    mc_info[:,5] = (mc_info[:,5])/150  #Z

    mc_info_v1[:,1] = (mc_info_v1[:,1])/5    #M_dtheta
    mc_info_v1[:,2] = (mc_info_v1[:,2])/10   #M_dphi
    mc_info_v1[:,3] = (mc_info_v1[:,3])/2    #P_dz
    mc_info_v1[:,4] = (mc_info_v1[:,4])/2    #P_dphi
    mc_info_v1[:,5] = (mc_info_v1[:,5])/150  #Z
    sizes = [ first.shape[1], first.shape[2], 1]
    print("first:",first.shape,",mc info:", mc_info.shape)
    print(sizes)


    logger.info('Building discriminator')

    real_image = Input(shape=sizes)
    real_info  = Input(shape=(6,))
    fake_image = Input(shape=sizes)
    fake_info  = Input(shape=(6,))
    aveg_image = RandomWeightedAverage ()([real_image, fake_image])
    aveg_info  = RandomWeightedAverage1()([real_info, fake_info])
    #################################################
    la_norm        = Lambda(normalize, arguments={'epsilon':0.001})
    la_Conv2D_1    = Conv2D(16, (2, 2) , padding='valid')
    la_LeakyReLU_1 = LeakyReLU()
    la_Dropout_1   = Dropout(0.1)
    la_Conv2D_2    = Conv2D(32, (3, 3), padding='valid')
    la_LeakyReLU_2 = LeakyReLU()
    la_Dropout_2   = Dropout(0.1)
    la_Conv2D_3    = Conv2D(64, (3, 3), padding='valid')
    la_LeakyReLU_3 = LeakyReLU()
    la_Dropout_3   = Dropout(0.1)
    la_Conv2D_s2    = Conv2D(64, (2, 2),strides=(2, 2), padding='valid')
    la_Flatten_1   = Flatten()
    la_K = MyDense2D(10, 10)
    la_minibatch = Lambda(minibatch_discriminator)

    la_mom      = Lambda(lambda x:x[:,0:1], name='mom_layer')
    la_energies = calculate_energy
    la_e3x3     = calculate_e3x3
    la_e5x5     = calculate_e5x5
    la_sparsity = Lambda(sparsity_level)
    la_en_ratio = Lambda(single_layer_relative)
    la_too_big  = Lambda(single_layer_relative_diff_too_big)
    la_mean = Lambda(mini_batch_mean)
    la_std  = Lambda(mini_batch_std)
    la_too_High = Lambda(is_High_Z)

    la_dense_1  = Dense(1, name='fakereal_output')
    #################################################

    real_norm = la_norm(real_image)
    real_la_Conv2D_1 = la_Conv2D_1(real_norm)
    real_la_LeakyReLU_1 = la_LeakyReLU_1(real_la_Conv2D_1)
    real_la_Dropout_1 = la_Dropout_1(real_la_LeakyReLU_1)
    real_la_Conv2D_2 = la_Conv2D_2(real_la_Dropout_1)
    real_la_LeakyReLU_2 = la_LeakyReLU_2(real_la_Conv2D_2)
    real_la_Dropout_2 = la_Dropout_2(real_la_LeakyReLU_2)
    real_la_Conv2D_3 = la_Conv2D_3(real_la_Dropout_2)
    real_la_LeakyReLU_3 = la_LeakyReLU_3(real_la_Conv2D_3)
    real_la_Dropout_3 = la_Dropout_3(real_la_LeakyReLU_3)
    real_la_Conv2D_s2 = la_Conv2D_s2(real_la_Dropout_3)
    real_la_Flatten_1 = la_Flatten_1(real_la_Conv2D_s2)
    real_K            = la_K(real_la_Flatten_1)
    real_la_minibatch = la_minibatch(real_K)
    real_con1          = concatenate([real_la_Flatten_1, real_la_minibatch])
    real_mom          = la_mom(real_info)
    real_en           = la_energies(real_image)
    real_e3x3         = la_e3x3(real_image)
    real_e5x5         = la_e5x5(real_image)
    real_sparsity     = la_sparsity(real_image)
    real_en_diff      = subtract([real_en, real_mom])    
    real_en_ratio     = la_en_ratio([real_en, real_mom])
    real_too_big      = la_too_big([real_en, real_mom])
    real_en_ratio_mean= la_mean(real_en_ratio)
    real_en_ratio_std = la_std (real_en_ratio)
    real_too_High     = la_too_High(real_info)
    real_con2         = concatenate([real_sparsity, real_mom, real_en, real_e3x3, real_e5x5, real_en_diff, real_en_ratio, real_en_ratio_mean, real_en_ratio_std, real_too_big, real_too_High])
    real_con3         = concatenate([real_con1, real_con2])
    real_out          = la_dense_1(real_con3)

    fake_norm = la_norm(fake_image)
    fake_la_Conv2D_1 = la_Conv2D_1(fake_norm)
    fake_la_LeakyReLU_1 = la_LeakyReLU_1(fake_la_Conv2D_1)
    fake_la_Dropout_1 = la_Dropout_1(fake_la_LeakyReLU_1)
    fake_la_Conv2D_2 = la_Conv2D_2(fake_la_Dropout_1)
    fake_la_LeakyReLU_2 = la_LeakyReLU_2(fake_la_Conv2D_2)
    fake_la_Dropout_2 = la_Dropout_2(fake_la_LeakyReLU_2)
    fake_la_Conv2D_3 = la_Conv2D_3(fake_la_Dropout_2)
    fake_la_LeakyReLU_3 = la_LeakyReLU_3(fake_la_Conv2D_3)
    fake_la_Dropout_3 = la_Dropout_3(fake_la_LeakyReLU_3)
    fake_la_Conv2D_s2 = la_Conv2D_s2(fake_la_Dropout_3)
    fake_la_Flatten_1 = la_Flatten_1(fake_la_Conv2D_s2)
    fake_K            = la_K(fake_la_Flatten_1)
    fake_la_minibatch = la_minibatch(fake_K)
    fake_con1          = concatenate([fake_la_Flatten_1, fake_la_minibatch])
    fake_mom          = la_mom(fake_info)
    fake_en           = la_energies(fake_image)
    fake_e3x3         = la_e3x3(fake_image)
    fake_e5x5         = la_e5x5(fake_image)
    fake_sparsity     = la_sparsity(fake_image)
    fake_en_diff      = subtract([fake_en, fake_mom])    
    fake_en_ratio     = la_en_ratio([fake_en, fake_mom])
    fake_too_big      = la_too_big([fake_en, fake_mom])
    fake_en_ratio_mean= la_mean(fake_en_ratio)
    fake_en_ratio_std = la_std (fake_en_ratio)
    fake_too_High     = la_too_High(fake_info)
    fake_con2         = concatenate([fake_sparsity, fake_mom, fake_en, fake_e3x3, fake_e5x5, fake_en_diff, fake_en_ratio, fake_en_ratio_mean, fake_en_ratio_std, fake_too_big, fake_too_High])
    fake_con3         = concatenate([fake_con1, fake_con2])
    fake_out          = la_dense_1(fake_con3)
    '''
    aveg_norm = la_norm(aveg_image)
    aveg_la_Conv2D_1 = la_Conv2D_1(aveg_norm)
    aveg_la_LeakyReLU_1 = la_LeakyReLU_1(aveg_la_Conv2D_1)
    aveg_la_Dropout_1 = la_Dropout_1(aveg_la_LeakyReLU_1)
    aveg_la_Conv2D_2 = la_Conv2D_2(aveg_la_Dropout_1)
    aveg_la_LeakyReLU_2 = la_LeakyReLU_2(aveg_la_Conv2D_2)
    aveg_la_Dropout_2 = la_Dropout_2(aveg_la_LeakyReLU_2)
    aveg_la_Conv2D_3 = la_Conv2D_3(aveg_la_Dropout_2)
    aveg_la_LeakyReLU_3 = la_LeakyReLU_3(aveg_la_Conv2D_3)
    aveg_la_Dropout_3 = la_Dropout_3(aveg_la_LeakyReLU_3)
    aveg_la_Conv2D_s2 = la_Conv2D_s2(aveg_la_Dropout_3)
    aveg_la_Flatten_1 = la_Flatten_1(aveg_la_Conv2D_s2)
    aveg_K            = la_K(aveg_la_Flatten_1)
    aveg_la_minibatch = la_minibatch(aveg_K)
    aveg_con1          = concatenate([aveg_la_Flatten_1, aveg_la_minibatch])
    aveg_mom          = la_mom(aveg_info)
    aveg_en           = la_energies(aveg_image)
    aveg_e3x3         = la_e3x3(aveg_image)
    aveg_e5x5         = la_e5x5(aveg_image)
    aveg_sparsity     = la_sparsity(aveg_image)
    aveg_en_diff      = subtract([aveg_en, aveg_mom])    
    aveg_en_ratio     = la_en_ratio([aveg_en, aveg_mom])
    aveg_too_big      = la_too_big([aveg_en, aveg_mom])
    aveg_en_ratio_mean= la_mean(aveg_en_ratio)
    aveg_en_ratio_std = la_std (aveg_en_ratio)
    aveg_too_High     = la_too_High(aveg_info)
    aveg_con2         = concatenate([aveg_sparsity, aveg_mom, aveg_en, aveg_e3x3, aveg_e5x5, aveg_en_diff, aveg_en_ratio, aveg_en_ratio_mean, aveg_en_ratio_std, aveg_too_big, aveg_too_High])
    aveg_con3         = concatenate([aveg_con1, aveg_con2])
    aveg_out          = la_dense_1(aveg_con3)
    '''



    #################################################
    #real_out =build_discriminator_2D_v2(image=real_image, info=real_info,  epsilon=0.001)
    #fake_out =build_discriminator_2D_v2(image=fake_image, info=fake_info,  epsilon=0.001)
    #aveg_out =build_discriminator_2D_v2(image=aveg_image, info=aveg_info,  epsilon=0.001)

    # The gradient penalty loss function requires the input averaged samples to get
    # gradients. However, Keras loss functions can only have two arguments, y_true and
    # y_pred. We get around this by making a partial() of the function with the averaged
    # samples here.
    #averaged_samples = [aveg_image, aveg_info]
    GAMMA = 0 #0.1
    discriminator_regularizer = partial(Discriminator_Regularizer,
                              real_samples=[real_image, real_info],
                              fake_out    =fake_out,
                              fake_samples=[fake_image, fake_info],
                              weight=0.5*GAMMA)
    # Functions need names or Keras will throw an error
    discriminator_regularizer.__name__ = ' Discriminator_Regularizer'


    #discriminator = Model([real_image, real_info, fake_image, fake_info ] , [real_out, fake_out, aveg_out], name='discriminator')
    discriminator = Model([real_image, real_info, fake_image, fake_info ] , [real_out, fake_out, real_out], name='discriminator')
    #bin_ = keras.backend.binary_crossentropy(target, output, from_logits=False)
    discriminator.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=[binary_loss, binary_loss,  discriminator_regularizer]
    )

    logger.info('Building generator')

    n_mc_info = 6
    latent = Input(shape=(latent_size+n_mc_info, ), name='gen_input')
    generator_inputs = latent
    #img_layer  = build_generator_2D_v1(latent, 11, 11)
    #img_layer  = build_generator_2D_v2(latent, 11, 11)
    img_layer  = build_generator_2D_v2_2(latent, 11, 11)
    print('img_layer shape:',img_layer.shape)
    #output_info = Lambda(lambda x: x)(input_info) # same as input
    output_info = Lambda(get_mc_info,output_shape=(n_mc_info,),  arguments={'n':n_mc_info})(latent) # same as input
    pos_info = Lambda(lambda x: x[:,3:6] ,output_shape=(3,) )(output_info) # get pos info
    print('pos_info shape:',pos_info.shape)
    generator_outputs =  [img_layer, output_info, pos_info]
    generator = Model(generator_inputs, generator_outputs, name='generator')

    generator.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    print('h3')
######### regression part ##########################
    reg_model = load_model(parse_args.reg_model_in, custom_objects={'tf': tf, 'single_layer_energy':single_layer_energy})
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
    discriminator_out = discriminator([generator(generator_inputs)[0], generator(generator_inputs)[1], generator(generator_inputs)[0], generator(generator_inputs)[1] ])
    combined_outputs = [discriminator_out[0], reg_model([ generator(generator_inputs)[0], generator(generator_inputs)[2] ] ) ]
    print('h31')
    #combined_losses = [ wasserstein_loss, 'mean_squared_error']
    combined_losses = [ binary_loss, 'mae']
    combined = Model(generator_inputs, combined_outputs, name='combined_model')
    print('h4')
    combined.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss=combined_losses
    )

    if LoadTrainedWeight:
        logger.info('ReLoading generator weights')
        discriminator.load_weights(parse_args.dis_model_weight_in)
        logger.info('ReLoading generator weights')
        generator = load_weights(parse_args.gen_model_weight_in)
        logger.info('ReLoading combined weights')
        combined = load_weights(parse_args.comb_model_weight_in)


    logger.info('commencing training')
 
    print('total sample:', first.shape[0])
    #disc_outputs_real = np.ones(batch_size)
    #disc_outputs_real = np.ones(batch_size)*0.95
    #tmp_real = 0.59
    #tmp_fake = 0.4
    tmp_real = 0.99
    #tmp_fake = 0.0
    tmp_fake = 0.01

    f_out = open(parse_args.loss_out,'w')

    loss_weights      = np.ones(batch_size)
    combined_loss_weights      = [np.ones(batch_size), 1*np.ones(batch_size)]
    disc_outputs =[tmp_real*np.ones(batch_size), tmp_fake*np.ones(batch_size), np.zeros(batch_size)]

    for epoch in range(nb_epochs):
        first, mc_info = shuffle(first, mc_info)
        mc_info_v1     = shuffle(mc_info_v1)
        mc_info_v1_mom      = shuffle(mc_info_v1[:,0:1] , random_state=1  )
        mc_info_v1_M_dtheta = shuffle(mc_info_v1[:,1:2] , random_state=1  )
        mc_info_v1_M_dphi   = shuffle(mc_info_v1[:,2:3] , random_state=1  )
        mc_info_v1_P_dz     = shuffle(mc_info_v1[:,3:4]   )
        mc_info_v1_P_dphi   = shuffle(mc_info_v1[:,4:5]   )
        mc_info_v1_P_z      = shuffle(mc_info_v1[:,5:6] , random_state=1  )
        mc_info_v1_new      = np.concatenate((mc_info_v1_mom, mc_info_v1_M_dtheta, mc_info_v1_M_dphi, mc_info_v1_P_dz, mc_info_v1_P_dphi, mc_info_v1_P_z),axis=-1)
        print('mc_info_v1_new shape=', mc_info_v1_new.shape)
        mc_info_v2_mom      = shuffle(mc_info_v1[:,0:1] , random_state=2  )
        mc_info_v2_M_dtheta = shuffle(mc_info_v1[:,1:2] , random_state=2  )
        mc_info_v2_M_dphi   = shuffle(mc_info_v1[:,2:3] , random_state=2  )
        mc_info_v2_P_dz     = shuffle(mc_info_v1[:,3:4]   )
        mc_info_v2_P_dphi   = shuffle(mc_info_v1[:,4:5]   )
        mc_info_v2_P_z      = shuffle(mc_info_v1[:,5:6] , random_state=2  )
        mc_info_v2_new      = np.concatenate((mc_info_v2_mom, mc_info_v2_M_dtheta, mc_info_v2_M_dphi, mc_info_v2_P_dz, mc_info_v2_P_dphi, mc_info_v2_P_z),axis=-1)
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(first.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        disc_outputs_real = np.ones(batch_size) *tmp_real
        disc_outputs_fake = np.zeros(batch_size)*tmp_fake
        #tmp_real = tmp_real + 0.05 if tmp_real<=0.95 else tmp_real
        #tmp_fake = tmp_fake - 0.05 if tmp_fake>=0.05 else tmp_fake

        for index in range(nb_batches):
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    logger.info('processed {}/{} batches'.format(index + 1, nb_batches))
                elif index % 10 == 0:
                    logger.debug('processed {}/{} batches'.format(index + 1, nb_batches))
           
            # get a batch of real images
            image_batch   =  first         [index * batch_size:(index + 1) * batch_size]
            info_batch    =  mc_info       [index * batch_size:(index + 1) * batch_size] 
            info_batch_v1 =  mc_info_v1_new[index * batch_size:(index + 1) * batch_size] 
            info_batch_v2 =  mc_info_v2_new[index * batch_size:(index + 1) * batch_size] 

            ########################
            noise = np.random.normal(0, 1, (batch_size, latent_size))
            generator_inputs = np.concatenate ((noise, info_batch_v1), axis=-1)
            generated_images = generator.predict(generator_inputs, verbose=0)
            disc_batch_loss = discriminator.train_on_batch( [image_batch, info_batch, generated_images[0], generated_images[1]],  disc_outputs)
            epoch_disc_loss.append( np.array(disc_batch_loss))
            ########################
            '''
            tried = 0
            n_critic = 5
            samp = np.random.randint(0,nb_batches,n_critic) # random int for 0 to nb_batches-1               
            while tried<n_critic:
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                info_batch_v11 = mc_info_v1_new[samp[tried] * batch_size:(samp[tried] + 1) * batch_size]
                generator_inputs = np.concatenate ((noise, info_batch_v11), axis=-1)
                generated_images = generator.predict(generator_inputs, verbose=0)
                image_batch   =  first         [samp[tried] * batch_size:(samp[tried] + 1) * batch_size]
                info_batch    =  mc_info       [samp[tried] * batch_size:(samp[tried] + 1) * batch_size] 
                disc_batch_loss = discriminator.train_on_batch( [image_batch, info_batch, generated_images[0], generated_images[1]],  disc_outputs)
                epoch_disc_loss.append( np.array(disc_batch_loss))
                tried = tried + 1
            '''
            ########################
            #n_gen = 2
            #samp = np.random.randint(0,nb_batches,n_gen) # random int for 0 to nb_batches-1               
            #tried = 0
            #while tried<n_gen:
            #    noise            = np.random.normal(0, 1, (batch_size, latent_size))
            #    info_batch_v22   = mc_info_v2_new[samp[tried] * batch_size:(samp[tried] + 1) * batch_size]
            #    combined_inputs  = np.concatenate ((noise, info_batch_v22), axis=-1)
            #    combined_outputs =[1*np.ones(batch_size), info_batch_v22[:,0:3] ]
            #    gen_train_loss   = combined.train_on_batch(combined_inputs, combined_outputs)
            #    epoch_gen_loss.append(np.array(gen_train_loss))
            #    tried = tried + 1
            ####################
            max_try = 2
            tried =0
            while True:
                noise            = np.random.normal(0, 1, (batch_size, latent_size))
                combined_inputs  = np.concatenate ((noise, info_batch_v2), axis=-1)
                combined_outputs =[np.ones(batch_size), info_batch_v2[:,0:3] ]
                gen_train_loss = combined.train_on_batch( combined_inputs, combined_outputs, combined_loss_weights )
                gen_train_loss = np.array(gen_train_loss)
                tried = tried + 1
                if tried >= max_try : break
            epoch_gen_loss.append(np.array(gen_train_loss))
            ####################

        logger.info('Epoch {:3d} Generator loss: {}'.format(
            epoch + 1, np.mean(epoch_gen_loss, axis=0)))
        logger.info('Epoch {:3d} Discriminator loss: {}'.format(
            epoch + 1, np.mean(epoch_disc_loss, axis=0)))

        tmp_tot_loss  = np.mean(epoch_disc_loss, axis=0)[0]
        tmp_real_loss = np.mean(epoch_disc_loss, axis=0)[1]
        tmp_fake_loss = np.mean(epoch_disc_loss, axis=0)[2]
        tmp_aveg_loss = np.mean(epoch_disc_loss, axis=0)[3]
        f_out.write('%d %f %f %f %f \n'%(epoch, tmp_tot_loss, tmp_real_loss, tmp_fake_loss, tmp_aveg_loss))    
        tmp_gen_out_name  = (parse_args.gen_out).replace('.h5','_epoch%d.h5'%epoch)
        tmp_dis_out_name  = (parse_args.dis_out).replace('.h5','_epoch%d.h5'%epoch)
        tmp_comb_out_name = (parse_args.comb_out).replace('.h5','_epoch%d.h5'%epoch)
        if (epoch+1)%10==0 or True:
            generator.save    (tmp_gen_out_name  )
            discriminator.save(tmp_dis_out_name  )
            combined .save    (tmp_comb_out_name )
    f_out.close()
    generator.save(parse_args.gen_out)
    discriminator.save(parse_args.dis_out)
    combined .save(parse_args.comb_out)
    tmp_gen_out_name  = (parse_args.gen_out).replace('.h5','_weight.h5')
    tmp_dis_out_name  = (parse_args.dis_out).replace('.h5','_weight.h5')
    tmp_comb_out_name  = (parse_args.comb_out).replace('.h5','_weight.h5')
    generator    .save_weights(tmp_gen_out_name)
    discriminator.save_weights(tmp_dis_out_name)
    combined     .save_weights(tmp_comb_out_name)
    print('done reg training')
