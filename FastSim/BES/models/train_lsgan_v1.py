#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
https://arxiv.org/pdf/1611.04076.pdf
https://arxiv.org/pdf/1703.10593.pdf
add history of images
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
from scipy.stats import ks_2samp
import sys
import yaml
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
import math
##########################################################################
#using Mom dtheta, Mom dphi, Pos dz, Pos dphi, array with 121 cell energy#
#change the input to one dimension, not a list
#try to simulate sum E better 
#try to add mean and std of sum hit E for D
#try to make the training more converge
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
    parser.add_argument('--removeZ', action='store',type=ast.literal_eval,
                        default=False,
                        help='remove Z info')
    parser.add_argument('--test-dataset', action='store',type=str,
                        default='',
                        help='dataset for test')

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
    from keras.layers.merge import add, concatenate, multiply, subtract
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.utils.generic_utils import Progbar
    #from keras.utils.vis_utils     import plot_model
    K.set_image_dim_ordering('tf')

    from ops import (minibatch_discriminator, minibatch_output_shape, Dense3D,
                     calculate_energy, scale, inpainting_attention, sparsity_level, get_mc_info, single_layer_relative_diff_too_big, calculate_e3x3, calculate_e5x5, single_layer_energy, single_layer_relative_diff, mini_batch_mean, mini_batch_std , is_High_Z, single_layer_relative, latMom, secondMom, calculate_e1x1)

    from architectures import build_generator_2D, build_discriminator_2D, build_generator_2D_v1, build_generator_2D_v2, build_generator_2D_v2_1, build_generator_2D_v2_2, build_generator_2D_v3, build_discriminator_2D_v1, LS_loss

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
    test_dataset = parse_args.test_dataset

    disc_lr = parse_args.disc_lr
    gen_lr = parse_args.gen_lr
    adam_beta_1 = parse_args.adam_beta
    removeZ = parse_args.removeZ
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

    d = h5py.File(test_dataset, 'r')
    test_first   = d['Barrel_Hit'][:]
    test_mc_info = d['MC_info'][:] 
    ###### do normalization ##############
    test_mc_info[:,1] = (test_mc_info[:,1])/5    #M_dtheta
    test_mc_info[:,2] = (test_mc_info[:,2])/10   #M_dphi
    test_mc_info[:,3] = (test_mc_info[:,3])/2    #P_dz
    test_mc_info[:,4] = (test_mc_info[:,4])/2    #P_dphi
    test_mc_info[:,5] = (test_mc_info[:,5])/150  #Z
    d.close()



    logger.info('Building discriminator')

    calorimeter = Input(shape=sizes)
    input_Info  = Input(shape=(6,))
    #if removeZ:
    #    input_Info  = Input(shape=(5,))
    mom_info = Lambda(lambda x:x[:,0:1], name='mom_layer')(input_Info)
    print('mom:',mom_info.shape)
    #features =build_discriminator_2D(image=calorimeter,  epsilon=0.001)
    features =build_discriminator_2D_v1(image=calorimeter,  epsilon=0.001)
    print('features:',features.shape)
    lat_mom = Lambda(latMom)(calorimeter)
    print('lat_mom:',lat_mom.shape)
    second_mom = Lambda(secondMom)(calorimeter)
    print('second_mom:',second_mom.shape)
    energies = calculate_energy(calorimeter)
    print('energies:',energies.shape)
    energies_1x1 = calculate_e1x1(calorimeter)
    print('energies 1x1:',energies_1x1.shape)
    energies_3x3 = calculate_e3x3(calorimeter)
    print('energies 3x3:',energies_3x3.shape)
    energies_5x5 = calculate_e5x5(calorimeter)
    print('energies 5x5:',energies_5x5.shape)
    sparsity = Lambda(sparsity_level)(calorimeter)
    print('sparsity:',sparsity.shape)
    p0 = concatenate([features, input_Info, sparsity])
    p1 = Dense(1, name='features_output')(p0)
    #en_diff = Lambda(calculate_energy_diff,  arguments={'x_mom':mom_info})(energies)
    en_diff = subtract([energies, mom_info])
    print('en_diff:', en_diff.shape)
    rel_diff = Lambda(single_layer_relative_diff)([energies,mom_info])
    print('rel_diff:', rel_diff.shape)
    en_ratio = Lambda(single_layer_relative)([energies,mom_info])
    print('en_ratio:', en_ratio.shape)
    e1x1_ratio = Lambda(single_layer_relative)([energies_1x1,energies_3x3])
    print('e1x1_ratio:', e1x1_ratio.shape)
    e3x3_ratio = Lambda(single_layer_relative)([energies_3x3,energies_5x5])
    print('e3x3_ratio:', e3x3_ratio.shape)
    too_big = Lambda(single_layer_relative_diff_too_big)([energies,mom_info])
    print('too_big:',too_big.shape)
    too_High = Lambda(is_High_Z)(input_Info)
    print('too_High:',too_High.shape)
    en_mean = Lambda(mini_batch_mean)(energies)
    print('en_mean:',en_mean.shape)
    en_std = Lambda(mini_batch_std)(energies)
    print('en_std:',en_std.shape)
    en_diff_mean = Lambda(mini_batch_mean)(en_diff)
    print('en_diff_mean:',en_diff_mean.shape)
    en_diff_std  = Lambda(mini_batch_std)(en_diff)
    print('en_diff_std:',en_diff_std.shape)
    ref_diff_mean = Lambda(mini_batch_mean)(rel_diff)
    print('ref_diff_mean:',ref_diff_mean.shape)
    ref_diff_std  = Lambda(mini_batch_std)(rel_diff)
    print('ref_diff_std:',ref_diff_std.shape)
    en_ratio_mean = Lambda(mini_batch_mean)(en_ratio)
    print('en_ratio_mean:',en_ratio_mean.shape)
    en_ratio_std = Lambda(mini_batch_std)(en_ratio)
    print('en_ratio_std:',en_ratio_std.shape)

    en_info  = concatenate([energies, energies_3x3, energies_5x5, mom_info, en_diff, rel_diff, too_big, ref_diff_mean, ref_diff_std, too_High])
    #en_info  = concatenate([energies, energies_3x3, energies_5x5, mom_info, en_diff, rel_diff, too_big, ref_diff_mean, ref_diff_std, too_High, lat_mom])
    #en_info  = concatenate([energies, energies_3x3, energies_5x5, mom_info, en_diff, en_ratio, e1x1_ratio, e3x3_ratio, lat_mom, second_mom, too_big, en_ratio_mean, en_ratio_std, too_High])
    print('en_info',en_info.shape)
    p_e_1 = Dense(10, name='energy_features1')(en_info)
    p_e_2 = Dense(10, activation='relu', name='energy_features2')(p_e_1)
    p_e_3 = Dense(1, name='energy_features3')(p_e_2)
    p  = concatenate([p1, p_e_3])
    #p  = concatenate([p1, energies, en_diff])
    #p = concatenate([features, input_Info, energies, sparsity])
    
    #fake = Dense(1, activation='sigmoid', name='fakereal_output')(p)
    fake = Dense(1, name='fakereal_output')(p)
    discriminator_outputs = fake
    #discriminator_losses = LS_loss
    discriminator_losses = 'mse' 

    discriminator = Model([calorimeter, input_Info] , discriminator_outputs, name='discriminator')

    discriminator.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
    )

    logger.info('Building generator')

    n_mc_info = 6
#    if removeZ:
#        n_mc_info = 5
    latent = Input(shape=(latent_size+n_mc_info, ), name='gen_input')
    #input_info = Input(shape=(6, ), dtype='float32')
    #if removeZ:
    #     input_info = Input(shape=(5, ), dtype='float32')
    #generator_inputs = [latent, input_info]
    generator_inputs = latent

    #h = concatenate([latent, input_info])
    #img_layer  = build_generator_2D_v1(latent, 11, 11)
    #img_layer  = build_generator_2D_v2_1(latent, 11, 11)
    img_layer  = build_generator_2D_v2_2(latent, 11, 11)
    #img_layer  = build_generator_2D_v2(latent, 11, 11)
    #img_layer  = build_generator_2D_v3(latent, 11, 11)
    print('img_layer shape:',img_layer.shape)
    
    #output_info = Lambda(lambda x: x)(input_info) # same as input
    output_info = Lambda(get_mc_info,output_shape=(n_mc_info,),  arguments={'n':n_mc_info})(latent) # same as input
    pos_info = Lambda(lambda x: x[:,3:6] ,output_shape=(1,) )(output_info) # get pos info
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
    
    
    #gen_depth_info = Lambda(lambda x: x/100)(generator(generator_inputs)[1]) # same as input
    #combined_outputs = [discriminator( [generator(generator_inputs)[0], gen_depth_info, generator(generator_inputs)[2]]), reg_model([generator(generator_inputs)[0], gen_depth_info]) ]
    combined_outputs = [discriminator([generator(generator_inputs)[0], generator(generator_inputs)[1] ]), reg_model([ generator(generator_inputs)[0], generator(generator_inputs)[2] ] ) ]
    print('h31')
    combined_losses = ['mse', 'mae']
    #combined_losses = [LS_loss, 'mae']
    #combined_losses = ['binary_crossentropy', 'mean_squared_error']

    combined = Model(generator_inputs, combined_outputs, name='combined_model')
    print('h4')
    combined.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss=combined_losses
    )

    logger.info('commencing training')
 
    print('total sample:', first.shape[0])
    #disc_outputs_real = np.ones(batch_size)
    #disc_outputs_real = np.ones(batch_size)*0.95
    #tmp_real = 0.99
    tmp_real = 1
    #tmp_fake = 0.01
    tmp_fake = 0

    loss_weights      = np.ones(batch_size)
    combined_loss_weights      = [np.ones(batch_size), 1*np.ones(batch_size)]

    buffer_size = 10*batch_size
    buffer_image = 0
    buffer_info  = 0
    buffer_info0 = 0
    buffer_first = True
    assert (batch_size/2.0)%1 == 0
    pass_ks = False
    for epoch in range(nb_epochs):
        test_first, test_mc_info = shuffle(test_first, test_mc_info)
        ###### KS test ##############################################################
        ks_size = 50000
        info_batch_ks    = test_mc_info[0:ks_size] 
        image_batch_ks   = test_first  [0:ks_size]
        ks_real_en       = np.sum(image_batch_ks             , axis=tuple(range(1, len(image_batch_ks.shape))))
        ks_real_e5x5     = np.sum(image_batch_ks[:,3:8,3:8,:], axis=tuple(range(1, len(image_batch_ks.shape))))
        ks_real_e3x3     = np.sum(image_batch_ks[:,4:7,4:7,:], axis=tuple(range(1, len(image_batch_ks.shape))))
        ks_real_en_Z     = np.sum(image_batch_ks             , axis=1)
        ks_real_en_Z     = np.reshape(ks_real_en_Z, (-1,))
        ks_real_en_phi   = np.sum(image_batch_ks             , axis=2)
        ks_real_en_phi   = np.reshape(ks_real_en_phi, (-1,))
        #image_batch_ks   = np.sum(image_batch_ks, axis=0)
        #image_batch_ks   = np.reshape(image_batch_ks, (-1,))
        ks_noise         = np.random.normal(0, 1, (ks_size, latent_size))
        ks_generator_inputs = np.concatenate ((ks_noise, info_batch_ks), axis=-1)
        ks_generated_images = generator.predict(ks_generator_inputs, verbose=0)[0]
        #ks_generated_images = np.sum(ks_generated_images, axis=0)
        #ks_generated_images = np.reshape(ks_generated_images, (-1,))
        ks_fake_en          = np.sum(ks_generated_images             , axis=tuple(range(1, len(image_batch_ks.shape))))
        ks_fake_e5x5        = np.sum(ks_generated_images[:,3:8,3:8,:], axis=tuple(range(1, len(image_batch_ks.shape))))
        ks_fake_e3x3        = np.sum(ks_generated_images[:,4:7,4:7,:], axis=tuple(range(1, len(image_batch_ks.shape))))
        ks_fake_en_Z        = np.sum(ks_generated_images             , axis=1)
        ks_fake_en_Z        = np.reshape(ks_fake_en_Z, (-1,))
        ks_fake_en_phi      = np.sum(ks_generated_images             , axis=2)
        ks_fake_en_phi      = np.reshape(ks_fake_en_phi, (-1,))
        ks_result_en     = ks_2samp(ks_real_en  , ks_fake_en  ).statistic
        ks_result_e3x3   = ks_2samp(ks_real_e3x3, ks_fake_e3x3).statistic
        ks_result_e5x5   = ks_2samp(ks_real_e5x5, ks_fake_e5x5).statistic
        ks_result_en_Z   = ks_2samp(ks_real_en_Z, ks_fake_en_Z).statistic
        ks_result_en_phi = ks_2samp(ks_real_en_phi, ks_fake_en_phi).statistic
        ## http://sparky.rice.edu/astr360/kstest.pdf
        cell_size = 11
        alpha = 0.05
        c_alpha = 1.36
        D_aplha = c_alpha*math.sqrt((ks_size+ks_size)/(ks_size*ks_size))
        D_aplha1 = c_alpha*math.sqrt((ks_size*cell_size+ks_size*cell_size)/(ks_size*cell_size*ks_size*cell_size))
        print('D=',D_aplha,',ks en=',ks_result_en,',e3x3=',ks_result_e3x3,',e5x5=',ks_result_e5x5, ',D1=',D_aplha1,',en_Z=',ks_result_en_Z,',en phi=',ks_result_en_phi)
        if ks_result_en < D_aplha and ks_result_e3x3 < D_aplha and ks_result_e5x5 < D_aplha and ks_result_en_Z < D_aplha1 and ks_result_en_phi < D_aplha1: ## can't reject the hypothesis that real and gen are the same at alpha CL
            pass_ks = True
            print('Find it!')
            tmp_gen_out_name  = (parse_args.gen_out).replace('.h5','_KSepoch%d.h5'%epoch)
            tmp_dis_out_name  = (parse_args.dis_out).replace('.h5','_KSepoch%d.h5'%epoch)
            tmp_comb_out_name = (parse_args.comb_out).replace('.h5','_KSepoch%d.h5'%epoch)
            generator.save    (tmp_gen_out_name  )
            discriminator.save(tmp_dis_out_name  )
            combined .save    (tmp_comb_out_name )
            sys.exit()
        #####################################################
        first, mc_info      = shuffle(first, mc_info)
        mc_info_cp1         = shuffle(mc_info)
        mc_info_cp2         = shuffle(mc_info)
        mc_info_v1          = shuffle(mc_info_v1)
        mc_info_v1_mom      = shuffle(mc_info_v1[:,0:1] , random_state=1 )
        mc_info_v1_M_dtheta = shuffle(mc_info_v1[:,1:2] , random_state=1 )
        mc_info_v1_M_dphi   = shuffle(mc_info_v1[:,2:3] , random_state=1 )
        mc_info_v1_P_dz     = shuffle(mc_info_v1[:,3:4]   )
        mc_info_v1_P_dphi   = shuffle(mc_info_v1[:,4:5]   )
        mc_info_v1_P_z      = shuffle(mc_info_v1[:,5:6] , random_state=1 )
        mc_info_v1_new      = np.concatenate((mc_info_v1_mom, mc_info_v1_M_dtheta, mc_info_v1_M_dphi, mc_info_v1_P_dz, mc_info_v1_P_dphi, mc_info_v1_P_z),axis=-1)
        print('mc_info_v1_new shape=', mc_info_v1_new.shape)
        mc_info_v2_mom      = shuffle(mc_info_v1[:,0:1] , random_state=2 )
        mc_info_v2_M_dtheta = shuffle(mc_info_v1[:,1:2] , random_state=2 )
        mc_info_v2_M_dphi   = shuffle(mc_info_v1[:,2:3] , random_state=2 )
        mc_info_v2_P_dz     = shuffle(mc_info_v1[:,3:4]   )
        mc_info_v2_P_dphi   = shuffle(mc_info_v1[:,4:5]   )
        mc_info_v2_P_z      = shuffle(mc_info_v1[:,5:6] , random_state=2 )
        mc_info_v2_new      = np.concatenate((mc_info_v2_mom, mc_info_v2_M_dtheta, mc_info_v2_M_dphi, mc_info_v2_P_dz, mc_info_v2_P_dphi, mc_info_v2_P_z),axis=-1)
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(first.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        disc_outputs_real = np.ones(batch_size)*tmp_real
        disc_outputs_fake = np.ones(batch_size)*tmp_fake
        #tmp_real = tmp_real + 0.05 if tmp_real<=0.95 else tmp_real
        #tmp_fake = tmp_fake - 0.05 if tmp_fake>=0.05 else tmp_fake

        for index in range(nb_batches):
            '''
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    logger.info('processed {}/{} batches'.format(index + 1, nb_batches))
                elif index % 10 == 0:
                    logger.debug('processed {}/{} batches'.format(index + 1, nb_batches))
            '''
           
            # generate a new batch of noise
            noise = np.random.normal(0, 1, (batch_size, latent_size))
            # get a batch of real images
            image_batch = first  [index * batch_size:(index + 1) * batch_size]
            info_batch    =  mc_info       [index * batch_size:(index + 1) * batch_size] 
            info_batch_v1 =  mc_info_v1_new[index * batch_size:(index + 1) * batch_size] 
            #info_batch_v1 =  mc_info_cp1   [index * batch_size:(index + 1) * batch_size] 
            info_batch_v2 =  mc_info_v2_new[index * batch_size:(index + 1) * batch_size] 
            #info_batch_v2 =  mc_info_cp2   [index * batch_size:(index + 1) * batch_size] 

            generator_inputs = np.concatenate ((noise, info_batch_v1), axis=-1)
            generated_images = generator.predict(generator_inputs, verbose=0)
            
            real_batch_loss = discriminator.train_on_batch( [image_batch,  info_batch],  disc_outputs_real,  loss_weights )

            if buffer_first==False:
                if buffer_image.shape[0] > batch_size:
                    index0 = shuffle(list(range(buffer_image.shape[0]))) 
                    index0 = index0[0:int(batch_size/2)]
                    index1 = shuffle(list(range(batch_size))) 
                    index1 = index1[0:int(batch_size/2)]
                    generated_images[0] = np.concatenate((buffer_image[index0],generated_images[0][index1]),axis=0)
                    generated_images[1] = np.concatenate((buffer_info [index0],generated_images[1][index1]),axis=0)
                    #generated_images[2] = np.concatenate((buffer_info0[index0],generated_images[2][index1]),axis=0)
            fake_batch_loss = discriminator.train_on_batch( [generated_images[0], generated_images[1] ],  disc_outputs_fake, loss_weights )
            #print('real_batch_loss=',real_batch_loss)
            #print('fake_batch_loss=',fake_batch_loss)
            if index == (nb_batches-1):
                real_pred = discriminator.predict_on_batch([image_batch, info_batch])
                fake_pred = discriminator.predict_on_batch([generated_images[0], generated_images[1]])
                print('real_pred:\n',real_pred)
                print('fake_pred:\n',fake_pred)

            epoch_disc_loss.append( (np.array(fake_batch_loss) + np.array(real_batch_loss)) / 2)

            ####################
            max_try = 2
            tried =0
            gen_train_loss = 0
            while True:
                noise            = np.random.normal(0, 1, (batch_size, latent_size))
                combined_inputs  = np.concatenate ((noise, info_batch_v2), axis=-1)
                combined_outputs =[np.ones(batch_size), info_batch_v2[:,0:3] ]
                gen_train_loss = combined.train_on_batch( combined_inputs, combined_outputs, combined_loss_weights )
                gen_train_loss = np.array(gen_train_loss)
                tried = tried + 1
                if tried >= max_try : break
            epoch_gen_loss.append(np.array(gen_train_loss))
            if buffer_first:
                buffer_image = generated_images[0]
                buffer_info  = generated_images[1]
                #buffer_info0 = generated_images[2]
                buffer_first = False
            else:
                if buffer_image.shape[0] < buffer_size:
                    buffer_image = np.concatenate ((buffer_image, generated_images[0]), axis=0)
                    buffer_info  = np.concatenate ((buffer_info , generated_images[1]), axis=0)
                    #buffer_info0 = np.concatenate ((buffer_info0, generated_images[2]), axis=0)
                else:
                    bindex = shuffle(list(range(buffer_image.shape[0]))) 
                    bindex = bindex[0:batch_size]
                    buffer_image[bindex] = generated_images[0]
                    buffer_info [bindex] = generated_images[1]
                    #buffer_info0[bindex] = generated_images[2]

        logger.info('Epoch {:3d} Generator loss: {}'.format(
            epoch + 1, np.mean(epoch_gen_loss, axis=0)))
        logger.info('Epoch {:3d} Discriminator loss: {}'.format(
            epoch + 1, np.mean(epoch_disc_loss, axis=0)))
        tmp_gen_out_name  = (parse_args.gen_out).replace('.h5','_epoch%d.h5'%epoch)
        tmp_dis_out_name  = (parse_args.dis_out).replace('.h5','_epoch%d.h5'%epoch)
        tmp_comb_out_name = (parse_args.comb_out).replace('.h5','_epoch%d.h5'%epoch)
        if (epoch+1)%1==0 and True :
            generator.save    (tmp_gen_out_name  )
            discriminator.save(tmp_dis_out_name  )
            combined .save    (tmp_comb_out_name )

    generator.save(parse_args.gen_out)
    discriminator.save(parse_args.dis_out)
    combined .save(parse_args.comb_out)
    print('done reg training')
