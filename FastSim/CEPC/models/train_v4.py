#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: train_v3.py
Using mc info : mom, dtheta, dphi, Z 
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

def load_data(datafile):
    d = h5py.File(datafile, 'r')
    first   = np.expand_dims(d['Barrel_Hit'][:],-1)
    mc_info = d['MC_info'][:] 
    ###### do normalization ##############
    mc_info[:,0] = (mc_info[:,0]-55)/45
    mc_info[:,1] = (mc_info[:,1]-90)/50
    mc_info[:,2] = (mc_info[:,2])/20
    mc_info[:,3] = (mc_info[:,3])/2000
    d.close()
    print("first:",first.shape,",mc info:", mc_info.shape)
    ###### do scale ##############
    first, mc_info = shuffle(first, mc_info)
    mc_info_v1 = np.copy(mc_info)
    mc_info_v1 = shuffle(mc_info_v1)
    return first, mc_info, mc_info_v1


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

    parser.add_argument('--datafile', action='store', type=str,
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

    from architectures import build_generator_3D, build_discriminator_3D, build_regression, build_discriminator_3D_v1, build_discriminator_3D_v2 ,build_discriminator_3D_v3, build_generator_3D_v1

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


    '''
    import h5py
    d = h5py.File(parse_args.dataset, 'r')
    first   = np.expand_dims(d['Barrel_Hit'][:],-1)
    mc_info = d['MC_info'][:] 
    d.close()
    ###### do normalization ##############
    mc_info[:,0] = (mc_info[:,0]-90)/45
    mc_info[:,1] = (mc_info[:,1])/10
    mc_info[:,2] = (mc_info[:,2])/100
    first, mc_info = shuffle(first, mc_info, random_state=0)
    '''


    logger.info('Building discriminator')

    sizes = [ 30, 30, 29, 1]
    calorimeter = Input(shape=sizes)

    input_Info = Input(shape=(4, )) ## scaled input !!

    #features =build_discriminator_3D(image=calorimeter, mbd=True, sparsity=False, sparsity_mbd=False)
    #features =build_discriminator_3D(image=calorimeter, mbd=False, sparsity=False, sparsity_mbd=False)
    #features =build_discriminator_3D_v1(image=calorimeter, mbd=False, sparsity=False, sparsity_mbd=False)
    #features =build_discriminator_3D_v2(image=calorimeter, epsilon=0.001)
    features =build_discriminator_3D_v3(image=calorimeter, info=input_Info, epsilon=0.001)
    print('features:',features.shape)
    #energies = calculate_energy(calorimeter)
    #print('energies:',energies.shape)

    ## construct MBD on the raw energies
    #nb_features = 10
    #vspace_dim = 10
    #minibatch_featurizer = Lambda(minibatch_discriminator,
    #                              output_shape=minibatch_output_shape)
    #K_energy = Dense3D(nb_features, vspace_dim)(energies)

    ## constrain w/ a tanh to dampen the unbounded nature of energy-space
    #mbd_energy = Activation('tanh')(minibatch_featurizer(K_energy))

    ## absolute deviation away from input energy. Technically we can learn
    ## this, but since we want to get as close as possible to conservation of
    ## energy, just coding it in is better
    #energy_well = Lambda(
    #    lambda x: K.abs(x[0] - x[1])
    #)([total_energy, input_energy])

    ## binary y/n if it is over the input energy
    #well_too_big = Lambda(lambda x: 10 * K.cast(x > 5, K.floatx()))(energy_well)

    #p = features
    #p = concatenate([features, energies])
    '''
    p = concatenate([
        features,
        scale(energies, 10),
        scale(total_energy, 100),
        energy_well,
        well_too_big,
        mbd_energy
    ])
    '''
    
    #print('features shape:', features.shape)
    fake = Dense(1, activation='sigmoid', name='fakereal_output')(features)
    #reg = build_regression(p)
    #print('fake shape:', fake.shape)
    #print('reg shape:', reg.shape)
    #discriminator_outputs = [fake, total_energy]
    discriminator_outputs = fake
    #discriminator_outputs = [fake]+reg
    #print('discriminator_outputs shape:', discriminator_outputs.shape)
    discriminator_losses = 'binary_crossentropy'

    #discriminator = Model(calorimeter, discriminator_outputs, name='discriminator')
    discriminator = Model([calorimeter, input_Info] , discriminator_outputs, name='discriminator')
    #print('discriminator check:', len(set(discriminator.inputs)), len(discriminator.inputs))

    discriminator.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
    )

    logger.info('Building generator')

    latent = Input(shape=(latent_size, ), name='z')
    input_info = Input(shape=(4, ), dtype='float32')
    generator_inputs = [latent, input_info]

    h = concatenate([latent, input_info])
    #img_layer  = build_generator_3D(h, 30 , 30 , 29)
    img_layer  = build_generator_3D_v1(h, 30 , 30 , 29)
    print('img_layer shape:',img_layer.shape)
    output_info = Lambda(lambda x: x)(input_info) # same as input
    generator_outputs =  [img_layer ,output_info]
    #generator_outputs =  img_layer 
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

    combined_outputs = [discriminator( generator(generator_inputs)), reg_model((generator(generator_inputs))[0]) ]
    #combined_outputs = [discriminator( generator(generator_inputs)), reg_model(generator(generator_inputs)) ]
    print('h31')
    combined_losses = ['binary_crossentropy', 'mae']

    combined = Model(generator_inputs, combined_outputs, name='combined_model')
    print('h4')
    combined.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss=combined_losses
    )

    logger.info('commencing training')
    f_DataSet = open(parse_args.datafile, 'r')
    Data = []
    Event = []
    Batch = []
    for line in f_DataSet: 
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
 
    #disc_outputs_real = np.ones(batch_size)
    #disc_outputs_real = np.ones(batch_size)*0.95
    #disc_outputs_fake = np.zeros(batch_size)
    tmp_real = 0.5
    tmp_fake = 0.5
    loss_weights      = np.ones(batch_size)
    combined_loss_weights      = [np.ones(batch_size), 1*np.ones(batch_size)]

    for epoch in range(nb_epochs):
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = sum(Batch)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        processed_batch =  0
        disc_outputs_real = np.ones (batch_size)*tmp_real
        #disc_outputs_fake = np.zeros(batch_size)*tmp_fake
        disc_outputs_fake = np.ones(batch_size)*tmp_fake
        tmp_real = tmp_real + 0.05 if tmp_real<=0.95 else tmp_real
        tmp_fake = tmp_fake - 0.05 if tmp_fake>=0.05 else tmp_fake
        for ib in range(len(Batch)):
            first, mc_info, mc_info_v1 = load_data(Data[ib])
            ibatch = Batch[ib]
            for index in range(ibatch):
                if verbose:
                    progress_bar.update(index)
                else:
                    if index % 100 == 0:
                        logger.info('processed {}/{} batches'.format(index + 1, ibatch))
                    elif index % 10 == 0:
                        logger.debug('processed {}/{} batches'.format(index + 1, ibatch))

                # generate a new batch of noise
                noise = np.random.normal(0, 1, (batch_size, latent_size))

                # get a batch of real images
                image_batch   =  first     [index * batch_size:(index + 1) * batch_size]
                info_batch    =  mc_info   [index * batch_size:(index + 1) * batch_size]
                info_batch_v1 =  mc_info_v1[index * batch_size:(index + 1) * batch_size]
                '''
                sampled_energies = np.random.uniform(0.1, 1, (batch_size, 1))
                sampled_theta    = np.random.uniform(-1 , 1, (batch_size, 1))
                sampled_phi      = np.random.uniform(-1 , 1, (batch_size, 1))
                sampled_info     = np.concatenate((sampled_theta, sampled_phi, sampled_energies),axis=-1)
                '''
                sampled_mom  = (np.random.uniform(10, 100, (batch_size, 1))-55)/45
                sampled_dphi = np.random.uniform(-15 ,17 , (batch_size, 1))/20
                #sampled_dtheta = (np.random.uniform(40 ,140 , (batch_size, 1))-90)/50
                #sampled_z      = np.random.uniform(-2000 ,2000 , (batch_size, 1))/2000
                sampled_dtheta = np.expand_dims(info_batch_v1[:,1], -1)
                sampled_z      = np.expand_dims(info_batch_v1[:,3], -1)
                sampled_info   = np.concatenate((sampled_mom, sampled_dtheta, sampled_dphi, sampled_z),axis=-1)
                

                generator_inputs = [noise, sampled_info]
                #generator_inputs = [noise]
                generated_images = generator.predict(generator_inputs, verbose=0)

                #disc_outputs_real = [np.ones(batch_size), info_batch]

                # downweight the energy reconstruction loss ($\lambda_E$ in paper)
                real_batch_loss = discriminator.train_on_batch(
                    [image_batch, info_batch],
                    #image_batch, 
                    disc_outputs_real,
                    loss_weights
                )
                #print('real_batch_loss=',real_batch_loss)
                # note that a given batch should have either *only* real or *only* fake,
                # as we have both minibatch discrimination and batch normalization, both
                # of which rely on batch level stats
                fake_batch_loss = discriminator.train_on_batch(
                    generated_images, 
                    disc_outputs_fake,
                    loss_weights          ##should we put reg here also?
                )
                '''
                print('fake_batch_loss=',fake_batch_loss)
                '''
                if index == (ibatch-1):
                    #real_pred = discriminator.predict_on_batch(image_batch)
                    real_pred = discriminator.predict_on_batch([image_batch, info_batch])
                    fake_pred = discriminator.predict_on_batch(generated_images)
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
            #for _ in range(20):
            max_iter = 2
            N_iter = 0
            while True:
                N_iter = N_iter+1
                noise            = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_mom  = (np.random.uniform(10, 100, (batch_size, 1))-55)/45
                sampled_dphi = np.random.uniform(-15 ,17 , (batch_size, 1))/20
                sampled_dtheta = np.expand_dims(info_batch_v1[:,1], -1)
                sampled_z      = np.expand_dims(info_batch_v1[:,3], -1)
                #sampled_dtheta = (np.random.uniform(40 ,140 , (batch_size, 1))-90)/50
                #sampled_z      = np.random.uniform(-2000 ,2000 , (batch_size, 1))/2000
                sampled_info   = np.concatenate((sampled_mom, sampled_dtheta, sampled_dphi, sampled_z),axis=-1)
                #sampled_info   = info_batch_v1
                '''
                sampled_energies = np.random.uniform(0.1, 1, (batch_size, 1))
                sampled_theta    = np.random.uniform(-1 , 1, (batch_size, 1))
                sampled_phi      = np.random.uniform(-1 , 1, (batch_size, 1))
                sampled_info     = np.concatenate((sampled_theta, sampled_phi, sampled_energies),axis=-1)
                '''

                combined_inputs  = [noise, sampled_info]
                combined_outputs = [np.ones(batch_size), sampled_info]
                #combined_outputs = [trick]
                loss_gen = combined.train_on_batch(
                    combined_inputs,
                    combined_outputs,
                    combined_loss_weights
                )
                gen_losses.append(loss_gen)
                if (loss_gen[1] < 1 and N_iter>=2) or N_iter > max_iter: break

            epoch_gen_loss.append(np.mean(np.array(gen_losses), axis=0))
            processed_batch = processed_batch + ibatch
            logger.info('processed {}/{} total batches'.format(processed_batch, nb_batches))

        logger.info('Epoch {:3d} Generator loss: {}'.format(
            epoch + 1, np.mean(epoch_gen_loss, axis=0)))
        logger.info('Epoch {:3d} Discriminator loss: {}'.format(
            epoch + 1, np.mean(epoch_disc_loss, axis=0)))

    # save weights every epoch
    '''
    generator    .save_weights(parse_args.gen_weight_out, overwrite=True)
    discriminator.save_weights(parse_args.dis_weight_out, overwrite=True)
    combined     .save_weights(parse_args.comb_weight_out, overwrite=True)
    gen_yaml_string = generator.to_yaml()
    dis_yaml_string = discriminator.to_yaml()
    comb_yaml_string = combined.to_yaml()
    open(parse_args.gen_model_out, 'w').write(gen_yaml_string)
    open(parse_args.dis_model_out, 'w').write(dis_yaml_string)
    open(parse_args.comb_model_out, 'w').write(comb_yaml_string)
    '''
    generator.save(parse_args.gen_out)
    discriminator.save(parse_args.dis_out)
    combined .save(parse_args.comb_out)
    print('done reg training')
