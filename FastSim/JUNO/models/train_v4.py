# -*- coding: utf-8 -*-
""" 
file: train_v1.py
##########################################################
Input image is theta(180)*phi(360) for time and nPE of PMT
Add regression part and use more dataset
###########################################################
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
import yaml
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model


if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)


def load_data(data):

    d = h5py.File(data, 'r')
    first = np.expand_dims(d['firstHitTimeByPMT'][:], -1)
    second = np.expand_dims(d['nPEByPMT'][:], -1)
    infoMuon = d['infoMuon'][:,:4]
    sizes = [ first.shape[1], first.shape[2], second.shape[1], second.shape[2] ]
    d.close()
    ###  normalize muon info ##############
    infoMuon[:,0]=(infoMuon[:,0])/math.pi
    infoMuon[:,1]=(infoMuon[:,1])/math.pi
    infoMuon[:,2]=(infoMuon[:,2])/math.pi
    infoMuon[:,3]=(infoMuon[:,3])/math.pi
    #infoMuon[:,4]=(infoMuon[:,4])/18000#17700.0
    '''
    first_mean = np.mean(first, axis=(1,2), keepdims=True)
    first_mean = first_mean.repeat(sizes[0],axis=1)
    first_mean = first_mean.repeat(sizes[1],axis=2)
    second_mean = np.mean(second, axis=(1,2), keepdims=True)
    second_mean = second_mean.repeat(sizes[0],axis=1)
    second_mean = second_mean.repeat(sizes[1],axis=2)
    first_std = np.std(first, axis=(1,2), keepdims=True)
    first_std = first_std.repeat(sizes[0],axis=1)
    first_std = first_std.repeat(sizes[1],axis=2)
    second_std = np.std(second, axis=(1,2), keepdims=True)
    second_std = second_std.repeat(sizes[0],axis=1)
    second_std = second_std.repeat(sizes[1],axis=2)
    first = (first - first_mean)/first_std
    second = (second - second_mean)/second_std
    '''
    print("first:",first.shape,", second:", second.shape,  ",info:", infoMuon.shape, ",sizes:", sizes)
    #print("first:",first[0,:,:,0],", second:", second[0,:,:,0],  ",info:", infoMuon[0:10])
    first, second, infoMuon = shuffle(first, second, infoMuon, random_state=0)
    return first, second, infoMuon



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

    parser.add_argument('--gen-weight-out', action='store', type=str,
                        default='',
                        help='output of gen weights')
    parser.add_argument('--gen-str-out', action='store', type=str,
                        default='',
                        help='output of gen structure')
    parser.add_argument('--gen-model-out', action='store', type=str,
                        default='',
                        help='output of whole gen model')
    parser.add_argument('--comb-weight-out', action='store', type=str,
                        default='',
                        help='output of combined weights')
    parser.add_argument('--comb-str-out', action='store', type=str,
                        default='',
                        help='output of combined structure')
    parser.add_argument('--comb-model-out', action='store', type=str,
                        default='',
                        help='output of whole combined model')
    parser.add_argument('--dis-model-out', action='store', type=str,
                        default='',
                        help='output of dis model')
    parser.add_argument('--reg-weight-in', action='store', type=str,
                        default='',
                        help='input of reg weights')
    parser.add_argument('--reg-str-in', action='store', type=str,
                        default='',
                        help='input of reg structure')
    parser.add_argument('--reg-model-in', action='store', type=str,
                        default='',
                        help='input of reg model')

    parser.add_argument('--datafile-temp', action='store', type=str,
                        help='HDF5 template file paths')
    parser.add_argument('--datafile', action='store', type=str,
                        help='HDF5 file paths')

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

    from architectures import build_generator, build_discriminator, build_generator_v1, build_generator_v2,build_generator_v3,build_generator_v4, build_discriminator_v1, build_discriminator_v2

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

    datafile = parse_args.datafile
    datafile_temp = parse_args.datafile_temp

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
    # read in data file spec from YAML file
    with open(yaml_file, 'r') as stream:
        try:
            s = yaml.load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise exc
    nb_classes = len(s.keys())
    logger.info('{} particle types found.'.format(nb_classes))
    for name, pth in s.items():
        logger.debug('class {} <= {}'.format(name, pth))
    '''

    sizes =[180,360, 180,360]## H and W of input image
    logger.info('Building discriminator')

    input_Info = Input(shape=(4, ), dtype='float32')
    calorimeter = [Input(shape=sizes[:2] + [1]),
                   Input(shape=sizes[2:4] + [1])]

    #input_energy = Input(shape=(1, ))

    '''
    features = []
    for l in range(2):
        # build features per layer of calorimeter
        features.append(build_discriminator(
            image=calorimeter[l],
            mbd=False,
            sparsity=False,
            sparsity_mbd=False
        ))
    #energies = calculate_energy(calorimeter[1])# get the total nPE for each event!
    #energies = scale(energies,10000)
    print('feature0:',features[0].shape)
    print('feature1:',features[1].shape)
    features = concatenate(features)
    '''
    #features=build_discriminator_v1([ calorimeter[0], calorimeter[1] ]) 
    features=build_discriminator_v2(calorimeter, input_Info)
    print('features:',features.shape)

    '''
    # construct MBD on the raw energies
    nb_features = 10
    vspace_dim = 10
    minibatch_featurizer = Lambda(minibatch_discriminator,
                                  output_shape=minibatch_output_shape)
    K_energy = Dense3D(nb_features, vspace_dim)(energies)

    # constrain w/ a tanh to dampen the unbounded nature of energy-space
    mbd_energy = Activation('tanh')(minibatch_featurizer(K_energy))

    # absolute deviation away from input energy. Technically we can learn
    # this, but since we want to get as close as possible to conservation of
    # energy, just coding it in is better
    energy_well = Lambda(
        lambda x: K.abs(x[0] - x[1])
    )([total_energy, input_energy])

    # binary y/n if it is over the input energy
    well_too_big = Lambda(lambda x: 10 * K.cast(x > 5, K.floatx()))(energy_well)
    '''
    #p = concatenate([features,energies])
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
    fake = Dense(1, activation='sigmoid', name='fakereal_output')(features)
    #reg  = Dense(5,                       name='muon_info_output')(p)
    #discriminator_outputs = [fake, total_energy]
    discriminator_outputs = fake
    #discriminator_losses = ['binary_crossentropy', 'mae']
    discriminator_losses = 'binary_crossentropy'
    # ACGAN case
    '''
    if nb_classes > 1:
        logger.info('running in ACGAN for discriminator mode since found {} '
                    'classes'.format(nb_classes))

        aux = Dense(1, activation='sigmoid', name='auxiliary_output')(p)
        discriminator_outputs.append(aux)

        # change the loss depending on how many outputs on the auxiliary task
        if nb_classes > 2:
            discriminator_losses.append('sparse_categorical_crossentropy')
        else:
            discriminator_losses.append('binary_crossentropy')
    '''

    #discriminator = Model(calorimeter, discriminator_outputs, name='discriminator')
    discriminator = Model(calorimeter+ [input_Info], discriminator_outputs, name='discriminator')
    #print('discriminator check:', len(set(discriminator.inputs)), len(discriminator.inputs))

    discriminator.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
    )

    logger.info('Building generator')

    latent = Input(shape=(latent_size, ), name='z')
    layer0_T0 = Input(shape=(180, 360,1), name='l0_T0')
    layer0_T1 = Input(shape=(180, 360,1), name='l0_T1')
    layer1_T0 = Input(shape=(180, 360,1), name='l1_T0')
    layer1_T1 = Input(shape=(180, 360,1), name='l1_T1')
    input_info = Input(shape=(4, ), dtype='float32')
    #generator_inputs = [latent]
    generator_inputs = [latent, layer0_T0, layer0_T1, layer1_T0, layer1_T1, input_info]

    # ACGAN case
    '''
    if nb_classes > 1:
        logger.info('running in ACGAN for generator mode since found {} '
                    'classes'.format(nb_classes))

        # label of requested class
        image_class = Input(shape=(1, ), dtype='int32')
        lookup_table = Embedding(nb_classes, latent_size, input_length=1,
                                 embeddings_initializer='glorot_normal')
        emb = Flatten()(lookup_table(image_class))

        # hadamard product between z-space and a class conditional embedding
        hc = multiply([latent, emb])

        # requested energy comes in GeV
        h = Lambda(lambda x: x[0] * x[1])([hc, scale(input_energy, 100)])
        generator_inputs.append(image_class)
    else:
        # requested energy comes in GeV
        h = Lambda(lambda x: x[0] * x[1])([latent, scale(input_energy, 100)])
    '''
    h = concatenate([latent,input_info])
    # each of these builds a LAGAN-inspired [arXiv/1701.05927] component with
    # linear last layer
    print('h1')
    #img_layer0 = build_generator_v2(h, 180, 360, layer0_T0, layer0_T1, 1e-2)
    #img_layer1 = build_generator_v2(h, 180, 360, layer1_T0, layer1_T1, 1e-3)
    img_layer1 = build_generator_v3(h, 180, 360, layer1_T0, layer1_T1, 1e-3)
    #img_layer0 = build_generator_v3(h, 180, 360, layer0_T0, layer0_T1, 1e-2)
    img_layer0 = build_generator_v4(img_layer1, 180, 360, layer0_T0, layer0_T1, 1e-2)
    print('h2')
    '''
    if not no_attn:

        logger.info('using attentional mechanism')

        # resizes from (3, 96) => (12, 12)
        zero2one = AveragePooling2D(pool_size=(1, 8))(
            UpSampling2D(size=(4, 1))(img_layer0))
        img_layer1 = inpainting_attention(img_layer1, zero2one)

        # resizes from (12, 12) => (12, 6)
        one2two = AveragePooling2D(pool_size=(1, 2))(img_layer1)
        img_layer2 = inpainting_attention(img_layer2, one2two)
    '''
    #generator_outputs = [
    #    Activation('relu')(img_layer0),
    #    Activation('relu')(img_layer1)
    #]
    output_info = Lambda(lambda x: x*math.pi)(input_info) # to be original value
    generator_outputs = [
        img_layer0,
        img_layer1,
        output_info
    ]

    #############
    d_temp = h5py.File(datafile_temp, 'r')
    temp_time_de = np.expand_dims(d_temp['temp1_firstHitTimeByPMT'][0:1], -1)## default and 0
    temp_time_01 = np.expand_dims(d_temp['temp2_firstHitTimeByPMT'][0:1], -1)## 1 and 0
    temp_nPE_de = np.expand_dims(d_temp['temp1_nPEByPMT'][0:1], -1)
    temp_nPE_01 = np.expand_dims(d_temp['temp2_nPEByPMT'][0:1], -1)
    temp_time_de_ = temp_time_de.repeat(batch_size,axis=0)
    temp_time_01_ = temp_time_01.repeat(batch_size,axis=0)
    temp_nPE_de_  = temp_nPE_de.repeat(batch_size,axis=0)
    temp_nPE_01_  = temp_nPE_01.repeat(batch_size,axis=0)
    Temps_input = [temp_time_de_, temp_time_01_, temp_nPE_de_, temp_nPE_01_]
    d_temp.close()
    #############

    generator = Model(generator_inputs, generator_outputs, name='generator')
    #print('generator check:', len(set(generator_inputs)), len(generator_inputs))
    #generator = Model(generator_inputs, generator_outputs_new)

    generator.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    print('h3')
######### regression part ##########################
    '''
    ## raise error, not clear
    reg_model = model_from_yaml(open(parse_args.reg_str_in).read())
    reg_model.load_weights(parse_args.reg_weight_in)
    '''
    reg_model = load_model(parse_args.reg_model_in, custom_objects={'tf': tf})
    reg_model.trainable = False
    reg_model.name  = 'regression'
    reg_model.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
###################################


#############################
    # build combined model
    # we only want to be able to train generation for the combined model
    discriminator.trainable = False

    combined_outputs = [ discriminator( generator(generator_inputs) ), reg_model( (generator(generator_inputs))[:2] ) ]
    print('h31')

    combined = Model(generator_inputs, combined_outputs, name='combined_model')
    combined_losses = ['binary_crossentropy', 'mae']
    print('h4')
    combined.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss=combined_losses
    )

    logger.info('commencing training')
    f_DataSet = open(datafile, 'r')
    Data = []
    Event = []
    Batch = []
    for line in f_DataSet: 
        (idata, ievent) = line.split()
        if "#" in idata: continue ##skip the commented one
        Data.append(idata)
        Event.append(float(ievent))
        Batch.append(int(float(ievent)/batch_size))
    total_event = sum(Event)
    f_DataSet.close() 
    print('total sample:', total_event)
    print('All Batch:', Batch)
    disc_outputs_real = np.ones(batch_size)
    disc_outputs_fake = np.zeros(batch_size)
    loss_weights_real = np.ones(batch_size) 
    loss_weights_fake = np.ones(batch_size) 
    loss_weights_combine = [np.ones(batch_size), 1*np.ones(batch_size)] 
 
    for epoch in range(nb_epochs):
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = sum(Batch)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []
        processed_batch = 0
        for ib in range(len(Batch)):
            first, second, infoMuon = load_data(Data[ib])
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
                image_batch_1 = first [index * batch_size:(index + 1) * batch_size]
                image_batch_2 = second[index * batch_size:(index + 1) * batch_size]
                info_batch  = infoMuon[index * batch_size:(index + 1) * batch_size]
                # sampling
                sampled_ptheta   = np.random.uniform(0   , 1, (batch_size, 1))
                sampled_pphi     = np.random.uniform(-1  , 1, (batch_size, 1))
                sampled_rtheta   = np.random.uniform(0   , 1, (batch_size, 1))
                sampled_rphi     = np.random.uniform(-1  , 1, (batch_size, 1))
                sampled_info     = np.concatenate((sampled_ptheta, sampled_pphi, sampled_rtheta, sampled_rphi),axis=-1)

                generator_inputs = [noise]+Temps_input+[sampled_info]
                generated_images = generator.predict(generator_inputs, verbose=0)

                real_batch_loss = discriminator.train_on_batch(
                    [image_batch_1, image_batch_2, info_batch],
                    disc_outputs_real,
                    loss_weights_real
                )
                # note that a given batch should have either *only* real or *only* fake,
                # as we have both minibatch discrimination and batch normalization, both
                # of which rely on batch level stats
                fake_batch_loss = discriminator.train_on_batch(
                    generated_images ,
                    disc_outputs_fake,
                    loss_weights_fake
                )

                epoch_disc_loss.append( (np.array(fake_batch_loss) + np.array(real_batch_loss)) / 2)

                # we want to train the genrator to trick the discriminator
                # For the generator, we want all the {fake, real} labels to say
                # real

                gen_losses = []

                # we do this twice simply to match the number of batches per epoch used to
                # train the discriminator
                for _ in range(4):
                    noise = np.random.normal(0, 1, (batch_size, latent_size))
                    sampled_ptheta   = np.random.uniform(0   , 1, (batch_size, 1))
                    sampled_pphi     = np.random.uniform(-1  , 1, (batch_size, 1))
                    sampled_rtheta   = np.random.uniform(0   , 1, (batch_size, 1))
                    sampled_rphi     = np.random.uniform(-1  , 1, (batch_size, 1))
                    sampled_info     = np.concatenate((sampled_ptheta, sampled_pphi, sampled_rtheta, sampled_rphi),axis=-1)

                    combined_inputs = [noise]+Temps_input+[sampled_info]
                    combined_outputs = [np.ones(batch_size), sampled_info]
                    gen_losses.append(combined.train_on_batch(
                        combined_inputs,
                        combined_outputs,
                        loss_weights_combine
                    ))

                epoch_gen_loss.append(np.mean(np.array(gen_losses), axis=0))

        logger.info('Epoch {:3d} Generator loss: {}'.format(
            epoch + 1, np.mean(epoch_gen_loss, axis=0)))
        logger.info('Epoch {:3d} Discriminator loss: {}'.format(
            epoch + 1, np.mean(epoch_disc_loss, axis=0)))

    # save 
    '''
    generator.save_weights(parse_args.gen_weight_out , overwrite=True)
    combined .save_weights(parse_args.comb_weight_out, overwrite=True)
    gen_yaml_string  = generator.to_yaml()
    comb_yaml_string = combined.to_yaml()
    open(parse_args.gen_str_out, 'w').write(gen_yaml_string)
    open(parse_args.comb_str_out, 'w').write(comb_yaml_string)
    '''
    generator.save(parse_args.gen_model_out) 
    discriminator.save(parse_args.dis_model_out) 
    combined .save(parse_args.comb_model_out) 
    print('done')
