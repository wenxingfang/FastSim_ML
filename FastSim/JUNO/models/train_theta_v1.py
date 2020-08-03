# -*- coding: utf-8 -*-
""" 
file: train_theta_v1.py
##########################################################
Input is :17613 array for time and nPE of large PMT
###########################################################
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

def binary_crossentropy(target, output):
    output = -target * np.log(output) - (1.0 - target) * np.log(1.0 - output)
    return output


def load_data(data):
    print('load data:',data)
    d = h5py.File(data, 'r')
    first  = d['firstHitTimeByPMT'][:]
    second = d['nPEByPMT'][:]
    infoMC = d['infoMC'][:,:]
    infoPMT = d['infoPMT'][:,:]
    sizes = [ first.shape[1], second.shape[1] ]
    d.close()
    ###  normalize muon info ##############
    #infoMC[:,0]=(infoMC[:,0])/math.pi
    #infoMC[:,1]=(infoMC[:,1])/math.pi
    #infoMC[:,2]=(infoMC[:,2])/math.pi
    #infoMC[:,3]=(infoMC[:,3])/math.pi
    #infoMC[:,4]=(infoMC[:,4])/18000#17700.0
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
    print("first:",first.shape,", second:", second.shape,  ",infoPMT:", infoPMT.shape, ",sizes:", sizes)
    first, second = shuffle(first, second)
    return first, second, infoPMT



def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x

def bit_flip_v1(x, prob, target):
    """ flips a int array's values to target with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = target
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

    from architectures import build_generator, build_discriminator, build_generator_v1, build_generator_v2,build_generator_v3,build_generator_v4, build_discriminator_v1, build_discriminator_v2,build_discriminator_v3, build_discriminator_v5, build_generator_v7, build_generator_v8

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

    sizes =[124,360, 124,360]## H and W of input image
    logger.info('Building discriminator')

    input_Info  = Input(shape=(1, )   ,dtype='float32')
    calorimeter = Input(shape=(1, )   ,dtype='float32')

    #features=build_discriminator_v1([ calorimeter[0], calorimeter[1] ]) 
    #features=build_discriminator_v2(calorimeter, input_Info)
    #features=build_discriminator_v3(calorimeter, 0.001)
    #features=build_discriminator_v4(calorimeter, 0.001)
    #features=build_discriminator_v5(calorimeter, 0.001)
    features=build_discriminator_v6([calorimeter, input_Info], 0.001)
    #nPE     = calculate_energy(calorimeter) # get the total nPE for each event!
    print('features:',features.shape)
    #p = concatenate([features,nPE])
    p = features
    fake = Dense(1, activation='sigmoid', name='fakereal_output')(p)
    discriminator_outputs = fake
    discriminator_losses = 'binary_crossentropy'
    discriminator = Model(calorimeter, discriminator_outputs, name='discriminator')
    #print('discriminator check:', len(set(discriminator.inputs)), len(discriminator.inputs))
    discriminator.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
    )

    logger.info('Building generator')

    latent = Input(shape=(latent_size, ), name='latent')
    layer0_T0 = Input(shape=(180, 360,1), name='l0_T0')
    layer0_T1 = Input(shape=(180, 360,1), name='l0_T1')
    layer1_T0 = Input(shape=(180, 360,1), name='l1_T0')
    layer1_T1 = Input(shape=(180, 360,1), name='l1_T1')
    input_info = Input(shape=(4, ), dtype='float32')
    generator_inputs = latent
    #generator_inputs = [latent, layer0_T0, layer0_T1, layer1_T0, layer1_T1, input_info]
    #h = concatenate([latent,input_info])
    #img_layer1 = build_generator_v3(h, 180, 360, layer1_T0, layer1_T1, 1e-3)
    #img_layer0 = build_generator_v4(img_layer1, 180, 360, layer0_T0, layer0_T1, 1e-2)
    #img_layer0 = build_generator_v5(generator_inputs, 124, 360)
    #img_layer0 = build_generator_v7(generator_inputs)
    img_layer0 = build_generator_v8(generator_inputs)
    print('h2')
    #output_info = Lambda(lambda x: x*math.pi)(input_info) # to be original value
    #generator_outputs = [
    #    img_layer0,
    #    img_layer1,
    #    output_info
    #]
    generator_outputs = img_layer0
    #############
    '''
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
    '''
    #############

    generator = Model(generator_inputs, generator_outputs, name='generator')

    generator.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    print('h3')
######### regression part ##########################
    '''
    reg_model = load_model(parse_args.reg_model_in, custom_objects={'tf': tf})
    reg_model.trainable = False
    reg_model.name  = 'regression'
    reg_model.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    '''
###################################


#############################
    # build combined model
    # we only want to be able to train generation for the combined model
    discriminator.trainable = False

    #combined_outputs = [ discriminator( generator(generator_inputs) ), reg_model( (generator(generator_inputs))[:2] ) ]
    combined_outputs =  discriminator( generator(generator_inputs) )
    print('h31')

    combined = Model(generator_inputs, combined_outputs, name='combined_model')
    #combined_losses = ['binary_crossentropy', 'mae']
    combined_losses = 'binary_crossentropy'
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
        idata = line.strip('\n')
        idata = idata.strip(' ')
        if "#" in idata: continue ##skip the commented one
        Data.append(idata)
        print(idata)
        d = h5py.File(str(idata), 'r')
        ievent   = d['infoMC'].shape[0]
        d.close()
        Event.append(float(ievent))
        Batch.append(int(float(ievent)/batch_size))
    total_event = sum(Event)
    f_DataSet.close() 
    print('total sample:', total_event)
    print('All Batch:', Batch)
    loss_weights_real = np.ones(batch_size) 
    loss_weights_fake = np.ones(batch_size) 
    #loss_weights_combine = [np.ones(batch_size), 1*np.ones(batch_size)] 
    loss_weights_combine = np.ones(batch_size)
    flip_prob = 0.1
    for epoch in range(nb_epochs):
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = sum(Batch)
        if verbose:
            progress_bar = Progbar(target=nb_batches)
        flip_prob = flip_prob - 0.01 if (flip_prob - 0.01)>0 else 0.01
        epoch_gen_loss = []
        epoch_disc_loss = []
        processed_batch = 0
        disc_outputs_real = 0.95*np.ones(batch_size)
        disc_outputs_real= bit_flip_v1(disc_outputs_real, flip_prob, 0.01)
        disc_outputs_fake = 0.05*np.ones(batch_size)
        disc_outputs_fake= bit_flip_v1(disc_outputs_fake, flip_prob, 0.99)
        for ib in range(len(Batch)):
            first, second, infoMC = load_data(Data[ib])
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
                info_batch    = infoMC[index * batch_size:(index + 1) * batch_size]
                # sampling
                #sampled_ptheta   = np.random.uniform(0   , 1, (batch_size, 1))
                #sampled_pphi     = np.random.uniform(-1  , 1, (batch_size, 1))
                #sampled_rtheta   = np.random.uniform(0   , 1, (batch_size, 1))
                #sampled_rphi     = np.random.uniform(-1  , 1, (batch_size, 1))
                #sampled_info     = np.concatenate((sampled_ptheta, sampled_pphi, sampled_rtheta, sampled_rphi),axis=-1)
                #generator_inputs = [noise]+Temps_input+[sampled_info]
                #generated_images = generator.predict(generator_inputs, verbose=0)
                generator_inputs = noise
                generated_images = generator.predict(generator_inputs, verbose=0)
                image_batch_nos = 0
                for _ in range(1):
                    noise_for_real= np.random.uniform(0, 0.1, (batch_size, image_batch_2.shape[1]))
                    image_batch_nos = np.add(image_batch_2, noise_for_real)
                    real_batch_loss = discriminator.train_on_batch(
                        image_batch_nos,
                        disc_outputs_real
                    )
                fake_batch_loss = discriminator.train_on_batch(
                    generated_images ,
                    disc_outputs_fake
                )

                if index == (ibatch-1):
                    real_pred_0 = discriminator.predict_on_batch(image_batch_2)
                    real_pred = discriminator.predict_on_batch(image_batch_nos)
                    fake_pred = discriminator.predict_on_batch(generated_images)
                    print('real_pred 0:\n',real_pred_0)
                    print('real_pred:\n',real_pred)
                    print('fake_pred:\n',fake_pred)
                    binary_real = binary_crossentropy(disc_outputs_real, np.squeeze(real_pred))
                    binary_fake = binary_crossentropy(disc_outputs_fake, np.squeeze(fake_pred))
                    print('binary_crossentropy real shape=',binary_real.shape,',value=\n:', binary_real)
                    print('binary_crossentropy fake shape=',binary_fake.shape,',value=\n:', binary_fake)

                epoch_disc_loss.append( (np.array(fake_batch_loss) + np.array(real_batch_loss)) / 2)

                # we want to train the genrator to trick the discriminator
                # For the generator, we want all the {fake, real} labels to say
                # real

                gen_losses = []

                # we do this twice simply to match the number of batches per epoch used to
                # train the discriminator
                for _ in range(2):
                    noise = np.random.normal(0, 1, (batch_size, latent_size))
                    #sampled_ptheta   = np.random.uniform(0   , 1, (batch_size, 1))
                    #sampled_pphi     = np.random.uniform(-1  , 1, (batch_size, 1))
                    #sampled_rtheta   = np.random.uniform(0   , 1, (batch_size, 1))
                    #sampled_rphi     = np.random.uniform(-1  , 1, (batch_size, 1))
                    #sampled_info     = np.concatenate((sampled_ptheta, sampled_pphi, sampled_rtheta, sampled_rphi),axis=-1)

                    #combined_inputs = [noise]+Temps_input+[sampled_info]
                    combined_inputs = noise
                    #combined_outputs = [np.ones(batch_size), sampled_info]
                    #combined_outputs = 0.99*np.ones(batch_size)
                    combined_outputs = disc_outputs_real
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
        tmp_gen_out_name  = (parse_args.gen_model_out).replace('.h5','_epoch%d.h5'%epoch)
        tmp_dis_out_name  = (parse_args.dis_model_out).replace('.h5','_epoch%d.h5'%epoch)
        tmp_comb_out_name = (parse_args.comb_model_out).replace('.h5','_epoch%d.h5'%epoch)
        if (epoch+1)%10==0 or True :
            generator.save    (tmp_gen_out_name  )
            discriminator.save(tmp_dis_out_name  )
            combined .save    (tmp_comb_out_name )


    # save 
    generator.save(parse_args.gen_model_out) 
    discriminator.save(parse_args.dis_model_out) 
    combined .save(parse_args.comb_model_out) 
    print('done')
