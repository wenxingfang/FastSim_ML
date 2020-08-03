# -*- coding: utf-8 -*-
""" 
file: train_v1.py
##########################################################
Input image is theta(180)*phi(360) for time and nPE of PMT
Add regression part and use full dataset
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
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/models/')
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

def load_data(datafile):

    d = h5py.File(datafile, 'r')
    first = np.expand_dims(d['firstHitTimeByPMT'][:], -1)
    second = np.expand_dims(d['nPEByPMT'][:], -1)
    infoMuon = d['infoMuon'][:,:4]
    sizes = [ first.shape[1], first.shape[2], second.shape[1], second.shape[2] ]
    d.close()
    #print('infoMuon dtype',infoMuon.dtype)
    #infoMuon = infoMuon.astype(float)
    #print('infoMuon dtype',infoMuon.dtype)
    
    ###  normalize muon info ##############
    infoMuon[:,0]=(infoMuon[:,0])/math.pi #(0 to 1)
    infoMuon[:,1]=(infoMuon[:,1])/math.pi #(-1 to 1)
    infoMuon[:,2]=(infoMuon[:,2])/math.pi #(0 to 1)
    infoMuon[:,3]=(infoMuon[:,3])/math.pi #(-1 to 1)
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

    parser.add_argument('--model-out', action='store',type=str,
                        default='',
                        help='output of trained model')
    parser.add_argument('--weight-out', action='store',type=str,
                        default='',
                        help='output of trained weight')
    parser.add_argument('--model-all-out', action='store',type=str,
                        default='',
                        help='output of trained model')

    parser.add_argument('--datafile', action='store', type=str,
                        help='HDF5 file paths')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()

    # delay the imports so running train.py -h doesn't take 5,234,807 years
    import keras.backend as K
    import tensorflow as tf
    #session_conf = tf.ConfigProto()
    #session_conf.gpu_options.allow_growth = True
    #session = tf.Session(config=session_conf)
    #K.set_session(session)
    from keras.layers import (Activation, AveragePooling2D, Dense, Embedding,
                              Flatten, Input, Lambda, UpSampling2D)
    from keras.layers.merge import add, concatenate, multiply
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.utils.generic_utils import Progbar
    #from keras.utils.vis_utils     import plot_model
    K.set_image_dim_ordering('tf')

    from ops import (calculate_energy, scale)

    from architectures import build_regression, build_regression_v1, build_regression_v2, build_regression_v3

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
    logger.info('Building regression')

    calorimeter = [Input(shape=sizes[:2] + [1]),
                   Input(shape=sizes[2:4] + [1])]

    '''
    features = []
    for l in range(2):
        # build features per layer of calorimeter
        #features.append(build_regression(image=calorimeter[l]))
        features.append(build_regression_v1(image=calorimeter[l]))
        #features.append(build_regression_v2(image=calorimeter[l]))
        #features.append(build_regression_v3(image=calorimeter[l]))
        
    #input_energy = Input(shape=(1, ))
    #energies = calculate_energy(calorimeter[1])# get the total nPE for each event!
    #energies = scale(energies,10000)
    features = concatenate(features)
    print('features:',features.shape)
    '''
    features = build_regression_v3(images=[calorimeter[0], calorimeter[1]], epsilon=0.001)

    reg_ptheta  = Dense(1, activation='sigmoid',  name='ptheta_info_output')(features)
    reg_pphi    = Dense(1, activation='tanh'   ,  name='pphi_info_output')(features)
    reg_rtheta  = Dense(1, activation='sigmoid',  name='rtheta_info_output')(features)
    reg_rphi    = Dense(1, activation='tanh'   ,  name='rphi_info_output')(features)
    reg         = concatenate([reg_ptheta, reg_pphi, reg_rtheta, reg_rphi])
    print('reg shape:',reg.shape)
    regression_outputs = reg
    regression_losses = 'mae'
    #regression_losses = 'mse'

    regression = Model(calorimeter, regression_outputs)

    regression.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=regression_losses
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
    for epoch in range(nb_epochs):
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = sum(Batch)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_loss = []
        processed_batch = 0
        for ib in range(len(Batch)):
            first, second, infoMuon = load_data(Data[ib])
            ibatch = Batch[ib]
            for index in range(ibatch):
                if verbose:
                    progress_bar.update(index)
                else:
                    if index % 100 == 0:
                        logger.info('processed {}/{} batches'.format(index + 1 , ibatch))
                    elif index % 10 == 0:
                        logger.debug('processed {}/{} batches'.format(index + 1, ibatch))
   
                # get a batch of real images
                image_batch_1 = first [index * batch_size:(index + 1) * batch_size]
                image_batch_2 = second[index * batch_size:(index + 1) * batch_size]
                info_batch  = infoMuon[index * batch_size:(index + 1) * batch_size]

                real_batch_loss = regression.train_on_batch(
                    [image_batch_1, image_batch_2],
                    info_batch
                )
                #print('index=',index,'batch loss=',real_batch_loss)
                if math.isnan(real_batch_loss):continue
                #epoch_loss.append(np.array(real_batch_loss))
                epoch_loss.append(real_batch_loss)
            processed_batch = processed_batch + ibatch
            logger.info('processed {}/{} total batches'.format(processed_batch, nb_batches))
        #logger.info('Epoch {:3d} loss: {}'.format( epoch + 1, np.mean(epoch_loss, axis=0)))
        if len(epoch_loss)==0:print(epoch_loss)
        else:
            logger.info('Epoch {:3d} loss: {}'.format( epoch + 1, float(sum(epoch_loss))/len(epoch_loss)))
    # save weights and model
    regression.save_weights(parse_args.weight_out, overwrite=True)
    yaml_string = regression.to_yaml()
    open(parse_args.model_out, 'w').write(yaml_string)
    regression.save(parse_args.model_all_out) 
    print('done')
