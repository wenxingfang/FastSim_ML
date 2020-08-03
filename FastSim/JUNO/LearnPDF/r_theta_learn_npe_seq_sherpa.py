import h5py
from functools import partial
import sys
import argparse
import numpy as np
import math
import tensorflow as tf
#import tensorflow_probability as tfp
#import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random
import logging
import os
import ast
from tensorflow.python.framework import graph_util
#########################################################
import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.optimizers import Adam,SGD,RMSprop,Nadam
###########################
###########################
## assume npe is from possion distrbution
## for learning npe pdf , pdf(npe | r, theta)  #
## each h5 file has one npe array, different pmt for each row, first col is r, second col is theta, third is first event, fourth is second event and so on
## save produced result in one npe array
###########################

if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)

def del_empty(hd):## for hit time clean
    dele_list = []
    for i in range(hd.shape[0]):
        if hd[i][0]==0:
            dele_list.append(i) ## remove the event has 0 hits
    hd    = np.delete(hd   , dele_list, axis = 0)
    return hd


def load_data(data):
    print('load data:',data)
    d = h5py.File(data, 'r')
    hit_time  = d['firstHitTimeByPMT'][:]
    n_pe      = d['nPEByPMT'][:]
    theta_PMT = d['infoPMT'][:] # 0 to 180 degree
    d.close()
    ###  normalize theta ##############
    theta_PMT[:] = theta_PMT[:]/180
    ###  normalize time ##############
    hit_time[:] = hit_time[:]/100

    print("hit_time:",hit_time.shape,", n_pe:", n_pe.shape, ",theta:", theta_PMT.shape, ",event sizes:", n_pe.shape[0])
    hit_time, n_pe = shuffle(hit_time, n_pe)
    return hit_time, n_pe, theta_PMT

def Normal_cost(mu, sigma, y):
    dist = tf.distributions.Normal(loc=mu, scale=sigma)
    return tf.reduce_mean(-dist.log_prob(y))

#def Possion_cost(rate, y):
#    result = y*tf.math.log(rate) - tf.math.lgamma(1. + y) - rate
#    #return tf.reduce_mean(-result)
#    return tf.reduce_sum(-result)

def Possion_cost(y, rate, SAMPLE_SIZE):
    #print('rate=',rate.shape,',y=',y.shape)
    #rate = tf.keras.backend.repeat_elements(rate, y.shape[1], axis=1)
    rate = tf.keras.backend.repeat_elements(rate, SAMPLE_SIZE, axis=1)
    result = y*tf.math.log(rate) - tf.math.lgamma(1. + y) - rate
    result = tf.reduce_sum(-result, axis=1)
    return tf.reduce_mean(result)

def mae_cost(label_y, pred_y):
    pred_y  = tf.sort(pred_y , axis=1,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=1,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y)
    abs_sum  = tf.reduce_sum(abs_diff, axis=1, keepdims=False)
    return tf.reduce_mean(abs_sum)



#def mae_cost(pred_y, label_y):
#    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
#    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
#    abs_diff = tf.math.abs(pred_y - label_y)
#    #return tf.reduce_mean(abs_diff)
#    return tf.reduce_sum(abs_diff)

def mse_cost(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    diff = tf.math.pow((pred_y - label_y), 2)
    return tf.reduce_mean(diff)

def m4e_cost(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    diff = tf.math.pow((pred_y - label_y), 4)
    return tf.reduce_mean(diff)

def ks_cost(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    pred_y  = tf.math.cumsum(pred_y)
    label_y = tf.math.cumsum(label_y)
    abs_diff = tf.math.abs(pred_y - label_y)
    return tf.math.reduce_max(abs_diff)

def Exponential_Linear(x):
    return tf.nn.elu(x) + 1 


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run MDN training. '
        'Sensible defaults come from https://github.com/taboola/mdn-tensorflow-notebook-example/blob/master/mdn.ipynb',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--datafile', action='store', type=str,
                        help='HDF5 file paths')
    parser.add_argument('--nb-epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('--early_stop_interval', action='store', type=int, default=10,
                        help='early_stop_interval.')
    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--sample_size', action='store', type=int, default=1,
                        help='sample_size')
    parser.add_argument('--num_trials', action='store', type=int, default=40,
                        help='num_trials')
    parser.add_argument('--opt_mode', action='store', type=int, default=0,
                        help='opt_mode')
    parser.add_argument('--seq_size', action='store', type=int, default=1000,
                        help='seq_size')
    parser.add_argument('--ckpt_path', action='store', type=str,
                        help='ckpt file paths')
    parser.add_argument('--output_dir', action='store', type=str,
                        help='output_dir file paths')
    parser.add_argument('--saveCkpt', action='store', type=ast.literal_eval, default=False,
                        help='save ckpt file paths')
    parser.add_argument('--savePb', action='store', type=ast.literal_eval, default=False,
                        help='save pb file paths')
    parser.add_argument('--Restore', action='store', type=ast.literal_eval, default=False,
                        help='ckpt file paths')
    parser.add_argument('--use_uniform', action='store', type=ast.literal_eval, default=True,
                        help='use uniform noise')
    parser.add_argument('--produceEvent', action='store', type=int,
                        help='produceEvent')
    parser.add_argument('--outFilePath', action='store', type=str,
                        help='outFilePath file paths')
    parser.add_argument('--pb_file_path', action='store', type=str,
                        help='pb_file_path file paths')
    parser.add_argument('--validation_file', action='store', type=str,
                        help='validation_file file paths')
    parser.add_argument('--test_file', action='store', type=str,
                        help='test_file file paths')






    parser.add_argument('--disc-lr', action='store', type=float, default=2e-5,
                        help='Adam learning rate for discriminator')

    parser.add_argument('--gen-lr', action='store', type=float, default=2e-4,
                        help='Adam learning rate for generator')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    return parser

if __name__ == '__main__':

    print('start...')
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #if physical_devices:
    #    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #####################################
    parser = get_parser()
    parse_args = parser.parse_args()
    epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    datafile = parse_args.datafile
    ckpt_path = parse_args.ckpt_path
    output_dir= parse_args.output_dir
    saveCkpt  = parse_args.saveCkpt
    savePb    = parse_args.savePb
    Restore   = parse_args.Restore
    use_uniform   = parse_args.use_uniform
    produceEvent = parse_args.produceEvent
    outFilePath = parse_args.outFilePath
    pb_file_path = parse_args.pb_file_path
    validation_file = parse_args.validation_file
    test_file       = parse_args.test_file
    early_stop_interval       = parse_args.early_stop_interval
    sample_size       = parse_args.sample_size
    num_trials        = parse_args.num_trials
    seq_size          = parse_args.seq_size
    opt_mode          = parse_args.opt_mode
    #####################################
    # set up all the logging stuff
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s'
        '[%(levelname)s]: %(message)s'
    )

    hander = logging.StreamHandler(sys.stdout)
    hander.setFormatter(formatter)
    logger.addHandler(hander)
    #####################################
    logger.info('Sherpa setup')

    parameters = [sherpa.Continuous('learning_rate', [1e-7, 1e-2], scale='log'),
                  sherpa.Discrete('num_units', [10, 1000]),
                  sherpa.Continuous('drop_rate', [0.01, 0.5]),
                  sherpa.Discrete('hidden_layer', [1, 4]),
                  sherpa.Choice('activation', ['relu', 'tanh', 'elu'])
                 ]
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=num_trials)
    study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     disable_dashboard=True,
                     output_dir = output_dir,
                     lower_is_better=True)


    #####################################
    logger.info('Start trialing')

    for trial in study:
        lr = trial.parameters['learning_rate']
        num_units = trial.parameters['num_units']
        act = trial.parameters['activation']
        drop_rate = trial.parameters['drop_rate']
        hidden_layer = trial.parameters['hidden_layer']
        #Create model
        logger.info('Creating model')
        model = Sequential()
        model.add(Dense(num_units, input_dim=3))
        model.add(Activation(act))
        model.add(Dropout(drop_rate))
        for _ in range(hidden_layer):
            model.add(Dense(num_units))
            model.add(Activation(act))
            model.add(Dropout(drop_rate))
        model.add(Dense(seq_size))
        model.add(Activation('relu'))

        optimizer = Adam(lr=lr)
        if opt_mode == 1:
            optimizer = SGD(lr=lr, momentum=0.9, decay=0., nesterov=True)
        elif opt_mode == 2:
            optimizer = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
        elif opt_mode == 3:
            optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999)


        #partial_possion_loss = partial(Possion_cost, SAMPLE_SIZE=sample_size )
        # Functions need names or Keras will throw an error
        #partial_possion_loss.__name__ = 'possion_penalty'
        #model.compile(loss=partial_possion_loss, optimizer=optimizer )
        model.compile(loss=mae_cost, optimizer=optimizer )
        # Train model
        logger.info('Start training model')
        cost_list = []
        ealry_stop = False

        for epoch in range(epochs):
            total_cost = 0
            count = 0
            f_DataSet = open(datafile, 'r')
            lines = f_DataSet.readlines()
            random.shuffle(lines)
            for line in lines: 
                idata = line.strip('\n')
                idata = idata.strip(' ')
                if "#" in idata: continue ##skip the commented one
                if "" == idata: continue ##skip the commented one
                logger.info(str(idata))
                d = h5py.File(str(idata), 'r')
                r_theta_npe = d['nPEByPMT'][:]
                index = list(range(r_theta_npe.shape[0]))
                np.random.shuffle(index)
                r_theta_npe = r_theta_npe[index]
                batchs = int(float(r_theta_npe.shape[0])/batch_size)
                for ib in range(batchs):
                    value = r_theta_npe[ib*batch_size:(ib+1)*batch_size] 
                    n_seq = int(float(value.shape[1]-2)/seq_size)
                    for iq in range(n_seq):
                        label_y = value[:,2+iq*seq_size:2+(iq+1)*seq_size]
                        r     = value[:,0:1]/20   #normalize
                        theta = value[:,1:2]/180  #normalize
                        noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
                        inputs = np.concatenate ((r, theta, noise), axis=-1)
                        #if np.any(np.isnan(label_y)): print('find Nan in label_y')
                        #if np.any(np.isnan(inputs)) : print('find Nan in inputs' )
                        c    = model.train_on_batch(inputs, label_y)
                        total_cost += c
                        count = count + 1
                d.close()
            f_DataSet.close()
            avg_cost = total_cost/count
            if epoch % 1 == 0:
                print('Epoch {0} | cost = {1:.4f}'.format(epoch,avg_cost))
                logger.info('')
            ############ early stop ###################
            if len(cost_list) < early_stop_interval: cost_list.append(avg_cost)
            else:
                for ic in range(len(cost_list)-1):
                    cost_list[ic] = cost_list[ic+1]
                cost_list[-1] = avg_cost
            if epoch > len(cost_list) and avg_cost > cost_list[0]: ealry_stop = True
            ############ study ###################
            study.add_observation(trial=trial, iteration=epoch, objective=avg_cost, context={'loss': avg_cost})
            if study.should_trial_stop(trial) or ealry_stop:
                break 
        study.finalize(trial=trial)
        print('opt_mode = ',opt_mode,',get_best_result()=\n',study.get_best_result())
    print('save resluts to %s'%output_dir)
    study.save()
    #####################################
    print('done')
