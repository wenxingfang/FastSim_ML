import h5py
import ast
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
from tensorflow.python.framework import graph_util
#########################################################
import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.optimizers import Adam, SGD, RMSprop
###########################
## for learning time pdf , pdf(time | r, theta)  #
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

def Possion_cost(rate, y):
    #dist = tfp.distributions.Poisson(rate=rate, allow_nan_stats=False)
    #return tf.reduce_mean(-dist.log_prob(y))
    result = y*tf.math.log(rate) - tf.math.lgamma(1. + y) - rate
    return tf.reduce_mean(-result)

def mae_cost(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y)
    #return tf.reduce_mean(abs_diff)
    return tf.reduce_sum(abs_diff)

def mse_cost(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    diff = tf.math.pow((pred_y - label_y), 2)
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

def Inverse_Linear(x):
    x_1 = tf.add(x, tf.constant(1.0))
    x_2 = tf.subtract(tf.constant(1.0), x)
    x_3 = tf.divide(tf.constant(1.0), x_2)
    cond = tf.less(x, tf.constant(0.0))
    return tf.where(cond, x_3, x_1)

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
    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--act_mode', action='store', type=int, default=0,
                        help='activation for last layer')
    parser.add_argument('--opt_mode', action='store', type=int, default=0,
                        help='optimizer')
    parser.add_argument('--early_stop_interval', action='store', type=int, default=10,
                        help='early_stop_interval')
    parser.add_argument('--output_dir', action='store', type=str,
                        help='output_dir file paths')
    parser.add_argument('--saveCkpt', action='store', type=ast.literal_eval, default=False,
                        help='save ckpt file')
    parser.add_argument('--savePb', action='store', type=ast.literal_eval, default=False,
                        help='save pb file')
    parser.add_argument('--Restore', action='store', type=ast.literal_eval, default=False,
                        help='ckpt file paths')
    parser.add_argument('--use_uniform', action='store', type=ast.literal_eval, default=True,
                        help='use uniform noise')
    parser.add_argument('--produceEvent', action='store', type=int,
                        help='produceEvent')
    parser.add_argument('--outFileName', action='store', type=str,
                        help='outFileName file paths')
    parser.add_argument('--pb_file_path', action='store', type=str,
                        help='pb_file_path file paths')
    parser.add_argument('--validation_file', action='store', type=str,
                        help='validation_file file paths')
    parser.add_argument('--test_file', action='store', type=str,
                        help='test_file file paths')
    parser.add_argument('--norm_mode', action='store', type=int, default=0,
                        help='mode of normalize.')






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
    act_mode = parse_args.act_mode
    opt_mode = parse_args.opt_mode
    early_stop_interval = parse_args.early_stop_interval
    datafile = parse_args.datafile
    output_dir = parse_args.output_dir
    saveCkpt  = parse_args.saveCkpt
    savePb    = parse_args.savePb
    Restore   = parse_args.Restore
    use_uniform   = parse_args.use_uniform
    norm_mode     = parse_args.norm_mode
    produceEvent = parse_args.produceEvent
    outFileName = parse_args.outFileName
    pb_file_path = parse_args.pb_file_path
    validation_file = parse_args.validation_file
    test_file       = parse_args.test_file
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

    parameters = [#sherpa.Continuous('learning_rate', [1e-4, 1e-3]),
                  sherpa.Continuous('learning_rate', [1e-5, 1], scale='log'),
                  sherpa.Discrete('num_units', [10, 100]),
                  sherpa.Discrete('hidden_layer', [2, 5]),
                  sherpa.Choice('activation', ['relu', 'tanh']),
                  #sherpa.Choice('activation', ['relu', 'tanh', 'sigmoid']),
                  sherpa.Choice('last_activation', [Exponential_Linear])
                  #sherpa.Choice('last_activation', ['relu', Exponential_Linear])
                 ]
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=40)
    study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     disable_dashboard=True,
                     output_dir = output_dir,
                     lower_is_better=True)


    #####################################
    logger.info('Start trialing')
    i_trial = 1
    for trial in study:
        print('trial=',i_trial)
        logger.info('')
        lr = trial.parameters['learning_rate']
        num_units = trial.parameters['num_units']
        act = trial.parameters['activation']
        last_act = trial.parameters['last_activation']
        n_hidden = trial.parameters['hidden_layer']
        #Create model
        logger.info('Creating model')
        model = Sequential()
        model.add(Dense(num_units, input_dim=3))
        model.add(Activation(act))
        #for _ in range(2):
        for _ in range(n_hidden):
            model.add(Dense(num_units))
            model.add(Activation(act))
        model.add(Dense(1))
        model.add(Activation(last_act))

        optimizer = Adam(lr=lr)
        if opt_mode == 1:
            optimizer = SGD(lr=lr, momentum=0.9, decay=0., nesterov=True)
        elif opt_mode == 2:
            optimizer = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
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
                if ("#" in idata) or idata=='': continue ##skip the commented one
                logger.info(str(idata))
                d = h5py.File(str(idata), 'r')
                list_key = list(d.keys())
                random.shuffle(list_key)
                for ikey in list_key:
                    if 'HitTimeByPMT' not in ikey: continue
                    value = shuffle(d[ikey][:]/100)
                    value = np.reshape(value, (-1,1))
                    if value.shape[0]==0: continue
                    r     = np.array( [(float(ikey.split('_')[1])-10)/10]) if norm_mode==0 else  np.array( [float(ikey.split('_')[1])/20] )
                    theta = np.array( [(float(ikey.split('_')[2])-90)/90]) if norm_mode==0 else  np.array( [float(ikey.split('_')[2])/180])  #normalize
                    r     = np.reshape(r    , (1,1))
                    theta = np.reshape(theta, (1,1))
                    r     = r    .repeat(value.shape[0],axis=0)
                    theta = theta.repeat(value.shape[0],axis=0)
                    noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
                    inputs = np.concatenate ((r, theta, noise), axis=-1)
                    n_batch = int(float(value.shape[0])/batch_size)
                    if n_batch > 1 :
                        for ib in range(n_batch):
                            train_x = inputs [ib * batch_size:(ib + 1) * batch_size]
                            train_y = value  [ib * batch_size:(ib + 1) * batch_size]
                            #if np.any(np.isnan(train_x)): print('find Nan in train_x')
                            #if np.any(np.isnan(train_y)): print('find Nan in train_y')
                            loss = model.train_on_batch(train_x, train_y)
                            total_cost += loss
                            count = count + 1
                    else :
                        train_x = inputs
                        train_y = value 
                        #if np.any(np.isnan(train_x)): print('find Nan in train_x')
                        #if np.any(np.isnan(train_y)): print('find Nan in train_y')
                        loss = model.train_on_batch(train_x, train_y)
                        total_cost += loss
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
        print('get_best_result()=\n',study.get_best_result())
        i_trial += 1
    print('save resluts to %s'%output_dir)
    study.save()
    #####################################
    print('done')
