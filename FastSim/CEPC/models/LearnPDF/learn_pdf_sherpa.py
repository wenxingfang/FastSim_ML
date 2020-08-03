import h5py
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
import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization
###########################
## separate one r,theta array to number of batch
## now the data is one numpy array for each r,theta
## add selection for training, validing, testing
## for learning npe pdf , pdf(npe | r, theta)  #
## each h5 file has one npe numpy array, different pmt for each row, first col is r, second col is theta, third is first event, fourth is second event and so on
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


def load_data(datafile):
    d = h5py.File(datafile, 'r')
    first   = d['Barrel_Hit'][:]
    mc_info = d['MC_info'][:]
    d.close()
    ###### do normalization ##############
    mc_info[:,0] = (mc_info[:,0])/50.0
    mc_info[:,1] = (mc_info[:,1])/1850.0
    mc_info[:,2] = (mc_info[:,2])
    mc_info[:,3] = (mc_info[:,3])
    mc_info[:,4] = (mc_info[:,4])/90.0
    mc_info[:,5] = (mc_info[:,5])
    first[:,:,0] = (first[:,:,0]-1850.0)/10
    first[:,:,1] = (first[:,:,1])/10 #to cm
    first[:,:,2] = (first[:,:,2])/10 #to cm
    
    print("first:",first.shape,",mc info:", mc_info.shape)
    ###### do scale ##############
    first, mc_info = shuffle(first, mc_info)
    return first, mc_info

def Exponential_Linear(x):
    return tf.nn.elu(x) + 1


def Normal_cost(mu, sigma, y):
    dist = tf.distributions.Normal(loc=mu, scale=sigma)
    return tf.reduce_mean(-dist.log_prob(y))

def Possion_cost(rate, y):
    #dist = tfp.distributions.Poisson(rate=rate, allow_nan_stats=False)
    #return tf.reduce_mean(-dist.log_prob(y))
    result = y*tf.math.log(rate) - tf.math.lgamma(1. + y) - rate
    return tf.reduce_mean(-result)

def mae_cost(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=1,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=1,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y)
    #return tf.reduce_mean(abs_diff)
    return tf.reduce_mean( tf.reduce_sum(abs_diff, 1) )

def mae_cost_1(label_y, pred_y):
    abs_diff = tf.math.abs(pred_y - label_y)
    sum_pred_y  = tf.reduce_sum(pred_y, 1)
    sum_label_y = tf.reduce_sum(label_y, 1)
    abs_sum_diff = tf.math.abs(sum_pred_y - sum_label_y)
    return ( tf.reduce_mean( tf.reduce_sum(abs_diff, 1) ) + tf.reduce_mean(abs_sum_diff) )

def mae_cost_v1(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y) + 0.5
    abs_diff = tf.math.pow(abs_diff, 2)
    #return tf.reduce_mean(abs_diff)
    return tf.reduce_sum(abs_diff)


def mae_cost_v2(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y) + 0.5
    abs_diff = tf.math.pow(abs_diff, 2)
    return tf.reduce_mean(abs_diff)

def mae_cost_v3(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y) + 0.5
    abs_diff = tf.math.pow(abs_diff, 4)
    return tf.reduce_sum(abs_diff)
    #return tf.reduce_mean(abs_diff)

def mae_cost_v4(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y) + 0.5
    abs_diff = tf.math.pow(abs_diff, 8)
    return tf.reduce_sum(abs_diff)

def mae_cost_v1_w(pred_y, label_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y) + 0.5
    abs_diff = tf.math.pow(abs_diff, 2)*(label_y+1)
    #return tf.reduce_mean(abs_diff)
    return tf.reduce_sum(abs_diff)

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
    parser.add_argument('--act_mode', action='store', type=int, default=0,
                        help='act_mode')
    parser.add_argument('--N_units', action='store', type=int, default=1000,
                        help='N_units')
    parser.add_argument('--N_hidden', action='store', type=int, default=3,
                        help='N_hidden')
    parser.add_argument('--cost_mode', action='store', type=int, default=0,
                        help='cost_mode')
    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--ckpt_path', action='store', type=str,
                        help='ckpt file paths')
    parser.add_argument('--model_out', action='store', type=str,
                        help='model out')
    parser.add_argument('--restore_ckpt_path', action='store', type=str,
                        help='restore_ckpt_path ckpt file paths')
    parser.add_argument('--saveCkpt', action='store', type=ast.literal_eval, default=False,
                        help='save ckpt file paths')
    parser.add_argument('--savePb', action='store', type=ast.literal_eval, default=False,
                        help='save pb file paths')
    parser.add_argument('--Restore', action='store', type=ast.literal_eval, default=False,
                        help='ckpt file paths')
    parser.add_argument('--doTraining', action='store', type=ast.literal_eval, default=False,
                        help='doTraining')
    parser.add_argument('--doValid', action='store', type=ast.literal_eval, default=False,
                        help='doValid')
    parser.add_argument('--doTest', action='store', type=ast.literal_eval, default=False,
                        help='doTest')
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
    parser.add_argument('--restored_model', action='store', type=str,
                        help='restored model')
    parser.add_argument('--num_trials', action='store', type=int, default=40,
                        help='num_trials')
    parser.add_argument('--output_dir', action='store', type=str,
                        help='output_dir file paths')
    parser.add_argument('--opt_mode', action='store', type=int, default=0,
                        help='opt_mode')

    parser.add_argument('--lr', action='store', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--Scale', action='store', type=float, default=1,
                        help='scale npe')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    return parser

if __name__ == '__main__':

    print('start...')
    from tensorflow.python.framework import graph_util
    from keras.models import load_model
    from keras.layers import (Dense, Input)
    from keras.layers.merge import add, concatenate, multiply
    from keras.models import Model
    from keras.optimizers import Adam

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
    model_out = parse_args.model_out
    restore_ckpt_path = parse_args.restore_ckpt_path
    saveCkpt  = parse_args.saveCkpt
    savePb    = parse_args.savePb
    Restore   = parse_args.Restore
    doTraining    = parse_args.doTraining
    doValid       = parse_args.doValid
    doTest        = parse_args.doTest
    use_uniform   = parse_args.use_uniform
    produceEvent = parse_args.produceEvent
    outFilePath = parse_args.outFilePath
    pb_file_path = parse_args.pb_file_path
    validation_file = parse_args.validation_file
    test_file       = parse_args.test_file
    early_stop_interval       = parse_args.early_stop_interval
    act_mode        = parse_args.act_mode
    N_units         = parse_args.N_units
    N_hidden        = parse_args.N_hidden
    learning_rate   = parse_args.lr
    cost_mode       = parse_args.cost_mode
    Scale           = parse_args.Scale
    restore_model_in  = parse_args.restored_model
    num_trials        = parse_args.num_trials
    output_dir= parse_args.output_dir
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

    parameters = [sherpa.Continuous('learning_rate', [1e-9, 1e-2], scale='log'),
                  sherpa.Discrete('num_units', [10, 2000]),
                  #sherpa.Continuous('drop_rate', [0.01, 0.5]),
                  sherpa.Discrete('hidden_layer', [2, 5]),
                  sherpa.Choice('activation_E', [Exponential_Linear]),
                  #sherpa.Choice('activation_E', ['relu',Exponential_Linear]),
                  #sherpa.Choice('activation', ['relu', 'tanh', 'elu'])
                  sherpa.Choice('activation', ['relu', 'elu'])
                 ]
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=num_trials)
    study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     disable_dashboard=True,
                     output_dir = output_dir,
                     lower_is_better=True)
    #####################################
    logger.info('constructing graph')
    tried = 0
    for trial in study:
        lr = trial.parameters['learning_rate']
        num_units = trial.parameters['num_units']
        act = trial.parameters['activation']
        act_E = trial.parameters['activation_E']
        #drop_rate = trial.parameters['drop_rate']
        hidden_layer = trial.parameters['hidden_layer']
        #Create model
        print('tried=',tried)
        tried += 1
        logger.info('Creating model')

        tf.reset_default_graph()

        input_Info  = Input(name='input_Info', shape=(10,))
        layer = Dense(num_units, activation=act)(input_Info)
        for _ in range(hidden_layer-1):
            layer = Dense(num_units, activation=act)(layer)
        Pred_x = Dense(500, name = 'Pred_x')(layer)
        con_Pred_y = concatenate( [Pred_x, layer] )
        Pred_y = Dense(500, name = 'Pred_y')(con_Pred_y)
        con_Pred_z = concatenate( [Pred_x, Pred_y, layer] )
        Pred_z = Dense(500, name = 'Pred_z')(con_Pred_z)
        con_Pred_E = concatenate( [Pred_x, Pred_y, Pred_z, layer] )
        Pred_E = Dense(500, activation=act_E, name = 'Pred_E')(con_Pred_E)
        #Pred_E = Dense(500, activation='relu', name = 'Pred_E')(con_Pred_E)
        #Pred_E = Dense(500, activation=Exponential_Linear, name = 'Pred_E')(con_Pred_E)
        print('creating new model')
        Predictor = Model(input_Info , [Pred_x, Pred_y, Pred_z, Pred_E], name='predictor')

        optimizer = Adam(lr=lr)
        if opt_mode == 1:
            optimizer = SGD(lr=lr, momentum=0.9, decay=0., nesterov=True)
        elif opt_mode == 2:
            optimizer = RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)
        elif opt_mode == 3:
            optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999)

        Predictor.compile(
            optimizer=optimizer,
            loss=['mae','mae','mae', mae_cost_1] )


        ########################################
        print('commencing training')
        logger.info('')
        print('preparing data list for training')

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
            ievent   = d['Barrel_Hit'].shape[0]
            d.close()
            Event.append(float(ievent))
            Batch.append(int(float(ievent)/batch_size))
        total_event = sum(Event)
        f_DataSet.close()
        print('total sample:', total_event)
        print('All Batch:', Batch)
        logger.info('')
        ealry_stop = False
        cost_list = []
        epoch_loss = []
        loss_weights      = [np.ones(batch_size), 1*np.ones(batch_size), np.ones(batch_size), 1*np.ones(batch_size) ]
        for epoch in range(epochs):
            logger.info('Epoch {} of {}'.format(epoch + 1, epochs))
            nb_batches = sum(Batch)
            for ib in range(len(Batch)):
                first, mc_info = load_data(Data[ib])
                ibatch = Batch[ib]
                for index in range(ibatch):
                    hit_batch   =  first     [index * batch_size:(index + 1) * batch_size]
                    info_batch  =  mc_info   [index * batch_size:(index + 1) * batch_size]
                    noise = np.random.normal(0, 1, (batch_size, 4))
                    inputs= np.concatenate((info_batch, noise), axis=-1)
                    outputs_label = [ hit_batch[:,:,0], hit_batch[:,:,1], hit_batch[:,:,2], hit_batch[:,:,3] ]
                    batch_loss = Predictor.train_on_batch(
                    inputs,
                    outputs_label,
                    loss_weights
                    )
                    #logger.info('batch loss: {}'.format(batch_loss))
                    #sys.exit()
                    epoch_loss.append(np.array(batch_loss))
            logger.info('Epoch {:3d} loss: {}'.format(epoch + 1, np.mean(epoch_loss, axis=0)))
            avg_cost = np.mean(epoch_loss, axis=0 )[0]
            ############ early stop ###################
            if len(cost_list) < early_stop_interval: cost_list.append(avg_cost)
            else:
                for ic in range(len(cost_list)-1):
                    cost_list[ic] = cost_list[ic+1]
                cost_list[-1] = avg_cost
            if epoch > len(cost_list) and avg_cost >= cost_list[0]: ealry_stop=True
            ############ study ###################
            study.add_observation(trial=trial, iteration=epoch, objective=avg_cost, context={'loss': np.mean(epoch_loss, axis=0 )})
            if study.should_trial_stop(trial) or ealry_stop:
                break 
        study.finalize(trial=trial)
        print('opt_mode = ',opt_mode,',get_best_result()=\n',study.get_best_result())
    print('save resluts to %s'%output_dir)
    study.save()
    #####################################
    print('done')
