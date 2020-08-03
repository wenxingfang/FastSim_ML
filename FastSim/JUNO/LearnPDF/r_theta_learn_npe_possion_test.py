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
from tensorflow.python.framework import graph_util
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

def Possion_cost(rate, y):
    rate = tf.keras.backend.repeat_elements(rate, y.shape[1], axis=1)
    result = y*tf.math.log(rate) - tf.math.lgamma(1. + y) - rate
    result = tf.reduce_sum(-result, axis=1)
    return tf.reduce_mean(result)

def Possion_cost_v1(rate, y): # for possion, we have rate (or lambda) = E(y) = mean(y)
    y =  tf.reduce_mean(y, axis=1, keepdims=True)
    result = tf.abs(rate - y)
    return tf.reduce_mean(result)

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
    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--sample_size', action='store', type=int, default=1,
                        help='sample_size')
    parser.add_argument('--ckpt_path', action='store', type=str,
                        help='ckpt file paths')
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
    #### produce predicted data #################
    print('Saving produced data')
    logger.info('')
    r_theta = []
    f_DataSet = open(test_file, 'r')
    for line in f_DataSet: 
        idata = line.strip('\n')
        idata = idata.strip(' ')
        if "#" in idata: continue ##skip the commented one
        d = h5py.File(str(idata), 'r')
        r_theta_npe = d['nPEByPMT'][:]
        pred_nPEByPMT    = np.full((r_theta_npe.shape[0], produceEvent+2), 0, dtype=np.float32)
        for ipmt in range(r_theta_npe.shape[0]):
            pred_nPEByPMT[ipmt,0]=r_theta_npe[ipmt,0]
            pred_nPEByPMT[ipmt,1]=r_theta_npe[ipmt,1]
            y_pred = np.random.poisson(np.mean(r_theta_npe[ipmt,2:r_theta_npe.shape[1]]), produceEvent)
            y_pred = np.reshape(y_pred, produceEvent)
            pred_nPEByPMT[ipmt,2:pred_nPEByPMT.shape[1]]=y_pred
        in_str = idata.split('/')[-1]
        out_str = in_str.replace('.h5','_pred.h5')
        out_str = outFilePath+'/'+out_str
        hf = h5py.File(out_str, 'w')
        hf.create_dataset('nPEByPMT', data=pred_nPEByPMT)
        hf.close()
        print('Saved produced data %s'%out_str)
        d.close()
    f_DataSet.close()
    print('done')