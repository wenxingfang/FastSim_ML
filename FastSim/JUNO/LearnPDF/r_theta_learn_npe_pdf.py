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

###########################
## for learning npe pdf , pdf(npe | r, theta)  #
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
    return tf.reduce_mean(abs_diff)

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
    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--ckpt_path', action='store', type=str,
                        help='ckpt file paths')
    parser.add_argument('--produceEvent', action='store', type=int,
                        help='produceEvent')
    parser.add_argument('--outFileName', action='store', type=str,
                        help='outFileName file paths')
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
    produceEvent = parse_args.produceEvent
    outFileName = parse_args.outFileName
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
    logger.info('constructing graph')

    tf.reset_default_graph()
    x = tf.placeholder(name='x',shape=(None,3),dtype=tf.float32)
    y = tf.placeholder(name='y',shape=(None,1),dtype=tf.float32)
    layer = x
    for _ in range(3):
        layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.tanh)
    #Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.elu(x) + 1)
    Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x))
    #cost = mse_cost(Pred_y, y)
    #cost = mae_cost(Pred_y, y)
    cost = ks_cost(Pred_y, y)
    learning_rate = 0.0003
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    ########################################
    print('commencing training')
    logger.info('')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
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
                logger.info(str(idata))
                d = h5py.File(str(idata), 'r')
                list_key = list(d.keys())
                random.shuffle(list_key)
                for ikey in list_key:
                    if 'nPEByPMT' not in ikey: continue
                    value = shuffle(d[ikey][:])
                    value = np.reshape(value, (-1,1))
                    #noise0= np.random.uniform(-0.45, 0.45, (value.shape[0], 1))
                    #value = np.add(value, noise0)
                    #value[value<0] = 0
                    if float(ikey.split('_')[1]) > 17.7: continue
                    r     = np.array( [float(ikey.split('_')[1])/20] )  #normalize
                    theta = np.array( [float(ikey.split('_')[2])/180])  #normalize
                    r     = np.reshape(r    , (1,1))
                    theta = np.reshape(theta, (1,1))
                    r     = r    .repeat(value.shape[0],axis=0)
                    theta = theta.repeat(value.shape[0],axis=0)
                    #noise = np.random.uniform(-1, 1, (value.shape[0], 1))
                    noise = np.random.normal(0, 1, (value.shape[0], 1))
                    inputs = np.concatenate ((r, theta, noise), axis=-1)
                    n_batch = int(float(value.shape[0])/batch_size)
                    for ib in range(n_batch):
                        train_x = inputs [ib * batch_size:(ib + 1) * batch_size]
                        train_y = value  [ib * batch_size:(ib + 1) * batch_size]
                        if np.any(np.isnan(train_x)): print('find Nan in train_x')
                        if np.any(np.isnan(train_y)): print('find Nan in train_y')
                        _, c = sess.run([optimizer,cost], feed_dict={x:train_x, y:train_y})
                        total_cost += c
                        count = count + 1
                d.close()
            f_DataSet.close()
            avg_cost = total_cost/count
            if epoch % 1 == 0:
                print('Epoch {0} | cost = {1:.4f}'.format(epoch,avg_cost))
                logger.info('')
        ### validation ############################
        print('Do validation')
        logger.info('')
        cost_valid = 0 
        count = 0
        f_DataSet = open(validation_file, 'r')
        for line in f_DataSet: 
            idata = line.strip('\n')
            idata = idata.strip(' ')
            if "#" in idata: continue ##skip the commented one
            d = h5py.File(str(idata), 'r')
            for ikey in d.keys():
                if 'nPEByPMT' not in ikey: continue
                value = d[ikey][:]
                value = np.reshape(value, (-1,1))
                r     = np.array( [float(ikey.split('_')[1])/20] )  #normalize
                theta = np.array( [float(ikey.split('_')[2])/180])  #normalize
                r     = np.reshape(r    , (1,1))
                theta = np.reshape(theta, (1,1))
                r     = r    .repeat(value.shape[0],axis=0)
                theta = theta.repeat(value.shape[0],axis=0)
                #noise = np.random.uniform(-1, 1, (value.shape[0], 1))
                noise = np.random.normal(0, 1, (value.shape[0], 1))
                inputs = np.concatenate ((r, theta, noise), axis=-1)
                n_batch = int(float(value.shape[0])/batch_size)
                for ib in range(n_batch):
                    valid_x = inputs [ib * batch_size:(ib + 1) * batch_size]
                    valid_y = value  [ib * batch_size:(ib + 1) * batch_size]
                    if np.any(np.isnan(valid_x)): print('find Nan in valid_x')
                    if np.any(np.isnan(valid_y)): print('find Nan in valid_y')
                    _, c = sess.run([Pred_y, cost], feed_dict={x:valid_x, y:valid_y})
                    cost_valid += c
                    count = count + 1
            d.close()
        f_DataSet.close()
        avg_cost = cost_valid/count
        print('ave valid cost = {0:.4f}'.format(avg_cost))
        #### produce predicted data #################
        print('Saving produced data')
        logger.info('')
        hf = h5py.File(outFileName, 'w')
        r_theta = []
        f_DataSet = open(test_file, 'r')
        for line in f_DataSet: 
            idata = line.strip('\n')
            idata = idata.strip(' ')
            if "#" in idata: continue ##skip the commented one
            d = h5py.File(str(idata), 'r')
            for ikey in d.keys():
                if 'nPEByPMT' not in ikey: continue
                str_r_theta = '%s_%s'%(str(ikey.split('_')[1]),str(ikey.split('_')[2]))
                if str_r_theta in r_theta: continue
                r_theta.append(str_r_theta)

                r     = np.array( [float(ikey.split('_')[1])/20] )  #normalize
                theta = np.array( [float(ikey.split('_')[2])/180])  #normalize
                r     = np.reshape(r    , (1,1))
                theta = np.reshape(theta, (1,1))
                r     = r    .repeat(produceEvent,axis=0)
                theta = theta.repeat(produceEvent,axis=0)
                #noise = np.random.uniform(-1, 1, (produceEvent, 1))
                noise = np.random.normal(0, 1, (produceEvent, 1))
                inputs = np.concatenate ((r, theta, noise), axis=-1)
                y_pred = sess.run(Pred_y,feed_dict={x:inputs})
                y_pred = np.reshape(y_pred, produceEvent)
                hf.create_dataset('prednPEByPMT_%s'%str_r_theta, data=y_pred)
            d.close()
        f_DataSet.close()
        hf.close()
        print('Saved produced data %s'%outFileName)
        ############## Save the variables to disk.
        if False:
            save_path = saver.save(sess, "%s/model.ckpt"%(ckpt_path))
            print("Model saved in path: %s" % save_path)
    print('done')
