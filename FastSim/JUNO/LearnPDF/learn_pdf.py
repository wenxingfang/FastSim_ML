import h5py
import sys
import argparse
import numpy as np
import math
import tensorflow as tf
#import tensorflow_probability as tfp
#import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def load_data(data):
    print('load data:',data)
    d = h5py.File(data, 'r')
    hit_time  = d['firstHitTimeByPMT'][:]
    n_pe      = d['nPEByPMT'][:]
    theta_PMT = d['infoPMT'][:] # 0 to 180 degree
    d.close()
    ###  normalize theta ##############
    theta_PMT[:] = theta_PMT[:]/180
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
    #####################################
    print('constructing graph')
    tf.reset_default_graph()
    x = tf.placeholder(name='x',shape=(None,2),dtype=tf.float32)
    y = tf.placeholder(name='y',shape=(None,1),dtype=tf.float32)
    layer = x
    for _ in range(3):
        layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.tanh)
    #Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.elu(x) + 1)
    Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x))
    cost = mse_cost(Pred_y, y)
    learning_rate = 0.0003
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    ########################################
    print('preparing data')
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
    ########################################
    print('commencing training')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            total_cost = 0
            for ib in range(len(Batch)):
                hit_time, n_pe, theta = load_data(Data[ib])
                theta  = theta.repeat(batch_size,axis=0)
                ibatch = Batch[ib]
                print('ib {0}, ibatch {1}'.format(ib, ibatch))
                for index in range(ibatch):
                    hit_time_batch = hit_time [index * batch_size:(index + 1) * batch_size]
                    n_pe_batch     = n_pe     [index * batch_size:(index + 1) * batch_size]
                    for itheta in range(theta.shape[1]):
                        train_x = theta     [:,itheta:itheta+1]
                        noise = np.random.uniform(-1, 1, (train_x.shape[0], 1))
                        input_x = np.concatenate ((train_x, noise), axis=-1)
                        train_y = n_pe_batch[:,itheta:itheta+1]
                        if np.any(np.isnan(train_x)): print('find Nan in train_x')
                        if np.any(np.isnan(train_y)): print('find Nan in train_y')
                        _, c = sess.run([optimizer,cost], feed_dict={x:input_x, y:train_y})
                        total_cost += c
            avg_cost = total_cost/(sum(Batch)*theta.shape[1])
            if epoch % 1 == 0:
               print('Epoch {0} | cost = {1:.4f}'.format(epoch,avg_cost))
        ### validation ############################
        print('Do validation')
        hit_time, n_pe, theta = load_data(validation_file)
        theta  = theta.repeat(n_pe.shape[0],axis=0)
        cost_valid = 0 
        for itheta in range(theta.shape[1]):
            valid_x   = theta     [:,itheta:itheta+1]
            noise = np.random.uniform(-1, 1, (valid_x.shape[0], 1))
            input_x = np.concatenate ((valid_x, noise), axis=-1)
            rate_pred, c_pred = sess.run([Pred_y,cost],feed_dict={x:input_x, y:n_pe[:,itheta:itheta+1]})
            #print('valid cost = {0:.4f}'.format(c_pred))
            cost_valid = cost_valid + c_pred/theta.shape[1]
        print('ave valid cost = {0:.4f}'.format(cost_valid))
 
        #### produce predicted data #################
        print('Saving produced data')
        theta_list = list(set(theta[0,:]))
        theta_list.sort() 
        print('theta_list=', len(theta_list))
        pred_n_pe = np.full((produceEvent, len(theta_list) ), 0 ,dtype=np.float32)#init
        for i in theta_list:
            ithe = np.full((produceEvent, 1 ), i ,dtype=np.float32)
            noise = np.random.uniform(-1, 1, (ithe.shape[0], 1))
            input_x = np.concatenate ((ithe, noise), axis=-1)
            y_pred = sess.run(Pred_y,feed_dict={x:input_x})
            y_pred = y_pred.reshape((-1,1))
            pred_n_pe[:,theta_list.index(i):theta_list.index(i)+1] = y_pred
        hf = h5py.File(outFileName, 'w')
        hf.create_dataset('pred_n_pe', data=pred_n_pe)
        hf.create_dataset('theta_set', data=np.array(theta_list))
        hf.close()
        print('Saved produced data %s'%outFileName)
        ############## Save the variables to disk.
        if False:
            save_path = saver.save(sess, "%s/model.ckpt"%(ckpt_path))
            print("Model saved in path: %s" % save_path)
    print('done')
