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



def load_data(data, events):
    print('load data:',data)
    d = h5py.File(data, 'r')
    data_load  = d['Gamma'][:] if events >= d['Gamma'].shape[0] else d['Gamma'][0:events]
    d.close()
    ###  normalized ##############
    data_load[:,2] = data_load[:,2]/10
    print("data shape:",data_load.shape)
    data_load = shuffle(data_load)
    return data_load



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
    #return tf.reduce_sum(abs_diff)

def mae_cost0(pred_y, label_y):
    abs_diff = tf.math.abs(pred_y - label_y)
    return tf.reduce_mean(abs_diff)

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
    parser.add_argument('--N_units', action='store', type=int, default=12,
                        help='N_units')
    parser.add_argument('--N_hidden', action='store', type=int, default=3,
                        help='N_hidden')
    parser.add_argument('--cost_mode', action='store', type=int, default=0,
                        help='cost_mode')
    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--ckpt_path', action='store', type=str,
                        help='ckpt file paths')
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
    parser.add_argument('--outFile', action='store', type=str,
                        help='outFile file paths')
    parser.add_argument('--pb_file_path', action='store', type=str,
                        help='pb_file_path file paths')
    parser.add_argument('--validation_file', action='store', type=str,
                        help='validation_file file paths')
    parser.add_argument('--test_file', action='store', type=str,
                        help='test_file file paths')

    parser.add_argument('--lr', action='store', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--Scale', action='store', type=float, default=1,
                        help='scale npe')

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
    restore_ckpt_path = parse_args.restore_ckpt_path
    saveCkpt  = parse_args.saveCkpt
    savePb    = parse_args.savePb
    Restore   = parse_args.Restore
    doTraining    = parse_args.doTraining
    doValid       = parse_args.doValid
    doTest        = parse_args.doTest
    use_uniform   = parse_args.use_uniform
    produceEvent = parse_args.produceEvent
    outFile = parse_args.outFile
    pb_file_path = parse_args.pb_file_path
    validation_file = parse_args.validation_file
    test_file       = parse_args.test_file
    early_stop_interval       = parse_args.early_stop_interval
    act_mode        = parse_args.act_mode
    N_units         = parse_args.N_units
    N_hidden        = parse_args.N_hidden
    lr              = parse_args.lr
    cost_mode       = parse_args.cost_mode
    Scale           = parse_args.Scale
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
    act = {0:tf.nn.tanh, 1:tf.nn.elu, 2:tf.nn.relu}
    for _ in range(N_hidden):
        layer = tf.layers.dense(inputs=layer, units=N_units, activation=act[act_mode])
        #layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.tanh)
    Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.elu(x) + 1)
    #Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x), name = 'Pred')
    #cost = mse_cost(Pred_y, y)
    cost = mae_cost(Pred_y, y)
    #cost = mae_cost0(Pred_y, y)
    if cost_mode == 1:
        cost = mae_cost_v1(Pred_y, y)
    elif cost_mode == 2:
        cost = mae_cost_v2(Pred_y, y)
    elif cost_mode == 3:
        cost = mae_cost_v3(Pred_y, y)
    #cost = mae_cost_v1_w(Pred_y, y)
    #cost = ks_cost(Pred_y, y)
    learning_rate = lr
    #learning_rate = 0.0003
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    ########################################
    print('commencing training')
    logger.info('')
    with tf.Session() as sess:
        if Restore:
            ckpt = tf.train.get_checkpoint_state("%s"%restore_ckpt_path)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
            print('restored ckpt:',ckpt.all_model_checkpoint_paths[0])
        else:
            sess.run(tf.global_variables_initializer())
        if doTraining:
            cost_list = []
            for epoch in range(epochs):
                total_cost = 0
                count = 0
                dataset = load_data(datafile, 10000)
                data_x     = dataset[:,0:2]
                data_y     = dataset[:,2:3]
                noise = np.random.uniform(-1, 1, (data_x.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (data_x.shape[0], 1))
                inputs = np.concatenate ((data_x, noise), axis=-1)
                n_batch = int(float(data_x.shape[0])/batch_size)
                if n_batch > 1 :
                    for ib in range(n_batch):
                        train_x = inputs [ib * batch_size:(ib + 1) * batch_size]
                        train_y = data_y [ib * batch_size:(ib + 1) * batch_size]
                        _, c = sess.run([optimizer,cost], feed_dict={x:train_x, y:train_y})
                        total_cost += c
                        count = count + 1
                else :
                    train_x = inputs
                    train_y = data_y
                    _, c = sess.run([optimizer,cost], feed_dict={x:train_x, y:train_y})
                    total_cost += c
                    count = count + 1
                avg_cost = total_cost/count
                if epoch % 1 == 0:
                    print('Epoch {0} | cost = {1:.4f}'.format(epoch,avg_cost))
                    save_path = saver.save(sess, "%s/model.ckpt"%(ckpt_path))
                    print("Model saved in path: %s" % save_path)
                    logger.info('')
                ############ early stop ###################
                if len(cost_list) < early_stop_interval: cost_list.append(avg_cost)
                else:
                    for ic in range(len(cost_list)-1):
                        cost_list[ic] = cost_list[ic+1]
                    cost_list[-1] = avg_cost
                if epoch > len(cost_list) and avg_cost >= cost_list[0]: break

        ### validation ############################
        if doValid:
            pass
        #### produce predicted data #################
        if doTest:
            print('Saving produced data')
            logger.info('')
            dataset = load_data(datafile,1000000)
            data_x     = dataset[:,0:2] if produceEvent > dataset.shape[0] else dataset[0:produceEvent,0:2]
            data_y     = dataset[:,2:3] if produceEvent > dataset.shape[0] else dataset[0:produceEvent,2:3]
            noise = np.random.uniform(-1, 1, (data_x.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (data_x.shape[0], 1))
            inputs = np.concatenate ((data_x, noise), axis=-1)
            y_pred = sess.run(Pred_y,feed_dict={x:inputs}) 
            y_pred = np.reshape(y_pred, (data_x.shape[0],1))
            output = np.concatenate ((data_x, data_y, y_pred), axis=-1)
            hf = h5py.File(outFile, 'w')
            hf.create_dataset('x_y_pred', data=output)
            hf.close()

        ############## Save the variables to disk.
        if saveCkpt:
            save_path = saver.save(sess, "%s/model.ckpt"%(ckpt_path))
            print("Model saved in path: %s" % save_path)
        if savePb:
            for v in sess.graph.get_operations():
                print(v.name)
            # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Pred/Relu'])
            # 写入序列化的 PB 文件
            with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
    print('done')
