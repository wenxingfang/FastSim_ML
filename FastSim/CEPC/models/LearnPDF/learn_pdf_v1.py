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
###########################
## use pre defined noise 
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
    noise = d['noise'][:]
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
    first, mc_info, noise = shuffle(first, mc_info, noise)
    return first, mc_info, noise

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

    input_Info  = Input(name='input_Info', shape=(10,))
    


    #x = tf.placeholder(name='x',shape=(None,10),dtype=tf.float32)
    #y_E = tf.placeholder(name='y_E',shape=(None,500),dtype=tf.float32)
    #y_x = tf.placeholder(name='y_x',shape=(None,500),dtype=tf.float32)
    #y_y = tf.placeholder(name='y_y',shape=(None,500),dtype=tf.float32)
    #y_z = tf.placeholder(name='y_z',shape=(None,500),dtype=tf.float32)
    #layer = x
    #act = {0:tf.nn.tanh, 1:tf.nn.elu, 2:tf.nn.relu}
    act = {0:'tanh', 1:'elu', 2:'relu'}
    layer = Dense(N_units, activation=act[act_mode])(input_Info)
    for _ in range(N_hidden-1):
        #layer = tf.layers.dense(inputs=layer, units=N_units, activation=act[act_mode])
        layer = Dense(N_units, activation=act[act_mode])(layer)
        #layer = tf.layers.dense(inputs=layer, units=12, activation=tf.nn.tanh)
    #Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.elu(x) + 1)
    Pred_x = Dense(500, name = 'Pred_x')(layer)
    con_Pred_y = concatenate( [Pred_x, layer] )
    Pred_y = Dense(500, name = 'Pred_y')(con_Pred_y)
    con_Pred_z = concatenate( [Pred_x, Pred_y, layer] )
    Pred_z = Dense(500, name = 'Pred_z')(con_Pred_z)
    con_Pred_E = concatenate( [Pred_x, Pred_y, Pred_z, layer] )
    #Pred_E = Dense(500, activation='relu', name = 'Pred_E')(con_Pred_E)
    Pred_E = Dense(500, activation=Exponential_Linear, name = 'Pred_E')(con_Pred_E)
    Predictor = 0
    if Restore==False:
        print('creating new model')
        Predictor = Model(input_Info , [Pred_x, Pred_y, Pred_z, Pred_E], name='predictor')
        Predictor.compile(
            optimizer=Adam(lr=learning_rate),
            loss=['mae','mae','mae', mae_cost_1]
        )

    else:
        print('restore model:',restore_model_in)
        Predictor = load_model(restore_model_in)
        Predictor.trainable = True
    '''
    #cost = mse_cost(Pred_y, y)
    cost = mae_cost(Pred_E, y_E) + mae_cost(Pred_x, y_x) + mae_cost(Pred_y, y_y) + mae_cost(Pred_z, y_z)
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
    '''

    ########################################
    print('commencing training')
    logger.info('')
    if doTraining:
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
        cost_list = []
        epoch_loss = []
        loss_weights      = [np.ones(batch_size), 1*np.ones(batch_size), np.ones(batch_size), 1*np.ones(batch_size) ]
        for epoch in range(epochs):
            logger.info('Epoch {} of {}'.format(epoch + 1, epochs))
            nb_batches = sum(Batch)
            for ib in range(len(Batch)):
                first, mc_info, noise_info = load_data(Data[ib])
                ibatch = Batch[ib]
                for index in range(ibatch):
                    hit_batch   =  first     [index * batch_size:(index + 1) * batch_size]
                    info_batch  =  mc_info   [index * batch_size:(index + 1) * batch_size]
                    noise_batch =  noise_info[index * batch_size:(index + 1) * batch_size]
                    #noise = np.random.normal(0, 1, (batch_size, 4))
                    inputs= np.concatenate((info_batch, noise_batch), axis=-1)
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
            if epoch > len(cost_list) and avg_cost >= cost_list[0]: break

    ### validation ############################
        if doValid:
            print('Do validation')
            logger.info('')
            f_DataSet = open(validation_file, 'r')
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
            print('valid total sample:', total_event)
            print('valid all Batch:', Batch)
            logger.info('')
            valid_loss = []
            loss_weights      = [np.ones(batch_size), 1*np.ones(batch_size), np.ones(batch_size), 1*np.ones(batch_size) ]
            nb_batches = sum(Batch)
            for ib in range(len(Batch)):
                first, mc_info, noise_info = load_data(Data[ib])
                ibatch = Batch[ib]
                for index in range(ibatch):
                    hit_batch   =  first     [index * batch_size:(index + 1) * batch_size]
                    info_batch  =  mc_info   [index * batch_size:(index + 1) * batch_size]
                    noise_batch =  noise_info[index * batch_size:(index + 1) * batch_size]
                    #noise = np.random.normal(0, 1, (batch_size, 4))
                    inputs= np.concatenate((info_batch, noise_batch), axis=-1)
                    outputs_label = [ hit_batch[:,:,0], hit_batch[:,:,1], hit_batch[:,:,2], hit_batch[:,:,3] ]
                    batch_loss = Predictor.test_on_batch(
                    inputs,
                    outputs_label,
                    loss_weights
                    )
                    valid_loss.append(np.array(batch_loss))
            logger.info('valid loss: {}'.format( np.mean(valid_loss, axis=0)))
        #### produce predicted data #################
        if doTest:
            print('Saving produced data')
            logger.info('')
            first, mc_info, noise_info = load_data(test_file)
            noise = np.random.normal(0, 1, (mc_info.shape[0], 4))
            inputs= np.concatenate((mc_info, noise), axis=-1)
            preds = Predictor.predict_on_batch(inputs)
            preds[0] = preds[0]*10 + 1850
            preds[1] = preds[1]*10
            preds[2] = preds[2]*10
            preds[0] = np.expand_dims(preds[0], axis=-1)
            preds[1] = np.expand_dims(preds[1], axis=-1)
            preds[2] = np.expand_dims(preds[2], axis=-1)
            preds[3] = np.expand_dims(preds[3], axis=-1)
            preds= np.concatenate((preds[0], preds[1], preds[2], preds[3] ), axis=-1)
            #out_str = in_str.replace('.h5','_pred.h5')
            #out_str = 'Pred.h5'
            #out_str = outFilePath+'/'+out_str
            out_str = outFilePath
            hf = h5py.File(out_str, 'w')
            hf.create_dataset('Barrel_Hit', data=preds)
            hf.create_dataset('MC_info'   , data=mc_info)
            hf.close()
            print('Saved produced data %s'%out_str)
        ############## Save the variables to disk.
        Predictor.save(model_out)
        '''
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
        '''
    print('done')
