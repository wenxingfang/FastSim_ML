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

###########################
## separate one r,theta array to number of batch
## add event weight 
## add option for selecting training, validing, and testing
## for learning time pdf , pdf(time | r, theta)  #
## shuffle (r,theta)
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
    #return tf.reduce_sum(abs_diff)

def rel_mae_cost(label_y, pred_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs((pred_y - label_y)/label_y)
    return tf.reduce_mean(abs_diff)
    #return tf.reduce_sum(abs_diff)

def rel_mae_cost_w(label_y, pred_y, w):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs((pred_y - label_y)/label_y)*w
    return tf.reduce_mean(abs_diff)
    #return tf.reduce_sum(abs_diff)

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
    
#def Inverse_Linear(x):
#    return 1/(1-x) if x < 0 else x + 1

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
    parser.add_argument('--hidden_layer', action='store', type=int, default=3,
                        help='hidden_layer')
    parser.add_argument('--learning_rate', action='store', type=float, default=3e-4,
                        help='learning_rate')
    parser.add_argument('--head_weight', action='store', type=float, default=1,
                        help='head_weight')
    parser.add_argument('--head_percent', action='store', type=float, default=0.05,
                        help='head_percent')
    parser.add_argument('--neural_units', action='store', type=int, default=12,
                        help='neural_units')
    parser.add_argument('--last_act_mode', action='store', type=int, default=0,
                        help='activation for last layer')
    parser.add_argument('--opt_mode', action='store', type=int, default=0,
                        help='optimizer')
    parser.add_argument('--act_mode', action='store', type=int, default=0,
                        help='activation for hidden layer')
    parser.add_argument('--early_stop_interval', action='store', type=int, default=10,
                        help='early_stop_interval')
    parser.add_argument('--ckpt_path', action='store', type=str,
                        help='ckpt file paths')
    parser.add_argument('--restore_ckpt_path', action='store', type=str,
                        help='restore_ckpt_path file paths')
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
    parser.add_argument('--doTraining', action='store', type=ast.literal_eval, default=True,
                        help='do training')
    parser.add_argument('--doValid', action='store', type=ast.literal_eval, default=True,
                        help='do valid')
    parser.add_argument('--doTest', action='store', type=ast.literal_eval, default=True,
                        help='do test')






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
    act_mode   = parse_args.act_mode
    last_act_mode = parse_args.last_act_mode
    opt_mode      = parse_args.opt_mode
    early_stop_interval = parse_args.early_stop_interval
    datafile = parse_args.datafile
    ckpt_path = parse_args.ckpt_path
    restore_ckpt_path = parse_args.restore_ckpt_path
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
    hidden_layer    = parse_args.hidden_layer
    neural_units    = parse_args.neural_units
    learning_rate   = parse_args.learning_rate
    doTraining      = parse_args.doTraining
    doValid         = parse_args.doValid
    doTest          = parse_args.doTest
    head_weight     = parse_args.head_weight
    head_percent    = parse_args.head_percent
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
    w = tf.placeholder(name='w',shape=(None,1),dtype=tf.float32)
    layer = x
    hidden_activation=tf.nn.tanh
    if act_mode == 1:
        hidden_activation=tf.nn.relu
    for _ in range(hidden_layer):
        layer = tf.layers.dense(inputs=layer, units=neural_units, activation=hidden_activation)
    Pred_y = 0
    if last_act_mode == 0:
        Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x))
    elif last_act_mode == 1:
        Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.elu(x) + 1)
    elif last_act_mode == 2:
        Pred_y = tf.layers.dense(inputs=layer, units=1, activation= Inverse_Linear)
    #cost = ks_cost(Pred_y, y)
    #cost = mse_cost(Pred_y, y)
    #cost = mae_cost(Pred_y, y)
    #cost = rel_mae_cost(y, Pred_y)
    cost = rel_mae_cost_w(y, Pred_y,w)
    #learning_rate = 0.0003
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    if opt_mode == 1:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9, use_nesterov=True).minimize(cost)
    elif opt_mode == 2:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.9, momentum=0.9).minimize(cost)

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
        ############# do training ###########################
        if doTraining:
            print('Do training')
            cost_list = []
            hitTimeData_list = []
            f_DataSet = open(datafile, 'r')
            lines = f_DataSet.readlines()
            for line in lines: 
                idata = line.strip('\n')
                idata = idata.strip(' ')
                if ("#" in idata) or idata=='': continue ##skip the commented one
                logger.info(str(idata))
                try:
                    d = h5py.File(str(idata), 'r')
                    list_key = list(d.keys())
                    for ikey in list_key:
                        if 'HitTimeByPMT' not in ikey: continue
                        #hitTimeData_list.append('%s__%s'%(str(idata),ikey))
                        nsize =  d[ikey].shape[0]
                        n_batch = int(float(nsize)/batch_size)
                        for ib in range(n_batch):
                            hitTimeData_list.append('%s__%s__%d'%(str(idata),ikey, ib))
                        if n_batch == 0:
                            hitTimeData_list.append('%s__%s__-1'%(str(idata),ikey))
                    d.close()
                except:
                    print('bad file:',str(idata))
                else:
                    pass
            f_DataSet.close()
            hitTime_index = list(range(len(hitTimeData_list)))
            print('hitTimeData_list len=', len(hitTime_index))
            logger.info('')

            for epoch in range(epochs):
                total_cost = 0
                count = 0
                np.random.shuffle(hitTime_index)
                for ih in hitTime_index:
                    d = h5py.File(hitTimeData_list[ih].split('__')[0], 'r')
                    ikey = hitTimeData_list[ih].split('__')[1]
                    ib   = int(float(hitTimeData_list[ih].split('__')[2]))
                    value = shuffle(d[ikey][:]/100, random_state=epoch)
                    d.close()
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
                    #n_batch = int(float(value.shape[0])/batch_size)
                    if ib != -1 :
                        train_x = inputs [ib * batch_size:(ib + 1) * batch_size]
                        train_y = value  [ib * batch_size:(ib + 1) * batch_size]
                        #if np.any(np.isnan(train_x)): print('find Nan in train_x')
                        #if np.any(np.isnan(train_y)): print('find Nan in train_y')
                        we = np.array([1.0])
                        we = np.reshape(we, (1,1))
                        we = we.repeat(train_x.shape[0],axis=0)
                        we_s = list(range(int(head_percent*we.shape[0])))
                        we[we_s,0] = head_weight ## use weight 10 for begin 5% event
                        _, c = sess.run([optimizer,cost], feed_dict={x:train_x, y:train_y, w:we})
                        total_cost += c
                        count = count + 1
                    else :
                        train_x = inputs
                        train_y = value 
                        we = np.array([1.0])
                        we = np.reshape(we, (1,1))
                        we = we.repeat(train_x.shape[0],axis=0)
                        we_s = list(range(int(head_percent*we.shape[0])))
                        we[we_s,0] = head_weight ## use weight 2 for begin 5% event
                        #if np.any(np.isnan(train_x)): print('find Nan in train_x')
                        #if np.any(np.isnan(train_y)): print('find Nan in train_y')
                        _, c = sess.run([optimizer,cost], feed_dict={x:train_x, y:train_y, w:we})
                        total_cost += c
                        count = count + 1

                avg_cost = total_cost/count
                if len(cost_list) < early_stop_interval: cost_list.append(avg_cost)
                else:
                    for ic in range(len(cost_list)-1):
                        cost_list[ic] = cost_list[ic+1]
                    cost_list[-1] = avg_cost
                if epoch % 1 == 0:
                    print('act mode {0}, Epoch {1} | cost = {2:.4f}'.format(last_act_mode,epoch,avg_cost))
                    save_path = saver.save(sess, "%s/model.ckpt"%(ckpt_path))
                    print("Model saved in path: %s" % save_path)
                    logger.info('')
                if epoch > len(cost_list) and avg_cost >= cost_list[0]: 
                    print('stop here. Epoch {0} | cost = {1:.4f}, ref cost = {2:.4f}'.format(epoch,avg_cost,cost_list[0]))
                    logger.info('')
                    break
        ### validation ############################
        if doValid:
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
                    if 'HitTimeByPMT' not in ikey: continue
                    value = d[ikey][:]/100
                    #value = (100/(d[ikey][:]+1e-6))
                    value = np.reshape(value, (-1,1))
                    r     = np.array( [(float(ikey.split('_')[1])-10)/10]) if norm_mode==0 else  np.array( [float(ikey.split('_')[1])/20] )
                    theta = np.array( [(float(ikey.split('_')[2])-90)/90]) if norm_mode==0 else  np.array( [float(ikey.split('_')[2])/180])  #normalize
                    r     = np.reshape(r    , (1,1))
                    theta = np.reshape(theta, (1,1))
                    r     = r    .repeat(value.shape[0],axis=0)
                    theta = theta.repeat(value.shape[0],axis=0)
                    noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
                    inputs = np.concatenate ((r, theta, noise), axis=-1)
                    '''
                    n_batch = int(float(value.shape[0])/batch_size)
                    for ib in range(n_batch):
                        valid_x = inputs [ib * batch_size:(ib + 1) * batch_size]
                        valid_y = value  [ib * batch_size:(ib + 1) * batch_size]
                        if np.any(np.isnan(valid_x)): print('find Nan in valid_x')
                        if np.any(np.isnan(valid_y)): print('find Nan in valid_y')
                        pred, c = sess.run([Pred_y, cost], feed_dict={x:valid_x, y:valid_y})
                        cost_valid += c
                        count = count + 1
                    '''

                    train_x = inputs
                    train_y = value 
                    we = np.array([1.0])
                    we = np.reshape(we, (1,1))
                    we = we.repeat(train_x.shape[0],axis=0)
                    we_s = list(range(int(head_percent*we.shape[0])))
                    we[we_s,0] = head_weight ## use weight 2 for begin 5% event
                    _, c = sess.run([Pred_y, cost], feed_dict={x:train_x, y:train_y, w:we})
                    cost_valid += c
                    count = count + 1
                d.close()
            f_DataSet.close()
            avg_cost = cost_valid/count
            print('ave valid cost = {0:.4f}'.format(avg_cost))
        #### produce predicted data #################
        if doTest:
            print('Saving produced data')
            logger.info('')
            hf = h5py.File(outFileName, 'w')
            r_theta = []
            f_DataSet = open(test_file, 'r')
            Diffs = []
            for line in f_DataSet: 
                idata = line.strip('\n')
                idata = idata.strip(' ')
                if "#" in idata: continue ##skip the commented one
                d = h5py.File(str(idata), 'r')
                for ikey in d.keys():
                    if 'HitTimeByPMT' not in ikey: continue
                    #real         = d[ikey][:]          ## for check
                    #produceEvent = real.shape[0] ## for check
                    str_r_theta = '%s_%s'%(str(ikey.split('_')[1]),str(ikey.split('_')[2]))
                    if str_r_theta in r_theta: continue
                    r_theta.append(str_r_theta)

                    r     = np.array( [(float(ikey.split('_')[1])-10)/10]) if norm_mode==0 else  np.array( [float(ikey.split('_')[1])/20] )
                    theta = np.array( [(float(ikey.split('_')[2])-90)/90]) if norm_mode==0 else  np.array( [float(ikey.split('_')[2])/180])  #normalize
                    r     = np.reshape(r    , (1,1))
                    theta = np.reshape(theta, (1,1))
                    r     = r    .repeat(produceEvent,axis=0)
                    theta = theta.repeat(produceEvent,axis=0)
                    noise = np.random.uniform(-1, 1, (produceEvent, 1)) if use_uniform else np.random.normal(0, 1, (produceEvent, 1))
                    inputs = np.concatenate ((r, theta, noise), axis=-1)
                    y_pred = sess.run(Pred_y,feed_dict={x:inputs})
                    y_pred = np.reshape(y_pred, produceEvent)*100 # back to unnormalized value
                    #y_pred = 100/np.reshape(y_pred, produceEvent) # back to unnormalized value
                    hf.create_dataset('predHitTimeByPMT_%s'%str_r_theta, data=y_pred)
                    
                    #Diff = np.mean(np.abs(-np.sort(real)+np.sort(y_pred))/np.sort(real))
                    #Diffs.append(Diff) 
                d.close()
            f_DataSet.close()
            hf.close()
            #print('test diffs=', Diffs)
            #print('test diffs mean=', np.mean(Diffs))
            
            print('Saved produced data %s'%outFileName)
        ############## Save the variables to disk.
        if saveCkpt:
            save_path = saver.save(sess, "%s/model.ckpt"%(ckpt_path))
            print("Model saved in path: %s" % save_path)
        if savePb:
            for v in sess.graph.get_operations():
                print(v.name)
            # convert_variables_to_constants 需要指定output_node_names，list()，可以多个
            #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Pred/Relu'])
            #constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dense_2/add'])
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dense_3/add'])
            # 写入序列化的 PB 文件
            with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            print('saved %s'%pb_file_path)
    print('done')
