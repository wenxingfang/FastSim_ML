import h5py
import ast
import sys
import argparse
import numpy as np
import math
from sklearn.utils import shuffle
import random
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
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

class Loss_mae_v3(nn.Module):
    def __init__(self):
        super(Loss_mae_v3, self).__init__()

    def forward(self, pred, label, w):
        pred, _  = torch.sort(pred , dim=0, descending=False, out=None)
        label, _ = torch.sort(label, dim=0, descending=False, out=None)
        abs_diff = torch.abs((pred - label)/label)*w 
        return torch.mean(abs_diff)

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
    parser.add_argument('--saveToCPU', action='store', type=ast.literal_eval, default=False,
                        help='saveToCPU')
    parser.add_argument('--Restore', action='store', type=ast.literal_eval, default=False,
                        help='ckpt file paths')
    parser.add_argument('--use_uniform', action='store', type=ast.literal_eval, default=True,
                        help='use uniform noise')
    parser.add_argument('--produceEvent', action='store', type=int,
                        help='produceEvent')
    parser.add_argument('--outFileName', action='store', type=str,
                        help='outFileName file paths')
    parser.add_argument('--pt_file_path', action='store', type=str,
                        help='pt_file_path file paths')
    parser.add_argument('--KitModelPath', action='store', type=str,
                        help='KitModelPath file paths')
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
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
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
    saveToCPU    = parse_args.saveToCPU
    Restore   = parse_args.Restore
    use_uniform   = parse_args.use_uniform
    norm_mode     = parse_args.norm_mode
    produceEvent = parse_args.produceEvent
    outFileName = parse_args.outFileName
    pt_file_path = parse_args.pt_file_path
    KitModelPath = parse_args.KitModelPath
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
    sys.path.append(KitModelPath)
    import converted_pytorch
    net = converted_pytorch.KitModel(restore_ckpt_path)
    print(net)
    print(net.parameters())
    net.to(device)
    #cost = rel_mae_cost_w(y, Pred_y,w)
    cost = Loss_mae_v3()
    cost.to(device)
    ########################################
    print('commencing training')
    logger.info('')
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
                train_x = inputs
                train_y = value 
                train_x_t = torch.Tensor(train_x)
                train_y_t = torch.Tensor(train_y)
                train_x_t, train_y_t = train_x_t.to(device), train_y_t.to(device)
                outputs = net(train_x_t)
                we = np.array([1.0])
                we = np.reshape(we, (1,1))
                we = we.repeat(train_x.shape[0],axis=0)
                we_s = list(range(int(head_percent*we.shape[0])))
                we[we_s,0] = head_weight ## use weight 2 for begin 5% event
                we_t = torch.Tensor(we)
                we_t = we_t.to(device)
                loss = cost(outputs, train_y_t, we_t)
                loss_c = loss.cpu()
                cost_valid = cost_valid + loss_c.detach().numpy() if cost_valid is not None else loss_c.detach().numpy()
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
                train_x_t = torch.Tensor(inputs)
                train_x_t = train_x_t.to(device)
                outputs = net(train_x_t)
                y_pred = outputs.cpu()
                y_pred = y_pred.detach().numpy()
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
    ##################################
    r     = np.array( [16,16,16] )/20 
    theta = np.array( [0.1,0.13,0.15] )*180/(math.pi*180) 
    noise = np.array( [0.5, 0.5, 0.5] ) 
    r     = np.reshape(r    , (3,1))
    theta = np.reshape(theta, (3,1))
    noise = np.reshape(noise, (3,1))
    inputs= np.concatenate ((r, theta, noise), axis=-1)
    train_x_t = torch.Tensor(inputs)
    train_x_t = train_x_t.to(device)
    outputs = net(train_x_t)
    y_pred = outputs.cpu()
    y_pred = y_pred.detach().numpy()*100
    print("perd from test: ", y_pred)
    ############## Save to cpu######.
    if saveToCPU:
        device = torch.device("cpu")
        net.to(device)
        model = torch.jit.script(net)
        model.save(pt_file_path)
    print('done')
