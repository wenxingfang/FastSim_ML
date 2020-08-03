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
#########################################################
import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization
###########################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
###########################
import faiss                   # make faiss available
###########################
## for learning time pdf , pdf(time | pmt_theta, pmt_phi, src_r, src_theta, src_phi)  #
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

def mae_cost(label_y, pred_y):
    pred_y  = tf.sort(pred_y , axis=0,direction='ASCENDING',name=None)
    label_y = tf.sort(label_y, axis=0,direction='ASCENDING',name=None)
    abs_diff = tf.math.abs(pred_y - label_y)
    #return tf.reduce_mean(abs_diff)
    return tf.reduce_sum(abs_diff)

class rel_mae_cost(nn.Module):
    def __init__(self):
        super(rel_mae_cost, self).__init__()

    def forward(self, pred_y, label_y):
        pred_y, _  = torch.sort(pred_y  , dim=0, descending=False, out=None)
        label_y, _ = torch.sort(label_y , dim=0, descending=False, out=None)
        abs_diff = torch.abs((pred_y - label_y)/label_y)
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
    parser.add_argument('--max_evt', action='store', type=int,
                        help='max_evt')
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
    parser.add_argument('--num_trials', action='store', type=int, default=40,
                        help='num of trials.')






    parser.add_argument('--disc-lr', action='store', type=float, default=2e-5,
                        help='Adam learning rate for discriminator')

    parser.add_argument('--gen-lr', action='store', type=float, default=2e-4,
                        help='Adam learning rate for generator')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    return parser


class Net(torch.jit.ScriptModule):
    __constants__ = ['fcs', 'act', 'flayer', 'f1']
    def __init__(self, N_hidden, N_ne, act):
        super(Net, self).__init__()
        self.hlayers = N_hidden
        self.act    = act
        self.llayer = nn.Linear(N_ne,1)
        self.flayer = nn.Sequential( nn.Linear(6,N_ne),
                                      nn.ReLU()
                                    )
        l_list = []
        for i in range(self.hlayers):
                l_list.append(nn.Linear(N_ne, N_ne))
                l_list.append(nn.ReLU())
        self.fcs = nn.ModuleList(l_list)
        self.f1  = nn.ModuleList([self.llayer])
    @torch.jit.script_method
    def forward(self, x):
        x = self.flayer(x)
        for i in self.fcs:
            x = i(x)
        x = self.llayer(x)
        x = F.elu(x) + 1 
        return x



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
    num_trials      = parse_args.num_trials
    max_evt         = parse_args.max_evt
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
                  sherpa.Continuous('learning_rate', [1e-7, 1e-2], scale='log'),
                  sherpa.Discrete('num_units', [100, 1000]),
                  sherpa.Discrete('hidden_layer', [1, 4]),
                  #sherpa.Choice('initializers', [keras.initializers.RandomNormal(), keras.initializers.RandomUniform(), keras.initializers.lecun_normal()]),
                  #sherpa.Choice('initializers', [keras.initializers.RandomNormal(), keras.initializers.lecun_normal()]),
                  #sherpa.Choice('activation', ['relu', 'tanh'])
                  #sherpa.Choice('activation', ['relu', 'selu'])
                  #sherpa.Choice('activation', ['relu', 'elu','selu'])
                  #sherpa.Choice('activation', ['relu', 'tanh']),
                  #sherpa.Choice('activation', ['relu', 'tanh', 'sigmoid']),
                  #sherpa.Choice('last_activation', [Exponential_Linear]) # choosed
                  #sherpa.Choice('last_activation', ['relu'])
                 ]
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=num_trials)
    study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     disable_dashboard=True,
                     output_dir = output_dir,
                     lower_is_better=True)


    #############################
    logger.info('Preparing data list')
    Data_list = []
    f_DataSet = open(datafile, 'r')
    lines = f_DataSet.readlines()
    for line in lines: 
        idata = line.strip('\n')
        idata = idata.strip(' ')
        if ("#" in idata) or idata=='': continue ##skip the commented one
        Data_list.append(str(idata))
    f_DataSet.close()
    print('Data_list =', Data_list)
    logger.info('')
    #####################################
    logger.info('Start trialing')
    i_trial = 1
    for trial in study:
        print('trial=',i_trial)
        logger.info('')
        lr = trial.parameters['learning_rate']
        num_units = int(trial.parameters['num_units'])
        #act = trial.parameters['activation']
        #init = trial.parameters['initializers']
        #last_act = trial.parameters['last_activation']
        n_hidden = int(trial.parameters['hidden_layer'])
        print('Creating model:n_hidden=',n_hidden,',num_units=',num_units,',lr=',lr)
        logger.info('')
        net = Net(N_hidden=n_hidden, N_ne=num_units, act='')
        optimizer = optim.Adam(net.parameters(), lr=lr)
        criterion = rel_mae_cost()
        net.to(device)
        criterion.to(device)
    #####################################
        # Train model
        logger.info('Start training model')
        cost_list = []
        ealry_stop = False
        k = batch_size
        for epoch in range(epochs):
            total_cost = 0
            count = 0
            for ifile in Data_list:
                print('h5 file=',ifile)
                h5 = h5py.File(ifile, 'r')
                df = h5['data'][0:max_evt,] if max_evt<h5['data'].shape[0] else h5['data'][:]
                h5.close()
                df = shuffle(df)
                df = df.astype('float32')
                df = np.ascontiguousarray(df)
                print('df shape=',df.shape)
                xb = np.ascontiguousarray(df[:,0:5]) #src_r, src_theta, src_phi, pmt_theta, pmt_phi
                print('xb shape=',xb.shape)
                index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), xb.shape[1]) # build the index
                print(index.is_trained)
                index.add(xb)
                used = []
                for i in range(df.shape[0]):
                    if(i%10000==0):
                        print('processed %f %%'%(100*float(i)/df.shape[0]))
                        logger.info('')
                    if i in used:continue
                    D, I = index.search(df[i:i+1,0:5], k)     # actual search
                    I = np.squeeze(I)
                    used.extend(I)
                    value = df[I,0:5]
                    noise  = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
                    inputs = np.concatenate ((value, noise), axis=-1)
                    train_x = inputs
                    train_y = df[I,5:6]
                    train_x_t = torch.Tensor(train_x)
                    train_y_t = torch.Tensor(train_y)
                    train_x_t, train_y_t = train_x_t.to(device), train_y_t.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(train_x_t)
                    loss = criterion(outputs, train_y_t)
                    loss.backward()
                    optimizer.step()
                    loss_c = loss.cpu()
                    total_cost = total_cost + loss_c.detach().numpy() if total_cost is not None else loss_c.detach().numpy()
                    count = count + 1
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
            if epoch > len(cost_list) and avg_cost >= cost_list[0]: ealry_stop = True
            ############ study ###################
            #study.add_observation(trial=trial, iteration=epoch, objective=avg_cost, context={'loss': avg_cost})
            study.add_observation(trial=trial, iteration=epoch, objective=avg_cost)
            if study.should_trial_stop(trial) or ealry_stop:
                break 
        study.finalize(trial=trial)
        print('act mode=,',opt_mode,', get_best_result()=\n',study.get_best_result())
        i_trial += 1
    print('save resluts to %s'%output_dir)
    study.save()
    #####################################
    print('done')
