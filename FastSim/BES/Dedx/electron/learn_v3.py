import h5py
import sys
import argparse
import numpy as np
import math
from sklearn.utils import shuffle
import random
import logging
import os
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import faiss                   # make faiss available

#faiss.omp_set_num_threads(1)
###########################
## like Res net
## add min dis cut
## sikp used points for each epoch
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


def load_data(data, max_n):
    print('load data:',data)
    d = h5py.File(data, 'r')
    #df = d['dataset'][:]
    max_evt = max_n if max_n < d['dataset'].shape[0] else d['dataset'].shape[0]
    df = d['dataset'][:max_evt,]
    d.close()
    print("df shape:",df.shape)
    #df = shuffle(df)
    return np.ascontiguousarray(df)

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
    parser.add_argument('--saveToCPU', action='store', type=ast.literal_eval, default=False,
                        help='saveToCPU file paths')
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
    parser.add_argument('--pt_file_path', action='store', type=str,
                        help='pt_file_path file paths')
    parser.add_argument('--validation_file', action='store', type=str,
                        help='validation_file file paths')
    parser.add_argument('--test_file', action='store', type=str,
                        help='test_file file paths')

    parser.add_argument('--lr', action='store', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--Scale', action='store', type=float, default=1,
                        help='scale npe')
    parser.add_argument('--max_training_evt', action='store', type=int, default=10000,
                        help='max_training_evt')
    parser.add_argument('--max_valid_evt', action='store', type=int, default=100000,
                        help='max_valid_evt')
    parser.add_argument('--reLoadIndex', action='store', type=ast.literal_eval, default=False,
                        help='reLoadIndexFile')
    parser.add_argument('--IndexFile', action='store', type=str,
                        help='IndexFile')
    parser.add_argument('--MinD', action='store', type=float, default=0.1,
                        help='min distance')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    return parser



class Net(nn.Module):
    def __init__(self, N_layer, N_ne, act):
        super(Net, self).__init__()
        self.layers = N_layer
        self.act    = act
        self.fcs = nn.ModuleList()
        for i in range(self.layers):
            if i == 0: self.fcs.append(nn.Linear(3,N_ne))
            elif i == (self.layers-1): self.fcs.append(nn.Linear(N_ne, 1))
            else                     : self.fcs.append(nn.Linear(N_ne, N_ne))

    def forward(self, x):
        y = x
        #print('x size=',x.size())
        if self.act == 0:
            for i in range(self.layers-1):
                #print(self.fcs[i])
                y = F.tanh(self.fcs[i](y))
        elif self.act == 1:
            for i in range(self.layers-1):
                y = F.elu(self.fcs[i](y))
        elif self.act == 2:
            for i in range(self.layers-1):
                y = F.relu(self.fcs[i](y))
        else: 
            print('wrong act value')
            assert 0
        y = F.relu(self.fcs[self.layers-1](y))
        return y

class Nett(torch.jit.ScriptModule):
    #__constants__ = ['fcs']
    __constants__ = ['fcs', 'act']
    def __init__(self, N_layer, N_ne, act):
        super(Nett, self).__init__()
        self.layers = N_layer
        self.act    = act
        l_list = []
        for i in range(self.layers):
            if i == 0: 
                l_list.append(nn.Linear(3,N_ne))
                #l_list.append(nn.Tanh())
                l_list.append(nn.ReLU())
            elif i == (self.layers-1): 
                l_list.append(nn.Linear(N_ne, 1))
                #l_list.append(nn.ReLU())
            else                     : 
                l_list.append(nn.Linear(N_ne, N_ne))
                #l_list.append(nn.Tanh())
                l_list.append(nn.ReLU())
        self.fcs = nn.ModuleList(l_list)
    @torch.jit.script_method
    def forward(self, x):
        y = x
        #print('x size=',x.size())
        #if self.act == 0:
        for i in self.fcs:
            y = i(y)
        #elif self.act == 1:
        #    for i in range(self.layers-1):
        #        y = F.elu(self.fcs[i](y))
        #elif self.act == 2:
        #    for i in range(self.layers-1):
        #        y = F.relu(self.fcs[i](y))
        #else: 
        #    print('wrong act value')
        #    assert 0
        return y

class Net1(torch.jit.ScriptModule):
    __constants__ = ['fcs', 'act','fc_last']
    def __init__(self, N_layer, N_ne, act):
        super(Net1, self).__init__()
        self.layers = N_layer
        self.act    = act
        l_list = []
        l_list_last = []
        for i in range(self.layers-1):
            if i == 0: 
                l_list.append(nn.Linear(3,N_ne))
                l_list.append(nn.ReLU())
            else                     : 
                l_list.append(nn.Linear(N_ne, N_ne))
                l_list.append(nn.ReLU())
        l_list_last.append(nn.Linear(N_ne+2, 1))#2 is for pt and theta 
        self.fcs     = nn.ModuleList(l_list)
        self.fc_last = nn.ModuleList(l_list_last)
        print('len fcs=',len(self.fcs),',fc_last=',len(self.fc_last))
    @torch.jit.script_method
    def forward(self, x):
        y  = x
        y0 = x[:,0:2]
        for i in self.fcs:
            y = i(y)
        for i in self.fc_last:
            third_tensor = torch.cat((y, y0), -1)
            y = i(third_tensor)
        return y

                
            
class Loss_mae(nn.Module):
    def __init__(self):
        super(Loss_mae, self).__init__()

    def forward(self, pred, label):
        pred, _  = torch.sort(pred , dim=0, descending=False, out=None)
        label, _ = torch.sort(label, dim=0, descending=False, out=None)
        abs_diff = torch.abs(pred - label)
        #abs_diff = torch.abs(pred - label) + 0.5
        #abs_diff = torch.pow(abs_diff, 4)
        return torch.sum(abs_diff)
 



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
    datafile = parse_args.datafile
    ckpt_path = parse_args.ckpt_path
    restore_ckpt_path = parse_args.restore_ckpt_path
    saveCkpt  = parse_args.saveCkpt
    saveToCPU    = parse_args.saveToCPU
    Restore   = parse_args.Restore
    doTraining    = parse_args.doTraining
    doValid       = parse_args.doValid
    doTest        = parse_args.doTest
    use_uniform   = parse_args.use_uniform
    produceEvent = parse_args.produceEvent
    outFile = parse_args.outFile
    pt_file_path = parse_args.pt_file_path
    validation_file = parse_args.validation_file
    test_file       = parse_args.test_file
    early_stop_interval       = parse_args.early_stop_interval
    act_mode        = parse_args.act_mode
    N_units         = parse_args.N_units
    N_hidden        = parse_args.N_hidden
    lr              = parse_args.lr
    cost_mode       = parse_args.cost_mode
    Scale           = parse_args.Scale
    max_training_evt          = parse_args.max_training_evt
    max_valid_evt          = parse_args.max_valid_evt
    reLoadIndex  = parse_args.reLoadIndex
    IndexFile  = parse_args.IndexFile
    MinD       = parse_args.MinD
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

    '''
    tf.reset_default_graph()
    x = tf.placeholder(name='x',shape=(None,3),dtype=tf.float32)
    y = tf.placeholder(name='y',shape=(None,1),dtype=tf.float32)
    layer = x
    act = {0:tf.nn.tanh, 1:tf.nn.elu, 2:tf.nn.relu}
    for _ in range(N_hidden):
        layer = tf.layers.dense(inputs=layer, units=N_units, activation=act[act_mode])
    Pred_y = tf.layers.dense(inputs=layer, units=1, activation=lambda x: tf.nn.relu(x), name = 'Pred')
    '''
    #net = Nett(N_hidden+1, N_units, act_mode)
    net = Net1(N_hidden+1, N_units, act_mode)
    print(net)
    print(net.parameters())
    criterion = Loss_mae()
    print(criterion)
    '''
    cost = mae_cost(Pred_y, y)
    if cost_mode == 1:
        cost = mae_cost_v1(Pred_y, y)
    elif cost_mode == 2:
        cost = mae_cost_v2(Pred_y, y)
    elif cost_mode == 3:
        cost = mae_cost_v3(Pred_y, y)
    '''
    #learning_rate = lr
    #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #saver = tf.train.Saver()
    ########################################
    net.to(device)
    criterion.to(device)
    #optimizer.to(device)
    print('commencing training')
    logger.info('')
    if Restore:
        #ckpt = tf.train.get_checkpoint_state("%s"%restore_ckpt_path)
        #saver.restore(sess, ckpt.all_model_checkpoint_paths[0])
        print('restored from ',ckpt_path)
        net.load_state_dict(torch.load(restore_ckpt_path))
        #net.eval()
    #else:
    #    sess.run(tf.global_variables_initializer())
    if doTraining:
        print('preparing data list for training')
        logger.info('')
        cost_list = []
        df = load_data(datafile, max_training_evt)
        print('preparing index')
        df = df.astype('float32')
        xb = np.ascontiguousarray(df[:,0:2])
        print('shape=',xb.shape)
        index = faiss.IndexFlatL2(xb.shape[1]) # build the index
        print(index.is_trained)
        logger.info('')
        index.verbose = True 
        if reLoadIndex==False:
            index.add(xb)                  # add vectors to the index
            faiss.write_index(index, IndexFile)
        else:
            index = faiss.read_index(IndexFile)
        print('ntoal index=',index.ntotal)
        logger.info('')
        k = batch_size
        for epoch in range(epochs):
            total_cost = None
            count = 0
            used = []
            for i in range(df.shape[0]):
                #print('i=',i)
                if(i%10000==0): 
                    print('processed %f %%'%(100*float(i)/df.shape[0]))
                    logger.info('')
                if i in used:continue
                D, I = index.search(df[i:i+1,0:2], k)     # actual search
                I = np.squeeze(I)
                D = np.squeeze(D)
                #print('I=',len(I))
                #print('D=',D, ',len D=',len(D))
                #logger.info('')
                I_new = []
                for d in range(len(D)):
                    if D[d] < MinD: I_new.append(I[d])
                #print('I_new=',len(I_new))
                #logger.info('')
                used.extend(I_new)
                #print('D=',D)
                #print('I=',I.shape)
                value = df[I_new,0:2]
                #print('value=',value)
                #print(value.shape)
                noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
                inputs = np.concatenate ((value, noise), axis=-1)

                train_x = inputs
                train_y = df[I_new,2:3]
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
            if epoch % 1 == 0 and saveCkpt:
                print('Epoch {0} | cost = {1:.4f}'.format(epoch,avg_cost))
                #save_path = saver.save(sess, "%s/model.ckpt"%(ckpt_path))
                #PATH = './cifar_net.pth'
                torch.save(net.state_dict(), ckpt_path)
                print("Model saved in path: %s" % ckpt_path)
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
        print('Do validation')
        logger.info('')
        cost_valid = None 
        count = 0
        df = load_data(validation_file, max_valid_evt)
        n_batch = int(float(df.shape[0])/batch_size)
        print('n_batch=',n_batch)
        for i in range(n_batch): 
            value = df[i*batch_size:(1+i)*batch_size,0:2]
            noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
            inputs = np.concatenate ((value, noise), axis=-1)
            train_x = inputs
            train_y = df[i*batch_size:(1+i)*batch_size,2:3]
            train_x_t = torch.Tensor(train_x)
            train_y_t = torch.Tensor(train_y)
            train_x_t, train_y_t = train_x_t.to(device), train_y_t.to(device)
            outputs = net(train_x_t)
            loss = criterion(outputs, train_y_t)
            loss_c = loss.cpu()
            cost_valid = cost_valid + loss_c.detach().numpy() if cost_valid is not None else loss_c.detach().numpy()
            count = count + 1
        avg_cost = cost_valid/count
        print('ave valid cost = {0:.4f}'.format(avg_cost))
    #### produce predicted data #################
    if doTest:
        print('Saving produced data')
        logger.info('')
        df = load_data(test_file, produceEvent)
        #produceEvent = produceEvent if df.shape[0] >= produceEvent else df.shape[0] 
        value = df[:, 0:2]
        noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
        inputs = np.concatenate ((value, noise), axis=-1)
        train_x_t = torch.Tensor(inputs)
        train_x_t = train_x_t.to(device)
        outputs = net(train_x_t)
        y_pred = outputs.cpu()
        y_pred = y_pred.detach().numpy()
        y_pred = np.reshape(y_pred, (value.shape[0], 1))
        final_out = np.concatenate ((df[:, 0:3], y_pred), axis=-1)

        hf = h5py.File(outFile, 'w')
        hf.create_dataset('Pred', data=final_out)
        hf.close()
        print('Saved produced data %s'%outFile)
    ############## Save the variables to disk.
    if saveToCPU:
        device = torch.device("cpu")
        net.to(device)
        model = torch.jit.script(net)
        model.save(pt_file_path)
    print('done')
