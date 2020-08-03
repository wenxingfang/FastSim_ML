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
from typing import List, Dict, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.tensor import Tensor

import faiss                   # make faiss available
## do shuflle for each epoch

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
    max_evt = max_n if max_n < d['dataset'].shape[0] else d['dataset'].shape[0]
    df = d['dataset'][:max_evt,]
    d.close()
    df = shuffle(df)
    print("df shape:",df.shape)
    return np.ascontiguousarray(df)

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
    parser.add_argument('--g_hidden_size', action='store', type=int, default=15,
                        help='g_hidden_size')
    parser.add_argument('--cost_mode', action='store', type=int, default=0,
                        help='cost_mode')
    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')
    parser.add_argument('--D_restore_ckpt_path', action='store', type=str,
                        help='D restore_ckpt_path ckpt file paths')
    parser.add_argument('--D_ckpt_path', action='store', type=str,
                        help='D ckpt_path ckpt file paths')
    parser.add_argument('--G_restore_ckpt_path', action='store', type=str,
                        help='G restore_ckpt_path ckpt file paths')
    parser.add_argument('--G_ckpt_path', action='store', type=str,
                        help='G ckpt_path ckpt file paths')
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

    parser.add_argument('--d_learning_rate', action='store', type=float, default=2e-4,
                        help='d learning rate')
    parser.add_argument('--g_learning_rate', action='store', type=float, default=2e-4,
                        help='g learning rate')
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

    parser.add_argument('--discriminator_optim', action='store', type=str,
                        help='discriminator_optim')
    parser.add_argument('--generator_optim', action='store', type=str,
                        help='generator_optim')
    parser.add_argument('--loss_type', action='store', type=str,
                        help='loss_type')

    return parser

class Discriminator(nn.Module):
    def __init__(self, data_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(data_dim, hidden_size),
            nn.ELU(alpha=0.05)
            #nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.05)
            #nn.ReLU()
        )
        self.l3 = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y,_ = torch.sort(x, dim=1, descending=False, out=None)
        out = self.l1(y)
        feat = self.l2(out)
        out = self.l3(feat)
        out = self.sigmoid(out)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim: int, hidden_size: int, data_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.ELU(alpha=0.2)
            #nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(alpha=0.2)
            #nn.ReLU()
        )
        #self.l3 = nn.Linear(hidden_size, data_dim)
        self.l3 = nn.Sequential( 
                  nn.Linear(hidden_size, data_dim),
                  #nn.ReLU()
                  )

    def forward(self, x: Tensor) -> Tensor:
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)

        return out

def log(x: Tensor) -> Tensor:
    """custom log function to prevent log of zero(infinity/NaN) problem."""
    return torch.log(torch.max(x, torch.tensor(1e-6).to(x.device)))

def standard_d_loss(real_score: Tensor, fake_score: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
	# -log[d(x)]
	real_part = 0.5 * torch.mean(-log(real_score))

	# -log[1 - d(g(z))]
	fake_part = 0.5 * torch.mean(-log(1.0 - fake_score))

	loss = real_part + fake_part

	return loss, real_part, fake_part


def standard_g_loss(fake_score: Tensor) -> Tensor:
	return 1.0 * torch.mean(log(1 - fake_score))


def heuristic_g_loss(fake_score: Tensor) -> Tensor:
	return 1.0 * torch.mean(-log(fake_score))



if __name__ == '__main__':

    print('start...')
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #####################################
    parser = get_parser()
    parse_args = parser.parse_args()
    epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    datafile = parse_args.datafile
    D_restore_ckpt_path = parse_args.D_restore_ckpt_path
    G_restore_ckpt_path = parse_args.G_restore_ckpt_path
    G_ckpt_path = parse_args.G_ckpt_path
    D_ckpt_path = parse_args.D_ckpt_path
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
    #N_hidden        = parse_args.N_hidden
    d_learning_rate = parse_args.d_learning_rate
    g_learning_rate = parse_args.g_learning_rate
    cost_mode       = parse_args.cost_mode
    Scale           = parse_args.Scale
    max_training_evt          = parse_args.max_training_evt
    max_valid_evt          = parse_args.max_valid_evt
    reLoadIndex  = parse_args.reLoadIndex
    IndexFile  = parse_args.IndexFile
    MinD       = parse_args.MinD
    discriminator_optim  = parse_args.discriminator_optim
    generator_optim      = parse_args.generator_optim
    loss_type            = parse_args.loss_type
    g_hidden_size        = parse_args.g_hidden_size
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
    data_dim: int = batch_size
    d_hidden_size: int = 500
    #g_hidden_size: int = 15
    g_data_dim: int = 1
    z_dim: int = 3
    D = Discriminator(data_dim, d_hidden_size).to(device)
    G = Generator(z_dim, g_hidden_size, g_data_dim).to(device)
    if discriminator_optim == 'sgd':
        d_optimizer = torch.optim.SGD(D.parameters(), lr=d_learning_rate, momentum=0.65)
    else:
        d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))

    if generator_optim == 'sgd':
        g_optimizer = torch.optim.SGD(G.parameters(), lr=g_learning_rate, momentum=0.65)
    else:
        g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))
    print('commencing training')
    logger.info('')
    if Restore:
        print('restored from ',D_restore_ckpt_path)
        D.load_state_dict(torch.load(D_restore_ckpt_path))
        print('restored from ',G_restore_ckpt_path)
        G.load_state_dict(torch.load(G_restore_ckpt_path))
    if os.path.exists(outFile): os.remove(outFile) 
    if doTraining:
        k = batch_size
        for epoch in range(epochs):
            ##################
            print('preparing data list for training')
            logger.info('')
            df = load_data(datafile, max_training_evt)
            df = df.astype('float32')
            xb = np.ascontiguousarray(df[:,0:2])
            index = faiss.IndexFlatL2(xb.shape[1]) # build the index
            print(index.is_trained)
            index.verbose = True 
            index.add(xb)                  # add vectors to the index
            #################
            total_cost = None
            count = 0
            used = []
            g_loss_list      = []
            d_real_loss_list = []
            d_fake_loss_list = []
            g_score_list      = []
            d_real_score_list = []
            d_fake_score_list = []
            for i in range(df.shape[0]):
                if(i%10000==0): 
                    print('processed %f %%'%(100*float(i)/df.shape[0]))
                    logger.info('')
                if i in used:continue
                dis, I = index.search(df[i:i+1,0:2], k)     # actual search
                I = np.squeeze(I)
                used.extend(I)
                value = df[I,0:2]
                value_t = torch.Tensor(value)
                real_data = df[I,2:3]
                real_data = np.reshape(real_data, (1,real_data.shape[0]))
                real_data_t = torch.Tensor(real_data)
                noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
                train_x = np.concatenate ((value, noise), axis=-1)
                train_x_t = torch.Tensor(train_x)
                train_x_t, value_t, real_data_t = train_x_t.to(device), value_t.to(device), real_data_t.to(device)
                # Training discriminator
                real_score = D(real_data_t)
                d_real_score_list.append(real_score.tolist()[0][0])
                d_fake_data = G(train_x_t).detach()     # detach to avoid training G on these data.
                d_fake_data = torch.reshape(d_fake_data,(1,list(d_fake_data.size())[0]))
                d_fake_score = D(d_fake_data)
                d_fake_score_list.append(d_fake_score.tolist()[0][0])
                d_loss, real_loss, fake_loss = standard_d_loss(real_score, d_fake_score)
                # zero the parameter gradients
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                # Training generator
                for g in range(1):
                    g_z_noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
                    train_x = np.concatenate ((value, g_z_noise), axis=-1)
                    #train_x = np.concatenate ((value, noise), axis=-1)
                    train_x_t = torch.Tensor(train_x)
                    train_x_t = train_x_t.to(device)
                    fake_data = G(train_x_t)
                    fake_data = torch.reshape(fake_data,(1,list(fake_data.size())[0]))
                    fake_score = D(fake_data)    # this is D(G(z)+pt+theta)
                    g_score_list.append(fake_score.tolist()[0][0])
                    g_loss = standard_g_loss(fake_score)
                    if loss_type == 'hack':
                        g_loss = heuristic_g_loss(fake_score)
                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()
                    g_loss_list.append(g_loss.tolist())
                d_real_loss_list.append(real_loss.tolist())
                d_fake_loss_list.append(fake_loss.tolist())
            
            if epoch % 1 == 0 and saveCkpt:
                print('Epoch {0} '.format(epoch))
                #print('g_score_list=',g_score_list)
                print('g_loss_list= ',sum(g_loss_list)/len(g_loss_list),',score=',sum(g_score_list)/len(g_score_list))
                print('d_real_loss_list= ',2*sum(d_real_loss_list)/len(d_real_loss_list),',score=',sum(d_real_score_list)/len(d_real_score_list))
                print('d_fake_loss_list= ',2*sum(d_fake_loss_list)/len(d_fake_loss_list),',score=',sum(d_fake_score_list)/len(d_fake_score_list))
                torch.save(G.state_dict(), G_ckpt_path)
                torch.save(D.state_dict(), D_ckpt_path)
            if doTest and epoch >= 0:
                df1 = load_data(test_file, produceEvent)
                value = df1[:, 0:2]
                noise = np.random.uniform(-1, 1, (value.shape[0], 1)) if use_uniform else np.random.normal(0, 1, (value.shape[0], 1))
                inputs = np.concatenate ((value, noise), axis=-1)
                train_x_t = torch.Tensor(inputs)
                train_x_t = train_x_t.to(device)
                outputs = G(train_x_t)
                y_pred = outputs.cpu()
                y_pred = y_pred.detach().numpy()
                y_pred = np.reshape(y_pred, (value.shape[0], 1))
                final_out = np.concatenate ((df1[:, 0:3], y_pred), axis=-1)

                hf = h5py.File(outFile, 'a')
                hf.create_dataset('Pred_epoch%d'%(epoch), data=final_out)
                hf.close()
                print('Saved produced data in %s'%outFile)
    print('done')
