import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global __weights_dict
        __weights_dict = load_weights(weight_file)

        self.dense_MatMul = self.__dense(name = 'dense/MatMul', in_features = 3, out_features = 249, bias = True)
        self.dense_1_MatMul = self.__dense(name = 'dense_1/MatMul', in_features = 249, out_features = 249, bias = True)
        self.dense_2_MatMul = self.__dense(name = 'dense_2/MatMul', in_features = 249, out_features = 249, bias = True)
        self.dense_3_MatMul = self.__dense(name = 'dense_3/MatMul', in_features = 249, out_features = 249, bias = True)
        self.Pred_MatMul = self.__dense(name = 'Pred/MatMul', in_features = 249, out_features = 1, bias = True)

    def forward(self, x):
        dense_MatMul    = self.dense_MatMul(x)
        #dense_Tanh      = F.tanh(dense_MatMul)
        dense_Tanh      = torch.tanh(dense_MatMul)
        dense_1_MatMul  = self.dense_1_MatMul(dense_Tanh)
        #dense_1_Tanh    = F.tanh(dense_1_MatMul)
        dense_1_Tanh    = torch.tanh(dense_1_MatMul)
        dense_2_MatMul  = self.dense_2_MatMul(dense_1_Tanh)
        #dense_2_Tanh    = F.tanh(dense_2_MatMul)
        dense_2_Tanh    = torch.tanh(dense_2_MatMul)
        dense_3_MatMul  = self.dense_3_MatMul(dense_2_Tanh)
        #dense_3_Tanh    = F.tanh(dense_3_MatMul)
        dense_3_Tanh    = torch.tanh(dense_3_MatMul)
        Pred_MatMul     = self.Pred_MatMul(dense_3_Tanh)
        Pred_Relu       = F.relu(Pred_MatMul)
        return Pred_Relu


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
        if 'bias' in __weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

