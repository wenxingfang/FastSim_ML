import yaml
import ast
import h5py
import json
import argparse
import numpy as np
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
from sklearn.utils import shuffle
import tensorflow as tf
##########################################################################
#using Mom dtheta, Mom dphi, Pos dz, Pos dphi, array with 121 cell energy#
##########################################################################
def get_parser():
    parser = argparse.ArgumentParser(
        description='Run CalGAN training. '
        'Sensible defaults come from [arXiv/1511.06434]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch-size', action='store', type=int, default=2,
                        help='batch size per update')

    parser.add_argument('--disc-lr', action='store', type=float, default=2e-5,
                        help='Adam learning rate for discriminator')

    parser.add_argument('--gen-lr', action='store', type=float, default=2e-4,
                        help='Adam learning rate for generator')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    parser.add_argument('--prog-bar', action='store_true',
                        help='Whether or not to use a progress bar')

    parser.add_argument('--no-attn', action='store_true',
                        help='Whether to turn off the layer to layer attn.')

    parser.add_argument('--debug', action='store_true',
                        help='Whether to run debug level logging')

    parser.add_argument('--model-in', action='store',type=str,
                        default='',
                        help='input of trained reg model')
    parser.add_argument('--weight-in', action='store',type=str,
                        default='',
                        help='input of trained reg weight')

    parser.add_argument('--datafile', action='store', type=str,
                        help='yaml file with particles and HDF5 paths (see '
                        'github.com/hep-lbdl/CaloGAN/blob/master/models/'
                        'particles.yaml)')
    parser.add_argument('--output', action='store',type=str,
                        default='',
                        help='output of result real vs reco')
    parser.add_argument('--removeZ', action='store',type=ast.literal_eval,
                        default=False,
                        help='remove Z for reg')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    parse_args = parser.parse_args()
    removeZ = parse_args.removeZ
    model = load_model(parse_args.model_in, custom_objects={'tf': tf})
    d = h5py.File(parse_args.datafile, 'r')
    first   = d['Barrel_Hit'][:]
    mc_info = d['MC_info'][:] 
    d.close()
    ###### do normalization ##############
    mc_info[:,2] = (mc_info[:,2])/10
    mc_info[:,5] = (mc_info[:,5])/100
    first, mc_info = shuffle(first, mc_info, random_state=0)
    nBatch = int(first.shape[0]/parse_args.batch_size)
    iBatch = np.random.randint(nBatch, size=1)
    iBatch = iBatch[0] 
    inputs1 = first [iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size]

    result = model.predict(inputs1, verbose=True)
    real = mc_info[iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size]
    if removeZ:
        result = np.insert(result, 5, values=np.array([0]), axis=1)
    print('result shape:', result.shape)
    print('choose batch:', iBatch)
    print('pred:\n',result)
    print('real:\n',real)
    print('diff:\n',result - real)
    ######### transfer to actual value #######
    real[:,2]   = real[:,2]*10
    real[:,5]   = real[:,5]*100
    result[:,2] = result[:,2]*10
    result[:,5] = result[:,5]*100
    abs_diff = np.abs(result - real)
    print('abs error:\n', abs_diff)
    print('mean abs error:\n',np.mean(abs_diff, axis=0))
    print('std  abs error:\n',np.std (abs_diff, axis=0))
    ###### save ##########
    hf = h5py.File(parse_args.output, 'w')
    hf.create_dataset('input_info', data=real)
    hf.create_dataset('reco_info' , data=result)
    hf.close()
    print ('Done')

