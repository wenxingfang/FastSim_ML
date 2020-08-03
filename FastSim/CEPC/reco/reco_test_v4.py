import yaml
import h5py
import json
import argparse
import numpy as np
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
from sklearn.utils import shuffle
import tensorflow as tf
#############
# add HCAL  #
#############
def get_parser():
    parser = argparse.ArgumentParser(
        description='Run CalGAN training. '
        'Sensible defaults come from [arXiv/1511.06434]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch-size', action='store', type=int, default=0,
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

    return parser

if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()
    print('model=',parse_args.model_in)
    model = load_model(parse_args.model_in, custom_objects={'tf': tf})
    d = h5py.File(parse_args.datafile, 'r')
    first    = np.expand_dims(d['Barrel_Hit'     ][:],-1)
    second   = np.expand_dims(d['Barrel_Hit_HCAL'][:],-1)
    mc_info = d['MC_info'][:] 
    mc_info_v1 = np.copy(mc_info)
    d.close()
    ###### do normalization ##############
    mc_info[:,0] = (mc_info[:,0]-11)/9.0
    mc_info[:,1] = (mc_info[:,1]-90)/50
    mc_info[:,2] = (mc_info[:,2])/20
    mc_info[:,3] = (mc_info[:,3])/10
    mc_info[:,4] = (mc_info[:,4])/10
    mc_info[:,5] = (mc_info[:,5])/2000
    mc_info[:,6] = (mc_info[:,6])/600
    first, second, mc_info, mc_info_v1 = shuffle(first, second, mc_info, mc_info_v1, random_state=0)
    if parse_args.batch_size == 0 :
        parse_args.batch_size = first.shape[0]
    nBatch = int(first.shape[0]/parse_args.batch_size)
    iBatch = np.random.randint(nBatch, size=1)
    iBatch = iBatch[0] 
    input1 = first  [iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size]
    input2 = second [iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size]
    input3 = mc_info[iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size, 3:7]

    result = model.predict([input1, input2, input3], verbose=True)
    real = mc_info[iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size, 0:3]
    print('choose batch:', iBatch)
    print('pred:\n',result)
    print('real:\n',real)
    print('diff:\n',result - real)
    ######### transfer to actual value #######
    real[:,0]   = real[:,0]*9 + 11 
    real[:,1]   = real[:,1]*50 + 90
    real[:,2]   = real[:,2]*20
#    real[:,3]   = real[:,3]*10
#    real[:,4]   = real[:,4]*10
#    real[:,5]   = real[:,5]*2000
    result[:,0] = result[:,0]*9 + 11
    result[:,1] = result[:,1]*50 + 90
    result[:,2] = result[:,2]*20
#    result[:,3] = result[:,3]*10
#    result[:,4] = result[:,4]*10
#    result[:,5] = result[:,5]*2000
    abs_diff = np.abs(result - real)
    print('abs error:\n', abs_diff)
    print('mean abs error:\n',np.mean(abs_diff, axis=0))
    print('std  abs error:\n',np.std (abs_diff, axis=0))
    ###### save ##########
    hf = h5py.File(parse_args.output, 'w')
    hf.create_dataset('input_info', data=real)
    hf.create_dataset('reco_info' , data=result)
    hf.create_dataset('mc_info'   , data=mc_info_v1)
    hf.close()
    print ('Done')

