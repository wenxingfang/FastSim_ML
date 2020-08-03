import math
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

    return parser

if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()
    model = load_model(parse_args.model_in, custom_objects={'tf': tf})

    d = h5py.File(parse_args.datafile, 'r')
    first = np.expand_dims(d['firstHitTimeByPMT'][:], -1)
    second = np.expand_dims(d['nPEByPMT'][:], -1)
    infoMuon = d['infoMuon'][:,:4]
    d.close()
    print('infoMuon dtype',infoMuon.dtype)
    infoMuon = infoMuon.astype(float)
    print('infoMuon dtype',infoMuon.dtype)
    ###  normalize muon info ##############
    infoMuon[:,0]=(infoMuon[:,0])/math.pi
    infoMuon[:,1]=(infoMuon[:,1])/math.pi
    infoMuon[:,2]=(infoMuon[:,2])/math.pi
    infoMuon[:,3]=(infoMuon[:,3])/math.pi
    #infoMuon[:,4]=(infoMuon[:,4])/18000#17700.0

    first, second, infoMuon = shuffle(first, second, infoMuon, random_state=0)

    nBatch = int(first.shape[0]/parse_args.batch_size)
    iBatch = np.random.randint(nBatch, size=1)
    iBatch = iBatch[0] 
    input_first  = first [iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size]
    input_second = second[iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size]

    #result = model.predict([input_first, input_second], batch_size=parse_args.batch_size, verbose=True)
    result = model.predict([input_first, input_second], batch_size=128, verbose=True)
    real = infoMuon[iBatch*parse_args.batch_size:(iBatch+1)*parse_args.batch_size]
    print('choose batch:', iBatch)
    print('pred:\n',result)
    print('real:\n',real)
    print('diff:\n',result - real)
    ######### transfer to actual value #######
    real[:,0]   = real[:,0]*math.pi
    real[:,1]   = real[:,1]*math.pi
    real[:,2]   = real[:,2]*math.pi
    real[:,3]   = real[:,3]*math.pi
    #real[:,4]   = real[:,4]*18000
    result[:,0] = result[:,0]*math.pi
    result[:,1] = result[:,1]*math.pi
    result[:,2] = result[:,2]*math.pi
    result[:,3] = result[:,3]*math.pi
    #result[:,4] = result[:,4]*18000
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

