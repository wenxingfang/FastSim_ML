#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 
import h5py
import numpy as np
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
from keras.models import Model
import math
import argparse
import sys
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/models/')
from ops import roll_fn, MyDense2D
import tensorflow as tf


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run generator. '
        'Sensible defaults come from ...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nb-events', action='store', type=int, default=10,
                        help='Number of events to be generatored.')
    parser.add_argument('--latent-size', action='store', type=int, default=512,
                        help='size of random N(0, 1) latent space to sample')
    parser.add_argument('--output', action='store', type=str,
                        help='output file.')
    parser.add_argument('--gen-model-in', action='store',type=str,
                        default='',
                        help='input of gen model')
    parser.add_argument('--gen-weight-in', action='store',type=str,
                        default='',
                        help='input of gen weight')
    parser.add_argument('--comb-model-in', action='store',type=str,
                        default='',
                        help='input of combined model')
    parser.add_argument('--comb-weight-in', action='store',type=str,
                        default='',
                        help='input of combined weight')
    parser.add_argument('--dis-model-in', action='store',type=str,
                        default='',
                        help='input of dis model')
    parser.add_argument('--exact-model', action='store',type=bool,
                        default=False,
                        help='use exact input to generate')
    parser.add_argument('--exact-list', action='store',type=str,
                        default='',
                        help='exact event list to generate')
    parser.add_argument('--datafile-temp', action='store', type=str,
                        help='template file.')

    return parser


if __name__ == '__main__':
    
    parser = get_parser()
    parse_args = parser.parse_args()
    
    gen_out = parse_args.output
    hf = h5py.File(gen_out, 'w')

    ### using generator model  ############
    #print('gen model=',parse_args.gen_model_in)
    #gen_model = model_from_yaml(open(parse_args.gen_model_in).read())
    #print('gen weight=',parse_args.gen_weight_in)
    #gen_model.load_weights(parse_args.gen_weight_in)

    gen_model = load_model(parse_args.gen_model_in, custom_objects={'tf': tf, 'math':math})

    n_gen_images = parse_args.nb_events
    #noise            = np.random.normal ( 0 , 1, (n_gen_images, parse_args.latent_size))
    noise            = np.random.uniform ( -1 , 1, (n_gen_images, parse_args.latent_size))
    generator_inputs = noise
    #############
    images = gen_model.predict(generator_inputs, verbose=True)
    #### transfer to real parameters ##############################

    #hf.create_dataset('firstHitTimeByPMT', data=images[0])
    hf.create_dataset('nPEByPMT'         , data=images)
    #hf.create_dataset('nPEByPMT'         , data=images[0])
    #hf.create_dataset('infoMuon'         , data=actual_info)
    ### using combined model to check discriminator and regression part  ############
    if parse_args.comb_model_in !='':
        comb_model = load_model(parse_args.comb_model_in, custom_objects={'tf': tf, 'math':math, 'MyDense2D':MyDense2D})
        results = comb_model.predict(generator_inputs, verbose=True)
        hf.create_dataset('Disc_fake_real' , data=results[0])
    if parse_args.dis_model_in:
        pass

    ### save results ############
    hf.close()
    print ('Saved h5 file, done')
