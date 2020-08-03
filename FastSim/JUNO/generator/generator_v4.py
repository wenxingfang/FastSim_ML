#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 
import h5py
import numpy as np
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
import math
import argparse
import sys
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/models/')
from ops import roll_fn
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

    gen_model = load_model(parse_args.gen_model_in, custom_objects={'tf': tf})

    n_gen_images = parse_args.nb_events
    noise            = np.random.normal ( 0 , 1, (n_gen_images, parse_args.latent_size))
    sampled_ptheta   = np.random.uniform(0   , 1, (n_gen_images, 1))
    sampled_pphi     = np.random.uniform(-1  , 1, (n_gen_images, 1))
    sampled_rtheta   = np.random.uniform(0   , 1, (n_gen_images, 1))
    sampled_rphi     = np.random.uniform(-1  , 1, (n_gen_images, 1))
    #sampled_r        = np.random.uniform(0.99, 1, (n_gen_images, 1))
    sampled_info     = np.concatenate((sampled_ptheta, sampled_pphi, sampled_rtheta, sampled_rphi),axis=-1)

    if parse_args.exact_model:
        f_info=open(parse_args.exact_list, 'r')
        index_line=0
        for line in f_info:
            (ptheta, pphi, rtheta, rphi) = line.split(',')
            ptheta = float(ptheta.split('=')[-1])/math.pi
            pphi   = float(pphi  .split('=')[-1])/math.pi
            rtheta = float(rtheta.split('=')[-1])/math.pi
            rphi   = float(rphi  .split('=')[-1])/math.pi
            print('exact input=', ptheta, ':', pphi, ':', rtheta, ':', rphi)
            sampled_info[index_line, 0]=ptheta
            sampled_info[index_line, 1]=pphi
            sampled_info[index_line, 2]=rtheta
            sampled_info[index_line, 3]=rphi
            index_line = index_line + 1
            if index_line >= n_gen_images:
                print('Error: more than nb_events to produce, ignore rest part')
                break
        f_info.close()
    #############
    d_temp = h5py.File(parse_args.datafile_temp, 'r')
    temp_time_de = np.expand_dims(d_temp['temp1_firstHitTimeByPMT'][0:1], -1)## default and 0
    temp_time_01 = np.expand_dims(d_temp['temp2_firstHitTimeByPMT'][0:1], -1)## 1 and 0
    temp_nPE_de = np.expand_dims(d_temp['temp1_nPEByPMT'][0:1], -1)
    temp_nPE_01 = np.expand_dims(d_temp['temp2_nPEByPMT'][0:1], -1)
    temp_time_de_ = temp_time_de.repeat(n_gen_images,axis=0)
    temp_time_01_ = temp_time_01.repeat(n_gen_images,axis=0)
    temp_nPE_de_  = temp_nPE_de.repeat (n_gen_images,axis=0)
    temp_nPE_01_  = temp_nPE_01.repeat (n_gen_images,axis=0)
    Temps_input = [temp_time_de_, temp_time_01_, temp_nPE_de_, temp_nPE_01_]
    d_temp.close()
    generator_inputs = [noise]+Temps_input+[sampled_info]
    images = gen_model.predict(generator_inputs, verbose=True)
    #### transfer to real parameters ##############################
    actual_info      = sampled_info.copy()
    actual_info[:,0] = actual_info[:,0]*math.pi    
    actual_info[:,1] = actual_info[:,1]*math.pi 
    actual_info[:,2] = actual_info[:,2]*math.pi
    actual_info[:,3] = actual_info[:,3]*math.pi
    #print ('actual_info\n:',actual_info[0:10])

    hf.create_dataset('firstHitTimeByPMT', data=images[0])
    hf.create_dataset('nPEByPMT'         , data=images[1])
    hf.create_dataset('infoMuon'         , data=actual_info)
    ### using combined model to check discriminator and regression part  ############
    if parse_args.comb_model_in !='':
        comb_model = load_model(parse_args.comb_model_in, custom_objects={'tf': tf, 'roll_fn':roll_fn})
        results = comb_model.predict(generator_inputs, verbose=True)
        results[1][:,0] = results[1][:,0]*math.pi
        results[1][:,1] = results[1][:,1]*math.pi
        results[1][:,2] = results[1][:,2]*math.pi
        results[1][:,3] = results[1][:,3]*math.pi
        hf.create_dataset('Disc_fake_real' , data=results[0])
        hf.create_dataset('Reg'            , data=results[1])
    ### save results ############
    hf.close()
    print ('Saved h5 file, done')
