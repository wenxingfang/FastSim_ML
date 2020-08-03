#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 
import h5py
import numpy as np
#from keras.layers import Input, Lambda, Activation, AveragePooling2D, UpSampling2D, Dense
#from keras.layers.merge import add, concatenate, multiply
#from keras.models import Model
#from keras.layers.merge import multiply
#import keras.backend as K
#K.set_image_dim_ordering('tf')
from keras.models import model_from_json
from keras.models import model_from_yaml
from keras.models import load_model
import keras.backend as K
import tensorflow as tf
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
session = tf.Session(config=session_conf)
K.set_session(session)
import argparse
import sys
#sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/')
#from architectures import build_discriminator_3D, sparse_softmax, build_generator_3D, build_regression
#from ops import scale, inpainting_attention, calculate_energy


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
    parser.add_argument('--exact-model', action='store',type=str,
                        default='False',
                        help='use exact input to generate')
    parser.add_argument('--exact-list', action='store',type=str,
                        default='',
                        help='exact event list to generate')
    parser.add_argument('--check-dis-real', action='store',type=str,
                        default='False',
                        help='check dis for real image')
    parser.add_argument('--dis-model-in', action='store',type=str,
                        default='',
                        help='model for dis')
    parser.add_argument('--real-data', action='store',type=str,
                        default='',
                        help='real data input')

    return parser


if __name__ == '__main__':
    
    parser = get_parser()
    parse_args = parser.parse_args()
    
    gen_out = parse_args.output
    hf = h5py.File(gen_out, 'w')

    gen_model = load_model(parse_args.gen_model_in, custom_objects={'tf': tf})

    n_gen_images = parse_args.nb_events
    noise            = np.random.normal ( 0 , 1, (n_gen_images, parse_args.latent_size))
    sampled_mom      = np.random.uniform( 1   , 1.8, (n_gen_images, 1))
    sampled_dtheta   = np.random.uniform(35   , 145, (n_gen_images, 1))/180
    sampled_dphi     = np.random.uniform(-4   , 10 , (n_gen_images, 1))/10
    sampled_Z        = np.random.uniform(-130 ,130 , (n_gen_images, 1))/100
    sampled_info     = np.concatenate((sampled_mom, sampled_dtheta, sampled_dphi, sampled_Z),axis=-1)

    if parse_args.exact_model == 'True':
        f_info=open(parse_args.exact_list, 'r')
        index_line=0
        for line in f_info:
            (mom, dtheta, dphi, Z) = line.split(',')
            mom    = float(mom   .split('=')[-1])
            dtheta = float(dtheta.split('=')[-1])/180
            dphi   = float(dphi  .split('=')[-1])/10
            Z      = float(Z     .split('=')[-1])/100
            print('exact input=', mom, ":", dtheta, ':', dphi, ':', Z)
            sampled_info[index_line, 0]=mom
            sampled_info[index_line, 1]=dtheta
            sampled_info[index_line, 2]=dphi
            sampled_info[index_line, 3]=Z
            index_line = index_line + 1
            if index_line >= n_gen_images:
                print('Error: more than nb_events to produce, ignore rest part')
                break
        f_info.close()
    generator_inputs = [noise, sampled_info]
    images = gen_model.predict(generator_inputs, verbose=True)
    #### transfer to real parameters ##############################
    actual_info      = sampled_info.copy()
    actual_info[:,1] = actual_info[:,1]*180      
    actual_info[:,2] = actual_info[:,2]*10 
    actual_info[:,3] = actual_info[:,3]*100
    #print ('actual_info\n:',actual_info[0:10])

    hf.create_dataset('Barrel_Hit', data=images[0])
    hf.create_dataset('Hit_Depth' , data=images[1])
    hf.create_dataset('MC_info'   , data=actual_info)
    ### using combined model to check discriminator and regression part  ############
    if parse_args.comb_model_in !='':
        comb_model = load_model(parse_args.comb_model_in, custom_objects={'tf': tf})
        results = comb_model.predict(generator_inputs, verbose=True)
        results[1][:,1] = results[1][:,1]*180
        results[1][:,2] = results[1][:,2]*10
        results[1][:,3] = results[1][:,3]*100
        hf.create_dataset('Disc_fake' , data=results[0])
        hf.create_dataset('Reg_fake'  , data=results[1])
    ### check discriminator for real image #########
    if parse_args.check_dis_real =='True' and parse_args.dis_model_in !='':
        dis_model = load_model(parse_args.dis_model_in, custom_objects={'tf': tf})
        d = h5py.File(parse_args.real_data, 'r')
        real_input1   = d['Barrel_Hit'][:]
        real_input2   = d['Hit_Depth'][:]/100
        mc_info = d['MC_info'][:]
        ###### do normalization ##############
        mc_info[:,1] = (mc_info[:,1])/180
        mc_info[:,2] = (mc_info[:,2])/10
        mc_info[:,3] = (mc_info[:,2])/100
        dis_input = [real_input1, real_input2, mc_info]
        d.close()
        dis_result = dis_model.predict(dis_input, verbose=True)
        hf.create_dataset('Disc_real' , data=dis_result)
    ### save results ############
    hf.close()
    print ('Saved h5 file, done')
