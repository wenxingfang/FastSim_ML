#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 

########################################
## For PMT in theta(180)*phi(360) grid##
## Add regression and discriminator 
########################################
import h5py
import numpy as np
from keras.layers import Input, Lambda, Activation, AveragePooling2D, UpSampling2D, Dense
from keras.models import Model
from keras.layers.merge import multiply, concatenate
import keras.backend as K

import argparse
import sys
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/models/')
from architectures import build_generator, build_discriminator, sparse_softmax, build_generator_v1, build_generator_v2
from ops import scale, inpainting_attention, calculate_energy


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
    parser.add_argument('--weights', action='store', type=str,
                        help='weigths file to be loaded.')
    parser.add_argument('--dis-weights', action='store', type=str,
                        help='dis weigths file to be loaded.')
    parser.add_argument('--output', action='store', type=str,
                        help='output file.')
    parser.add_argument('--datafile_temp', action='store', type=str,
                        help='HDF5 template file paths')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()

    #Let's build the network architecture we proposed, load its trained weights, and use it to generate synthesized showers. We do all this using Keras with the TensorFlow backend
    #This was our choice for the size of the latent space  í µí±§ . If you want to retrain this net, you can try changing this parameter.
    latent_size = parse_args.latent_size
    latent = Input(shape=(latent_size, ), name='z') 
    layer0_T0 = Input(shape=(180, 360,1), name='l0_T0')
    layer0_T1 = Input(shape=(180, 360,1), name='l0_T1')
    layer1_T0 = Input(shape=(180, 360,1), name='l1_T0')
    layer1_T1 = Input(shape=(180, 360,1), name='l1_T1')
    input_info = Input(shape=(5, ), dtype='float32')
    generator_inputs = [latent, layer0_T0, layer0_T1, layer1_T0, layer1_T1, input_info]
    h = concatenate([latent,input_info])
    img_layer0 = build_generator_v2(h, 180, 360, layer0_T0, layer0_T1)
    img_layer1 = build_generator_v2(h, 180, 360, layer1_T0, layer1_T1)
    generator_outputs = [
        img_layer0,
        img_layer1
    ]
    
    generator = Model(generator_inputs, generator_outputs)
    #print(generator.summary())
    
    # load trained weights
    #generator.load_weights('/hpcfs/juno/junogpu/fangwx/FastSim/params_generator_epoch_049.hdf5')
    weigths = parse_args.weights
    generator.load_weights(weigths)
    
    n_gen_images = parse_args.nb_events
    
    noise = np.random.normal(0, 1, (n_gen_images, latent_size))
    sampled_ptheta   = np.random.uniform(0   , 1, (n_gen_images, 1))
    sampled_pphi     = np.random.uniform(0   , 1, (n_gen_images, 1))
    sampled_rtheta   = np.random.uniform(0   , 1, (n_gen_images, 1))
    sampled_rphi     = np.random.uniform(0   , 1, (n_gen_images, 1))
    sampled_r        = np.random.uniform(0.99, 1, (n_gen_images, 1))
    sampled_info     = np.concatenate((sampled_ptheta, sampled_pphi, sampled_rtheta, sampled_rphi, sampled_r ),axis=-1)
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

    images = generator.predict([noise]+Temps_input+[sampled_info], verbose=True)
    ###############
    

    print('H1:',images[0].shape)
    print('H1:',images[1].shape)
    ### for discriminator check############
    calorimeter = [Input(shape=(180,360,1)),
                   Input(shape=(180,360,1))]
    input_energy = Input(shape=(1, ))
    features = []
    print('H2')
    for l in range(2):
        # build features per layer of calorimeter
        features.append(build_discriminator(
            image=calorimeter[l],
            mbd=False,
            sparsity=False,
            sparsity_mbd=False
        ))
        
    print('H3')
    energies = calculate_energy(calorimeter[1])# get the total nPE for each event!
    energies = scale(energies,10000)
    print('Hi')
    features = concatenate(features)
    print('hi')
    p = concatenate([features,energies])
    print('hii')
    fake = Dense(1, activation='sigmoid', name='fakereal_output')(features)
    reg  = Dense(5,                       name='muon_info_output')(p)
    discriminator_outputs = [fake, reg]
    discriminator = Model(calorimeter, discriminator_outputs)
    dis_weigths = parse_args.dis_weights
    discriminator.load_weights(dis_weigths)
    results = discriminator.predict(images, verbose=True)

    ########### save ################

    gen_out = parse_args.output
    hf = h5py.File(gen_out, 'w')
    hf.create_dataset('firstHitTimeByPMT', data=images[0])
    hf.create_dataset('nPEByPMT'         , data=images[1])
    hf.create_dataset('fake_real'        , data=results[0])
    hf.create_dataset('reg_para'         , data=results[1])
    hf.close()
    print ('Done')
    
    
    
