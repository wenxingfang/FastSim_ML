#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 
import h5py
import numpy as np
from keras.layers import Input, Lambda, Activation, AveragePooling2D, UpSampling2D, Dense
from keras.layers.merge import add, concatenate, multiply
from keras.models import Model
from keras.layers.merge import multiply
import keras.backend as K
K.set_image_dim_ordering('tf')

import argparse
import sys
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/')
from architectures import build_discriminator_3D, sparse_softmax, build_generator_3D, build_regression
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

    return parser


if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()

    #Let's build the network architecture we proposed, load its trained weights, and use it to generate synthesized showers. We do all this using Keras with the TensorFlow backend
    #This was our choice for the size of the latent space  í µí±§ . If you want to retrain this net, you can try changing this parameter.
    latent_size = parse_args.latent_size
    latent = Input(shape=(latent_size, ), name='z') 
    input_info = Input(shape=(3, ), dtype='float32')
    generator_inputs = [latent, input_info]
    h = concatenate([latent, input_info])
    img_layer  = build_generator_3D(h, 30 , 30 , 29)
    generator_outputs =  img_layer
    generator = Model(generator_inputs, generator_outputs)
    #print(generator.summary())
    
    # load trained weights
    #generator.load_weights('/hpcfs/juno/junogpu/fangwx/FastSim/params_generator_epoch_049.hdf5')
    weigths = parse_args.weights
    generator.load_weights(weigths)
    
    n_gen_images = parse_args.nb_events
    
    noise            = np.random.normal ( 0, 1  , (n_gen_images, latent_size))
    sampled_energies = np.random.uniform(10, 100, (n_gen_images, 1))
    sampled_theta    = np.random.uniform(45, 135, (n_gen_images, 1))
    sampled_phi      = np.random.uniform(-10, 10, (n_gen_images, 1))
    sampled_info     = np.concatenate((sampled_theta, sampled_phi, sampled_energies),axis=-1)
    generator_inputs = [noise, sampled_info]
    images = generator.predict(generator_inputs, verbose=True)

    ### discriminator part check ############
    calorimeter = Input(shape=[30, 30, 29, 1])
    features =build_discriminator_3D(image=calorimeter, mbd=True, sparsity=False, sparsity_mbd=False)
    energies = calculate_energy(calorimeter)
    p = concatenate([features, energies])
    fake = Dense(1, activation='sigmoid', name='fakereal_output')(features)
    reg = build_regression(p)
    discriminator_outputs = [fake, reg]
    discriminator = Model(calorimeter, discriminator_outputs)
    dis_weigths = parse_args.dis_weights
    discriminator.load_weights(dis_weigths)
    print ('start predict dis')
    dis_result = discriminator.predict(images, verbose=True)
    
    ### save results ############
    print ('start to save')
    print('images shape:', images.shape)
    print('sampled_info shape:', sampled_info.shape)
    #print('dis_result shape:', dis_result.shape)
    print('dis_result len:', len(dis_result))
    print('dis_result 0 :', dis_result[0])
    print('dis_result 1 :', dis_result[1])
    gen_out = parse_args.output
    hf = h5py.File(gen_out, 'w')
    hf.create_dataset('Barrel_Hit', data=images)
    hf.create_dataset('MC_info'   , data=sampled_info)
    hf.create_dataset('Disc_fake_real' , data=dis_result[0])
    hf.create_dataset('Reg'            , data=dis_result[1])
    hf.close()
    print ('Saved h5 file, done')
    
    
