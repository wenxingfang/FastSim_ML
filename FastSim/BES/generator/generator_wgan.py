#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 
import h5py
import numpy as np
import ast
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
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/models/')
from architectures import RandomWeightedAverage, RandomWeightedAverage1, wasserstein_loss 
from ops import MyDense2D, single_layer_energy
from convert import freeze_session
#################################################
## merge the input [laten, mc_info] to one array#
## only reg mom, dtheta, dphi
#################################################
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
    parser.add_argument('--removeZ', action='store',type=ast.literal_eval,
                        default=False,
                        help='remove Z info')
    parser.add_argument('--convert2pb', action='store',type=ast.literal_eval,
                        default=False,
                        help='convert to pb')
    parser.add_argument('--path_pb', action='store',type=str,
                        default='',
                        help='output path for pb')
    parser.add_argument('--name_pb', action='store',type=str,
                        default='',
                        help='name for pb')
    parser.add_argument('--SavedModel', action='store',type=ast.literal_eval,
                        default=False,
                        help='convert to pb')
    parser.add_argument('--export_path', action='store',type=str,
                        default='',
                        help='path for SavedModel')

    return parser


if __name__ == '__main__':
    
    parser = get_parser()
    parse_args = parser.parse_args()
    
    gen_out = parse_args.output
    removeZ = parse_args.removeZ
    convert2pb = parse_args.convert2pb
    path_pb    = parse_args.path_pb
    name_pb    = parse_args.name_pb
    SavedModel = parse_args.SavedModel
    export_path= parse_args.export_path


    hf = h5py.File(gen_out, 'w')

    gen_model = load_model(parse_args.gen_model_in, custom_objects={'tf': tf})

    if convert2pb:
        Model = gen_model
        print('convert to pb now')
        print('input is :', Model.input.name)
        print ('output 0 is:', Model.output[0].name)
        print ('output 1 is:', Model.output[1].name)
        print ('output 0 op is:', Model.output[0].op.name)
        print ('output 1 op is:', Model.output[1].op.name)
        #print('input is :', Model.input)
        #print ('output is:', Model.output)
        frozen_graph = freeze_session(K.get_session(), output_names=[Model.output[0].op.name])
        from tensorflow.python.framework import graph_io
        output_path= path_pb
        pb_model_name= name_pb #'5_trained_model.pb'
        graph_io.write_graph(frozen_graph, output_path, pb_model_name, as_text=False)
    if SavedModel:
        Model = gen_model
        tf.saved_model.simple_save(
        K.get_session(),
        export_path,
        inputs={'Gen_input': Model.input},
        outputs={t.name:t for t in Model.outputs})
        print('Saved model to %s'%export_path)


    n_gen_images = parse_args.nb_events
    noise            = np.random.normal ( 0 , 1, (n_gen_images, parse_args.latent_size))
    ''' 
    sampled_mom      = np.random.uniform( 1   , 1.8, (n_gen_images, 1))
    sampled_M_dtheta   = np.random.uniform(35   , 145, (n_gen_images, 1))/180
    sampled_M_dphi     = np.random.uniform(-4   , 10 , (n_gen_images, 1))/10
    sampled_Z        = np.random.uniform(-130 ,130 , (n_gen_images, 1))/100
    sampled_info     = np.concatenate((sampled_mom, sampled_M_dtheta, sampled_M_dphi, sampled_Z),axis=-1)
    ''' 
    
    d = h5py.File(parse_args.real_data, 'r')
    mc_info = d['MC_info'][:]
    ###### do normalization ##############
    mc_info[:,1] = (mc_info[:,1])/5    #M_dtheta
    mc_info[:,2] = (mc_info[:,2])/10   #M_dphi
    mc_info[:,3] = (mc_info[:,3])/2    #P_dz
    mc_info[:,4] = (mc_info[:,4])/2    #P_dphi
    mc_info[:,5] = (mc_info[:,5])/150  #Z
    sampled_info = mc_info[0:n_gen_images]

    if parse_args.exact_model == 'True':
        f_info=open(parse_args.exact_list, 'r')
        index_line=0
        for line in f_info:
            (mom, M_dtheta, M_dphi, P_dz, P_dphi, P_z) = line.split(',')
            mom      = float(mom     .split('=')[-1])
            M_dtheta = float(M_dtheta.split('=')[-1])/5
            M_dphi   = float(M_dphi  .split('=')[-1])/10
            P_dz     = float(P_dz    .split('=')[-1])/2
            P_dphi   = float(P_dphi  .split('=')[-1])/2
            P_z      = float(P_z     .split('=')[-1])/150
            print('exact input=', mom, ":", M_dtheta, ':', M_dphi,':',P_dz,':',P_dphi, ':', P_z)
            sampled_info[index_line, 0]=mom
            sampled_info[index_line, 1]=M_dtheta
            sampled_info[index_line, 2]=M_dphi
            sampled_info[index_line, 3]=P_dz
            sampled_info[index_line, 4]=P_dphi
            sampled_info[index_line, 5]=P_z
            index_line = index_line + 1
            if index_line >= n_gen_images:
                print('Error: more than nb_events to produce, ignore rest part')
                break
        f_info.close()
    #mc_sampled_info = sampled_info
    #if removeZ:
    #    mc_sampled_info = sampled_info[:,0:5]
    generator_inputs = np.concatenate ((noise, sampled_info), axis=-1)
    images = gen_model.predict(generator_inputs, verbose=True)
    #### transfer to real parameters ##############################
    actual_info      = sampled_info.copy()
    actual_info[:,1] = actual_info[:,1]*5    #M_dtheta
    actual_info[:,2] = actual_info[:,2]*10   #M_dphi
    actual_info[:,3] = actual_info[:,3]*2    #P_dz
    actual_info[:,4] = actual_info[:,4]*2    #P_dphi
    actual_info[:,5] = actual_info[:,5]*150  #Z
    #print ('actual_info\n:',actual_info[0:10])

    hf.create_dataset('Barrel_Hit', data=images[0])
    hf.create_dataset('MC_info'   , data=actual_info)
    ### using combined model to check discriminator and regression part  ############
    if parse_args.comb_model_in !='':
        comb_model = load_model(parse_args.comb_model_in, custom_objects={'tf': tf,'MyDense2D':MyDense2D(output_dim1=10, output_dim2=10), 'single_layer_energy':single_layer_energy,'RandomWeightedAverage':RandomWeightedAverage, 'RandomWeightedAverage1':RandomWeightedAverage1, 'wasserstein_loss':wasserstein_loss })
        results = comb_model.predict(generator_inputs, verbose=True)
        results[1][:,1] = results[1][:,1]*5    #M_dtheta
        results[1][:,2] = results[1][:,2]*10   #M_dphi
        #results[1][:,3] = results[1][:,3]*2    #P_dz
        #results[1][:,4] = results[1][:,4]*2    #P_dphi
        #results[1][:,5] = results[1][:,5]*150
        #if removeZ:
        #    results[1] = np.insert(results[1], 5, values=np.array([0]), axis=1)
        #else:
        #    results[1][:,5] = results[1][:,5]*150
        hf.create_dataset('Disc_fake' , data=results[0])
        hf.create_dataset('Reg_fake'  , data=results[1])
    ### check discriminator for real image #########
    if parse_args.check_dis_real =='True' and parse_args.dis_model_in !='':
        dis_model = load_model(parse_args.dis_model_in, custom_objects={'tf': tf,'MyDense2D':MyDense2D(output_dim1=10, output_dim2=10), 'single_layer_energy':single_layer_energy, 'RandomWeightedAverage':RandomWeightedAverage, 'RandomWeightedAverage1':RandomWeightedAverage1, 'wasserstein_loss':wasserstein_loss })
        d = h5py.File(parse_args.real_data, 'r')
        real_input1   = d['Barrel_Hit'][:]
        mc_info = d['MC_info'][:]
        ###### do normalization ##############
        mc_info[:,1] = (mc_info[:,1])/5    #M_dtheta
        mc_info[:,2] = (mc_info[:,2])/10   #M_dphi
        mc_info[:,3] = (mc_info[:,3])/2    #P_dz
        mc_info[:,4] = (mc_info[:,4])/2    #P_dphi
        mc_info[:,5] = (mc_info[:,5])/150  #Z
        dis_input = [real_input1, mc_info, images[0], images[1]]
        #if removeZ:
        #    dis_input = [real_input1, mc_info[:,0:5]]
        d.close()
        dis_result = dis_model.predict(dis_input, verbose=True)
        hf.create_dataset('Disc_real'  , data=dis_result[0])
        hf.create_dataset('Disc_fake1' , data=dis_result[1])
        hf.create_dataset('Disc_aveg'  , data=dis_result[2])
    ### save results ############
    hf.close()
    print ('Saved h5 file, done')
