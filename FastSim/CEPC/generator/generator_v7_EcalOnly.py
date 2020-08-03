#!/hpcfs/juno/junogpu/fangwx/python/Python-3.6.6/python 
import h5py
import ast
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
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/BES/models/')
from convert import freeze_session
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/CEPC/models/')
#from architectures import build_discriminator_3D, sparse_softmax, build_generator_3D, build_regression
from ops import MyDense2D 
##############################
## use cell ID for ECAL and one input #
## only reg mom, dtheta, dphi
## add HCAL
##############################
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
    parser.add_argument('--exact-model', action='store',type=ast.literal_eval,
                        default=False,
                        help='use exact input to generate')
    parser.add_argument('--exact-list', action='store',type=str,
                        default='',
                        help='exact event list to generate')
    parser.add_argument('--check-dis-real', action='store',type=ast.literal_eval,
                        default=False,
                        help='check dis for real image')
    parser.add_argument('--dis-model-in', action='store',type=str,
                        default='',
                        help='model for dis')
    parser.add_argument('--real-data', action='store',type=str,
                        default='',
                        help='real data input')
    parser.add_argument('--for-mc-reco', action='store',type=ast.literal_eval,
                        default=False,
                        help='generate shower for reco')
    parser.add_argument('--info-mc-reco', action='store',type=str,
                        default='',
                        help='mc reco info data input')


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


def readInput(datafile):
    print('reading input')
    if '.h5' in datafile:
        d = h5py.File(datafile, 'r')
        df1 = d['Barrel_Hit'][:]
        df2 = d['Barrel_Hit_HCAL'][:]
        df3 = d['MC_info'][:]
        d.close()
        return df1, df2, df3
    
    df1 = None
    df2 = None
    df3 = None
    First = True
    f_DataSet = open(datafile, 'r')
    for line in f_DataSet: 
        idata = line.strip('\n')
        idata = idata.strip(' ')
        if "#" in idata: continue ##skip the commented one
        d = h5py.File(str(idata), 'r')
        if First:
            df1 = d['Barrel_Hit'][:]
            df2 = d['Barrel_Hit_HCAL'][:]
            df3 = d['MC_info'][:]
            First = False
        else:
            df1 = np.concatenate ((df1, d['Barrel_Hit'][:])     , axis=0)
            df2 = np.concatenate ((df2, d['Barrel_Hit_HCAL'][:]), axis=0)
            df3 = np.concatenate ((df3, d['MC_info'][:])        , axis=0)
        d.close()
    f_DataSet.close() 
    print('df1=',df1.shape)
    return df1, df2, df3



if __name__ == '__main__':
    
    parser = get_parser()
    parse_args = parser.parse_args()
    

    convert2pb = parse_args.convert2pb
    path_pb    = parse_args.path_pb
    name_pb    = parse_args.name_pb
    SavedModel = parse_args.SavedModel
    export_path= parse_args.export_path


    gen_out = parse_args.output
    hf = h5py.File(gen_out, 'w')

    ### using generator model  ############
    #print('gen model=',parse_args.gen_model_in)
    #gen_model = model_from_yaml(open(parse_args.gen_model_in).read())
    #print('gen weight=',parse_args.gen_weight_in)
    #gen_model.load_weights(parse_args.gen_weight_in)

    gen_model = load_model(parse_args.gen_model_in, custom_objects={'tf': tf})


    if convert2pb:
        Model = gen_model
        print('convert to pb now')
        print('input is :', Model.input.name)
        print ('output 0 is:', Model.output[0].name)
        print ('output 1 is:', Model.output[1].name)
        print ('output 0 op is:', Model.output[0].op.name)
        print ('output 1 op is:', Model.output[1].op.name)
        frozen_graph = freeze_session(K.get_session(), output_names=[Model.output[0].op.name, Model.output[1].op.name])
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



    '''
    n_gen_images = 0
    noise = 0
    mc_info = 0
    if parse_args.for_mc_reco:
        d = h5py.File(parse_args.info_mc_reco, 'r')
        n_gen_images = d['MC_info'].shape[0]
        noise   = np.random.normal ( 0 , 1, (n_gen_images, parse_args.latent_size))
        mc_info = d['MC_info'][0:n_gen_images]
    else:
        print('not for mc re reco')
        n_gen_images = parse_args.nb_events
        noise   = np.random.normal ( 0 , 1, (n_gen_images, parse_args.latent_size))
        d = h5py.File(parse_args.real_data, 'r')
        mc_info = d['MC_info'][0:n_gen_images]
        d.close()
    '''
    E_Hit, H_Hit, mc_info = readInput(parse_args.real_data)
    n_gen_images = mc_info.shape[0]
    noise   = np.random.normal ( 0 , 1, (n_gen_images, parse_args.latent_size))
    ###### do normalization ##############
    mc_info[:,0] = (mc_info[:,0]-11)/9.0
    mc_info[:,1] = (mc_info[:,1]-90)/50
    mc_info[:,2] = (mc_info[:,2])/20
    mc_info[:,3] = (mc_info[:,3])/10
    mc_info[:,4] = (mc_info[:,4])/10
    mc_info[:,5] = (mc_info[:,5])/2000
    mc_info[:,6] = (mc_info[:,6])/600
    sampled_info=np.copy(mc_info)


    if parse_args.exact_model:
        print('using exact event list')
        f_info=open(parse_args.exact_list, 'r')
        index_line=0
        for line in f_info:
            (mom, M_dtheta, M_dphi, P_dz, P_dy, Z, Y) = line.split(',')
            mom      = (float(mom.split('=')[-1])-11)/9.0
            M_dtheta = (float(M_dtheta.split('=')[-1])-90)/50
            M_dphi   = float(M_dphi  .split('=')[-1])/20
            P_dz     = float(P_dz  .split('=')[-1])/10
            P_dy     = float(P_dy  .split('=')[-1])/10
            Z        = float(Z     .split('=')[-1])/2000
            Y        = float(Y     .split('=')[-1])/600
            #print('exact input=', theta, ':', phi, ':', Energy)
            sampled_info[index_line, 0]=mom
            sampled_info[index_line, 1]=M_dtheta
            sampled_info[index_line, 2]=M_dphi
            sampled_info[index_line, 3]=P_dz
            sampled_info[index_line, 4]=P_dy
            sampled_info[index_line, 5]=Z
            sampled_info[index_line, 6]=Y
            index_line = index_line + 1
            if index_line >= n_gen_images:
                print('Error: more than nb_events to produce, ignore rest part')
                break
        f_info.close()
    #generator_inputs = [noise, sampled_info]
    print('noise shape=',noise.shape,", sampled_info shape=",sampled_info.shape)
    #generator_inputs = np.concatenate ((noise, sampled_info), axis=-1)
    generator_inputs = noise
    images = gen_model.predict(generator_inputs, verbose=True)
    #### transfer to real parameters ##############################
    actual_info      = sampled_info.copy()
    actual_info[:,0] = actual_info[:,0]*9  + 11    
    actual_info[:,1] = actual_info[:,1]*50 + 90      
    actual_info[:,2] = actual_info[:,2]*20
    actual_info[:,3] = actual_info[:,3]*10
    actual_info[:,4] = actual_info[:,4]*10
    actual_info[:,5] = actual_info[:,5]*2000
    actual_info[:,6] = actual_info[:,6]*600
    #print ('actual_info\n:',actual_info[0:10])

    hf.create_dataset('Barrel_Hit', data=images)
    #hf.create_dataset('Barrel_Hit', data=images[0])
    #hf.create_dataset('Barrel_Hit_HCAL', data=images[1])
    hf.create_dataset('MC_info'   , data=actual_info)
    ### using combined model to check discriminator and regression part  ############
    if parse_args.comb_model_in !='':
        comb_model = load_model(parse_args.comb_model_in, custom_objects={'tf': tf,'MyDense2D':MyDense2D(output_dim1=10, output_dim2=10) })
        results = comb_model.predict(generator_inputs, verbose=True)
        #results[1][:,0] = results[1][:,0]*9  + 11
        #results[1][:,1] = results[1][:,1]*50 + 90
        #results[1][:,2] = results[1][:,2]*20
        hf.create_dataset('Disc_fake' , data=results[0])
        #hf.create_dataset('Reg_fake'  , data=results[1])
    ### check discriminator for real image #########
    if parse_args.check_dis_real and parse_args.dis_model_in !='':
        dis_model = load_model(parse_args.dis_model_in, custom_objects={'tf': tf, 'MyDense2D':MyDense2D(output_dim1=10, output_dim2=10)})
        real_input1   = np.expand_dims(E_Hit,-1)
        real_input2   = np.expand_dims(H_Hit,-1)
        #dis_input = [real_input1, real_input2, mc_info]
        dis_input = [real_input1]
        dis_result = dis_model.predict(dis_input, verbose=True)
        hf.create_dataset('Disc_real' , data=dis_result)
        #d = h5py.File(parse_args.real_data, 'r')
        #real_input1   = np.expand_dims(d['Barrel_Hit'][:],-1)
        #real_input2   = np.expand_dims(d['Barrel_Hit_HCAL'][:],-1)
        #mc_info = d['MC_info'][:]
        ###### do normalization ##############
        #mc_info[:,0] = (mc_info[:,0]-11)/9.0
        #mc_info[:,1] = (mc_info[:,1]-90)/50
        #mc_info[:,2] = (mc_info[:,2])/20
        #mc_info[:,3] = (mc_info[:,3])/10
        #mc_info[:,4] = (mc_info[:,4])/10
        #mc_info[:,5] = (mc_info[:,5])/2000
        #mc_info[:,6] = (mc_info[:,6])/600
        #d.close()
    ### save results ############
    hf.close()
    print ('Saved h5 file, done')
