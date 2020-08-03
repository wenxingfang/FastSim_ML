import sys
sys.path.append('/hpcfs/juno/junogpu/fangwx/FastSim/reco/')
from junonn_inference_vgg16 import inference
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_data', '', 'input data')
tf.app.flags.DEFINE_string('load_weight', '', 'load weight')
tf.app.flags.DEFINE_integer('batch_size', 1 , 'batch size')

if __name__ == '__main__':
    ### do some test #####
    
    import h5py
    d = h5py.File(FLAGS.input_data,'r')
    first = np.expand_dims(d['firstHitTimeByPMT'][:], -1)
    second = np.expand_dims(d['nPEByPMT'][:], -1)
    muon   = np.expand_dims(d['infoMuon'][:], -1)
    FLAGS.batch_size = 1  
    #for index in range(0, 3):
    for index in range(200, 210):
        tf.reset_default_graph()
        print('index:', index)
        image_batch_1 = first [index * FLAGS.batch_size:(index + 1) * FLAGS.batch_size]
        image_batch_2 = second[index * FLAGS.batch_size:(index + 1) * FLAGS.batch_size]
        meanq = np.mean(image_batch_2)
        meant = np.mean(image_batch_1)
        diffq=image_batch_2-meanq
        difft=image_batch_1-meant
        stdq=np.sqrt(np.mean(np.square(diffq)))
        stdt=np.sqrt(np.mean(np.square(difft)))
        resultimageq=diffq/stdq
        resultimaget=difft/stdt
        image = np.concatenate((resultimageq, resultimaget),axis=-1) # nPE should go first , then is time

        muon_batch    = muon  [index * FLAGS.batch_size:(index + 1) * FLAGS.batch_size]
        #print (image.shape)
        image = tf.convert_to_tensor(image)
        if image.dtype == tf.float64:
            image = tf.cast(image, tf.float32)
        reco = inference(image)
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_op)           
            saver.restore(sess, FLAGS.load_weight)
            result = sess.run(reco)        
            print ('x,y,z,px,py,pz:\n',result)      
            #print ('muon_batch shape',muon_batch.shape)
            thep_true = muon_batch[0][0][0]      
            phip_true = muon_batch[0][1][0]      
            thes_true = muon_batch[0][2][0]      
            phis_true = muon_batch[0][3][0]      
            r_r       = muon_batch[0][4][0]      
            E         = muon_batch[0][5][0]      
            real    = [np.sin(thes_true)*np.cos(phis_true),np.sin(thes_true)*np.sin(phis_true),np.cos(thes_true),np.sin(thep_true)*np.cos(phip_true),np.sin(thep_true)*np.sin(phip_true),np.cos(thep_true)]
            print ('real:',real)
            #result = np.expand_dims(result, -1)
            #muon_r = np.sqrt( result[:,0,:]* result[:,0,:] + result[:,1,:]* result[:,1,:] + result[:,2,:]* result[:,2,:] )
            #muon_p = np.sqrt( result[:,3,:]* result[:,3,:] + result[:,4,:]* result[:,4,:] + result[:,5,:]* result[:,5,:] )
            #print ('muon_r :',muon_r)
            #muon_theta = np.arccos(result[:,2]/muon_r)
            #print ('muon_theta:',muon_theta)
            #muon_phi = np.arctan(result[:,1]/result[:,0])
            #muon_phi[muon_phi<0] = muon_phi[muon_phi<0]+2*3.14159
            #print ('muon_phi:',muon_phi)
            #muon_ptheta = np.arccos(result[:,5]/muon_p)
            #print ('muon_ptheta:',muon_ptheta)
            #muon_pphi = np.arctan(result[:,4]/result[:,3])
            #muon_pphi[muon_pphi<0] = muon_pphi[muon_pphi<0]+2*3.14159
            #print ('muon_pphi:',muon_pphi)
            #print ('muon_p:',muon_p)
            #reco_muon = np.concatenate((muon_ptheta, muon_pphi, muon_theta, muon_phi, muon_r, muon_p),axis=-1) 
            #print ('reco muon:\n',reco_muon)
            #print ('real muon:\n',np.squeeze(muon_batch,axis=(-1,)))

