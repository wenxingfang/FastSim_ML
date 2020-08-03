import tensorflow as tf
import numpy as np
import os

def predict(input_data, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            # print (tensors)

        session_conf = tf.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        #with tf.Session() as sess:
        with tf.Session(config=session_conf) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.graph.get_operations()
            input_x = sess.graph.get_tensor_by_name("gen_input:0")  
            out = sess.graph.get_tensor_by_name("re_lu_5/Relu:0")  
            img_out = sess.run(out,feed_dict={input_x: input_data})

            #print (img_out)
            return img_out



if __name__ == '__main__':
    pb_path = '/hpcfs/juno/junogpu/fangwx/FastSim/BES/generator/model_em_low.pb'
    batch_size = 1
    latent_size = 512
    noise = np.full((batch_size, latent_size), 0.5 ,dtype=np.float32)
    sampled_mom        = np.full((batch_size, 1), 1.8    ,dtype=np.float32)
    sampled_M_dtheta   = np.full((batch_size, 1), 0.5    ,dtype=np.float32)
    sampled_M_dphi     = np.full((batch_size, 1), -5.0/10,dtype=np.float32)
    sampled_P_dz       = np.full((batch_size, 1), 0.6    ,dtype=np.float32)
    sampled_P_dphi     = np.full((batch_size, 1), 0.6    ,dtype=np.float32)

    '''
    noise = np.random.normal(0, 1, (batch_size, latent_size))
    sampled_mom        = np.random.uniform( 1.7  , 1.8 ,(batch_size, 1))
    sampled_M_dtheta   = np.random.uniform(0.49 , 0.51 ,(batch_size, 1))
    sampled_M_dphi     = np.random.uniform(-5.1 , -5.0 ,(batch_size, 1))/10
    sampled_P_dz       = np.random.uniform(0.59 , 0.61, (batch_size, 1))
    sampled_P_dphi     = np.random.uniform(0.59 , 0.61, (batch_size, 1))
    '''
    #sampled_Z          = np.random.uniform(-119 ,-118 , (batch_size, 1))/100
    #sampled_info       = np.concatenate((noise, sampled_mom, sampled_M_dtheta, sampled_M_dphi,sampled_P_dz, sampled_P_dphi, sampled_Z),axis=-1)
    sampled_info       = np.concatenate((noise, sampled_mom, sampled_M_dtheta, sampled_M_dphi,sampled_P_dz, sampled_P_dphi),axis=-1)
    result = predict(sampled_info, pb_path)
    print (result)
    print (result.shape)
