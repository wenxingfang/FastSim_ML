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
            input_x = sess.graph.get_tensor_by_name("x:0")  
            out = sess.graph.get_tensor_by_name("dense_2/Elu:0")  
            img_out = sess.run(out,feed_dict={input_x: input_data})

            #print (img_out)
            return img_out

def predict_v1(pb_file_path, r, theta, batch_size, N_event, out_file):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            # print (tensors)

        session_conf = tf.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        #with tf.Session() as sess:
        f_out = open(out_file,'w') 
        r        = np.full((batch_size, 1), r      ,dtype=np.float32)/20
        theta    = np.full((batch_size, 1), theta  ,dtype=np.float32)/180
        with tf.Session(config=session_conf) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.graph.get_operations()
            input_x = sess.graph.get_tensor_by_name("x:0")  
            #out = sess.graph.get_tensor_by_name("dense_2/Elu:0")  
            out = sess.graph.get_tensor_by_name("dense_2/add:0")  
            for i in range(N_event):
                noise = np.random.normal(0, 1, (batch_size, 1))
                sampled_info       = np.concatenate((r, theta, noise),axis=-1)
                result = sess.run(out,feed_dict={input_x: sampled_info})*100
                f_out.write(str(result[0,0]))
                f_out.write('\n')
        f_out.close()
        print ('done')

if __name__ == '__main__':
    #pb_path = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/hit_time.pb'
    pb_path = '/hpcfs/juno/junogpu/fangwx/FastSim/JUNO/LearnPDF/hit_time_new.pb'
    batch_size = 1
    latent_size = 1
    '''
    #noise = np.full((batch_size, latent_size), 0.1 ,dtype=np.float32)
    r        = np.full((batch_size, 1), 6.401   ,dtype=np.float32)/20
    theta    = np.full((batch_size, 1), 25.1   ,dtype=np.float32)/180
    f_out = open('time_test.txt','w') 
    for i in range(10000):
        noise = np.random.normal(0, 1, (batch_size, 1))
        sampled_info       = np.concatenate((r, theta, noise),axis=-1)
        result = predict(sampled_info, pb_path)*100
        f_out.write(str(result[0,0]))
        f_out.write('\n')
        #print (result)
        #print (result.shape)
    f_out.close()
    print ('done')
    '''

    predict_v1(pb_file_path=pb_path, r=6.401, theta=25.1, batch_size=batch_size, N_event=10000, out_file='time_test.txt')
