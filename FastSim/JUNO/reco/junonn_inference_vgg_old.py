from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np

TOWER_NAME = 'tower'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_data', '', 'input data')
tf.app.flags.DEFINE_string('load_weight', '', 'load weight')
tf.app.flags.DEFINE_integer('batch_size', 1 , 'batch size')
#tf.contrib.layers.xavier_initializer_conv2d()
#tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
def conv_op(input_op,kernel_shape,stride,name):
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w", shape=kernel_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())        
        conv=tf.nn.conv2d(input_op,kernel,strides=stride,padding="SAME")
        
        
        n_out=kernel_shape[-1]
        #biases=tf.Variable(tf.constant(0.0,shape=[n_out],dtype=tf.float32),trainable=True,name='b')
        biases=tf.get_variable(scope+"b", shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        z=tf.nn.bias_add(conv, biases)
        activation=tf.nn.relu(z,name=scope+'relu')
        return activation
        
def fc_op(input_op,n_out,name):
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        #initializer=tf.contrib.layers.xavier_initializer()
        kernel=tf.get_variable(scope+"w", shape=[n_in,n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
        #add L2 regularizer to w
        weight_decay = tf.multiply(tf.nn.l2_loss(kernel),0.004, name=scope+'weight_loss')
        tf.add_to_collection('losses', weight_decay)                
        
        biases=tf.get_variable(scope+"b", shape=[n_out], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        #activation=tf.nn.relu_layer(input_op, kernel, biases, name=scope+'relu')
        activation = tf.nn.relu(tf.matmul(input_op, kernel) + biases, name=scope+'relu')
        return activation
    
def mpool_op(input_op,pool_shape,stride,name):
    return tf.nn.max_pool(input_op, ksize=pool_shape, strides=stride, padding='SAME',  
                       name=name)

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    #tf.summary.histogram(tensor_name + '/activations', x)
    #tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def inference(images):
    conv1_1=conv_op(images, [3,3,2,64], [1,1,1,1],name='conv1_1')
    conv1_2=conv_op(conv1_1, [3,3,64,64], [1,1,1,1], name='conv1_2')
    pool1=mpool_op(conv1_2, [1,3,3,1], [1,2,2,1], name='pool1')
        
    conv2_1=conv_op(pool1, [3,3,64,128], [1,1,1,1],name='conv2_1')
    conv2_2=conv_op(conv2_1, [3,3,128,128], [1,1,1,1],name='conv2_2')
    pool2=mpool_op(conv2_2, [1,3,3,1], [1,2,2,1], name='pool2')
    conv3_1=conv_op(pool2, [3,3,128,256], [1,1,1,1],name='conv3_1')
    conv3_2=conv_op(conv3_1, [3,3,256,256], [1,1,1,1],name='conv3_2')
    conv3_3=conv_op(conv3_2, [3,3,256,256], [1,1,1,1],name='conv3_3')
    pool3=mpool_op(conv3_3, [1,3,3,1], [1,2,2,1], name='pool3')

    resh1 = tf.reshape(pool3, [FLAGS.batch_size, -1])
    fc6=fc_op(resh1, 1024,name='fc6')
    fc7=fc_op(fc6, 512,name='fc7')
    fc8=fc_op(fc7, 256,name='fc8')

    with tf.variable_scope('softmax_linear') as scope:
        kernel=tf.get_variable("w", shape=[256,6], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        biases=tf.get_variable("b", shape=[6], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        softmax_linear=tf.add(tf.matmul(fc8, kernel), biases, name=scope.name) 
        
    _activation_summary(fc6)
    _activation_summary(fc7)
    _activation_summary(fc8)
 

    return softmax_linear


if __name__ == '__main__':
    ### do some test #####
    
    import h5py
    d = h5py.File(FLAGS.input_data,'r')
    first = np.expand_dims(d['firstHitTimeByPMT'][:], -1)
    second = np.expand_dims(d['nPEByPMT'][:], -1)
    muon   = np.expand_dims(d['infoMuon'][:], -1)
   
    for index in range(0, 2):
        print('index:', index)
        image_batch_1 = first [index * FLAGS.batch_size:(index + 1) * FLAGS.batch_size]
        image_batch_2 = second[index * FLAGS.batch_size:(index + 1) * FLAGS.batch_size]
        muon_batch_   = muon  [index * FLAGS.batch_size:(index + 1) * FLAGS.batch_size]
        image = np.concatenate((image_batch_2, image_batch_1),axis=-1) # nPE should go first , then is time
        print (image.shape)
        image = tf.convert_to_tensor(image)
        if image.dtype == tf.float64:
            image = tf.cast(image, tf.float32)
        reco = inference(image)
        init_op = tf.initialize_all_variables()
        saver = tf.train.Saver()
        print ('H1')
        with tf.Session() as sess:
            sess.run(init_op)           
            print ('H2')
            saver.restore(sess, FLAGS.load_weight)
            print ('H3')
            print (sess.run(reco))            

