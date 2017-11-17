#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Author: Benedict Diederich
## This is the convolutional Neural network which tries to find an implicit model
# between complex object transmission functions and its optimized illumination 
# source shapes which enhance the phase-contrast in the image
# 
# The software is for private use only and gives no guarrantee, that it's
# working as it should! 
# 
#
# Written by Benedict Diederich, benedict.diederich@leibniz-ipht.de
# www.nanoimaging.de
# License: GPL v3 or later.
'''

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm





# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1, name = 'conv2d'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', name=name)
    x = tf.nn.bias_add(x, b)
    return x


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
    

# used this solution here: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
def batch_norm_layer(x, train_phase, scope_bn='bn'):
    if train_phase == True:
        # bn_train
        z = tf.layers.batch_normalization(x, axis = -1, momentum=0.999, center=True, scale=True, training=True, reuse=None, trainable=True)
    else:
        # bn_inference 
        z = tf.layers.batch_normalization(x, axis = -1, momentum=0.999, center=True, scale=True, training=True, reuse=None, trainable=True)
        z = tf.layers.batch_normalization(x, axis = -1, momentum=0.999, center=True, scale=True, training=False, reuse=None, trainable=True)
    return z

# used this solution here: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
def tf_batch_norm_layer(x, train_phase, scope_bn='bn'):
    if train_phase == True:
        # bn_train
        z = batch_norm(x, decay=0.999, center=True, scale=True, updates_collections=None, is_training=True, reuse=None, trainable=True, scope=scope_bn)
    else:
        # bn_inference 
        z = batch_norm(x, decay=0.999, center=True, scale=True, updates_collections=None, is_training=False, reuse=None, trainable=True, scope=scope_bn)
    return z

#   return x 
    
        
#visualize conv-layer
#http://stackoverflow.com/questions/33802336/visualizing-output-of-convolutional-layer-in-tensorflow
def vis_conv(v,ix,iy,ch,cy,cx, p = 0) :
    v = np.reshape(v,(iy,ix,ch))
    ix += 2
    iy += 2
    npad = ((1,1), (1,1), (0,0))
    v = np.pad(v, pad_width=npad, mode='constant', constant_values=p)
    v = np.reshape(v,(iy,ix,cy,cx)) 
    v = np.transpose(v,(2,0,3,1)) #cy,iy,cx,ix
    v = np.reshape(v,(cy*iy,cx*ix))
    return v
    
# Create model
def conv_net(x_cplx, weights, biases, dropout, train_phase, generation_phase):
    # Reshape input picture
    #x = tf.reshape(x, shape=[-1, n_length, n_length, 2], name = 'x_reshape')
    
    # this has to be done in a different way! feed it with 2D images
    # this is really expenssive
    
    if(not generation_phase): #differentiate between training and generation of the graph - dropout is not yet implemented in native lib)
        with tf.variable_scope('conv_1'):
            # Convolution Layer
            conv1 = tf.nn.tanh(tf_batch_norm_layer(conv2d(x_cplx, weights['wc1'], biases['bc1']), train_phase=train_phase))
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(conv1, k=2)
    
        with tf.variable_scope('conv_2'):        
            # Convolution Layer
            conv2 = tf.nn.tanh(tf_batch_norm_layer(conv2d(conv1, weights['wc2'], biases['bc2']), train_phase=train_phase))
            
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(conv2, k=2)
    
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]], name="fc1_reshape")
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.tanh(fc1)
        
        fc1 = tf.nn.dropout(fc1, dropout)

    # Apply Dropout only in Training phase (Android won't like it)
    else:
        with tf.variable_scope('conv_1'):
            # Convolution Layer
            conv1 = tf.nn.tanh(conv2d(x_cplx, weights['wc1'], biases['bc1']))
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(conv1, k=2)

        with tf.variable_scope('conv_2'):        
            # Convolution Layer
            conv2 = tf.nn.tanh(conv2d(conv1, weights['wc2'], biases['bc2']))
            
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]], name="fc1_reshape")
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.tanh(fc1)

        #fc1 = tf.nn.dropout(fc1, dropout)
        
        

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'], name = 'output')
    return out

def get_cost(pred, y, learning_rate=0.001):    
    with tf.name_scope('cost-node'):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.square(pred - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(cost)
        
        # create a summary for our cost
        tf.summary.scalar("cost", cost)
    return cost, optimizer

def get_accuracy(pred, y):
    with tf.name_scope('accuracy-node'):
        
        # round the output i.e. discretize the contionus output-function 
        y_discr = tf.div(tf.round(y*10), 10)
        pred_discr = tf.div(tf.round(pred*10), 10)
        
        # compare the output of the calculated and original result
        correct_pred = tf.equal(y_discr, pred_discr)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # threshold the output to have a weaker accuracy measure
        y_thres = tf.round((.5*y+.5))
        pred_thres = tf.round((.5*pred+.5))
        correct_pred_weak = tf.equal(y_thres, pred_thres)
        accuracy_weak = tf.reduce_mean(tf.cast(correct_pred_weak, tf.float32))

                
        # create a summary for our accuracy
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("accuracy_weak", accuracy_weak)
        
    return accuracy_weak, accuracy