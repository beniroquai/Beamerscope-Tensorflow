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




from __future__ import print_function

import tensorflow as tf
import numpy as np
import h5py 
from matplotlib import pyplot as plt
from scipy import io
import scipy as scipy
from scipy import stats
import modeldef_conv_2D as mod
import input_data as indata
import time


# num of iterations
num_steps = 1000000  # 272640
batch_size = 128



# logpath for tensorboard
logs_path = "./logs/nn_logs"
  
# necessary to load and/or preprocess the data again? This takes lots of time 
# and can be disabled for debugging purposes
read_data = True#
preprocess_data = True

# Parameters
learning_rate = 0.001
training_iters = num_steps
display_step = 5

# Network Parameters
n_length = 128
n_input = n_length**2 # Complex input image size (i.e. img shape: 28*28)
n_classes = 48 # Total output nodes (i.e. 48 illuminstion segments)
dropout = 0.5 # Dropout, probability to keep units

inputmat_name = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Python/Beamerscope_TENSORFLOW/Beamerscope_IllOpt/nninputs.mat'

outputmat_name = '/home/useradmin/Dropbox/Dokumente/Promotion/PROJECTS/Beamerscope/Python/Beamerscope_TENSORFLOW/Beamerscope_IllOpt/nnoutputs.mat'

# read and preprocess data
train_x_cplx_2D, train_y, test_x_cplx_2D, test_y = indata.handle_input_data(read_data, preprocess_data, inputmat_name, outputmat_name)


# session.close()
tf.reset_default_graph()



n_feat_1 = n_length #128
n_feat_2 = 128
n_fully = 256


# Store layers weight & bias
weights = {
    # [5, 5, 1, 32] = kernel_m, kernel_n, feature_n_old, feature_n_new           
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([9, 9, 2, n_feat_1]), name = 'wc1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([7, 7, n_feat_1, n_feat_2]), name = 'wc2'),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([8*8*n_feat_2*4*4, n_fully]), name = 'wd1'),
    # 1024 inputs, 10 outputs (class prediction)
    'wout': tf.Variable(tf.random_normal([n_fully, n_classes]), name = 'wout')
}


biases = {
    'bc1': tf.Variable(tf.random_normal([n_feat_1]), name = 'bc1'),
    'bc2': tf.Variable(tf.random_normal([n_feat_2]), name = 'bc2'),
    'bd1': tf.Variable(tf.random_normal([n_fully]), name = 'bd1'),
    'bout': tf.Variable(tf.random_normal([n_classes]), name = 'bout')
}


# input images
with tf.name_scope('input'):
    # tf Graph input
    x_cplx = tf.placeholder(tf.float32, [None, n_length, n_length, 2], name = 'input_r')
    y = tf.placeholder(tf.float32, [None, n_classes], name = 'yin')
    keep_prob = tf.placeholder(tf.float32, name = 'dropout-factor') #dropout (keep probability)
    train_phase = tf.placeholder(tf.bool, name='phase_train')

# Construct model
pred = mod.conv_net(x_cplx, weights, biases, keep_prob, train_phase, generation_phase = False)

# get cost
cost, optimizer = mod.get_cost(pred, y, learning_rate)

# get accuracy
correct_pred, accuracy = mod.get_accuracy(pred, y)

## Tensorboard-STUFF
# Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
    
# summarize all tensorboard summaries
summary_op = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
sess = tf.InteractiveSession() #with tf.Session() as sess: #
sess.run(init)
step = 1

# initialize variables
loss_iter = []
acc_iter = []
t = time.time()



# create writer object
writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


# start process here
while step * batch_size < training_iters:
    
    offset = (step * batch_size) % (train_y.shape[0] - batch_size)
    # Generate a minibatch.
    # Keep training until reach max iterations
    batch_x_cplx = train_x_cplx_2D[offset:(offset + batch_size), :, :, :] # np.random.rand(128, n_input) # 
    batch_y = train_y[offset:(offset + batch_size), :] # np.random.rand(128, 48) #
    
                      
    # backup all variables
#    c step % 4000 == 0:
#        save_path = tf.train.Saver() tf.saver.save(sess, "./logs/nn_logs/model.ckpt")
#        print("Model saved in file: %s" % save_path)                      
                              
    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict={x_cplx: batch_x_cplx, y: batch_y, keep_prob: dropout, train_phase: True})
    if step % display_step == 0:
        # Calculate batch loss and accuracy
        loss, weak_acc, acc, summary = sess.run([cost, correct_pred, accuracy, summary_op], feed_dict={x_cplx: batch_x_cplx, y: batch_y, keep_prob: 1.,  train_phase: False})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc) + ", Training (weak) Accuracy= " + \
              "{:.5f}".format(weak_acc))
        
                
        # write log
        writer.add_summary(summary, step)
        loss_iter.append(loss)
        acc_iter.append(acc)
        
    step += 1
    
print("Optimization Finished!")


elapsed = time.time() - t
print("Elapsed time:")
print(elapsed)

# show one conv-filter
plt.imshow(np.squeeze(weights['wc1'].eval()[:,:,0,0]), interpolation = 'nearest', cmap = 'gray')
plt.show()


# test learned network with random spectrum
testobj = np.concatenate((np.random.rand(1, n_length, n_length,1)*.005, np.random.rand(1, n_length, n_length, 1)), 3)*.05
y_pred = sess.run(pred, feed_dict={x_cplx: testobj, keep_prob: 1., train_phase: False})
plt.plot(y_pred.T)
plt.show()

#    # Calculate accuracy for 256 mnist test images
#    print("Testing Accuracy:", \
#        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
#                                      y: mnist.test.labels[:256],
#                                      keep_prob: 1.}))




print('Execute: ')
print('cd /home/useradmin/Documents/Benedict\ /Tensorflow/TF_Illopt_Modeldef_conv_2D/TF_Illopt_Modeldef_conv_2D/logs/')
print('tensorboard --logdir=run1:./nn_logs\ --port 6006')
print(' and visit this page: http://localhost:6006/')







#################recreate model and safe for android###########################
    


# extract/backup all learned variables
wc1 = weights['wc1'].eval(sess)
wc2 = weights['wc2'].eval(sess)
wd1 = weights['wd1'].eval(sess)
out1 = weights['wout'].eval(sess)

bc1 = biases['bc1'].eval(sess)
bc2 = biases['bc2'].eval(sess)
bd1 = biases['bd1'].eval(sess)
bout = biases['bout'].eval(sess)


V = weights['wc1']
_, ix, iy, channels = V.get_shape().as_list()


V = tf.slice(V,(0,0,0,0),(1,-1,-1,-1)) #V[0,...]
V = tf.reshape(V,(iy,ix,channels))


# Next add a couple of pixels of zero padding around the image
ix += 4
iy += 4
V = tf.image.resize_image_with_crop_or_pad(V, iy, ix)

# reshape so that instead of 32 channels you have 4x8 channels, lets call them cy=4 and cx=8
cx=8
cy=16
V = tf.reshape(V,(iy,ix,cy,cx)) 

# transform to numpy
V = tf.transpose(V,(2,0,3,1)) #cy,iy,cx,ix

# image_summary needs 4d input
V = tf.reshape(V,(1,cy*iy,cx*ix,1))

# add to image summary
tf.summary.image('Conv_WC1', V)



# to visualize 1st conv layer output
# http://stackoverflow.com/questions/33802336/visualizing-output-of-convolutional-layer-in-tensorflow
V = weights['wc2']
ix, iy, _, channels = V.get_shape().as_list()
cx=16
cy=8
vv1 = V.eval()
vv1 = vv1[:,:,0,:]   # in case of bunch out - slice first img

v  = mod.vis_conv(vv1,ix,iy,channels,cy,cx)


plt.figure(figsize = (8,8))
plt.imshow(v,cmap="Greys_r",interpolation='nearest')



# write data to summary file
writer = tf.summary.FileWriter('./newgraph', graph=tf.get_default_graph())



sess.close()






# Create new graph for exporting
g_2 = tf.Graph()
with g_2.as_default():
    # Store layers weight & bias
    weights_store = {
        # [5, 5, 1, 32] = kernel_m, kernel_n, feature_n_old, feature_n_new           
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.constant(wc1, name = 'wc1'),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.constant(wc2, name = 'wc2'),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.constant(wd1, name = 'wd1'),
        # 1024 inputs, 10 outputs (class prediction)
        'wout': tf.constant(out1, name = 'wout')
    }
    
    
    biases_store = {
        'bc1': tf.constant(bc1, name = 'bc1'),
        'bc2': tf.constant(bc2, name = 'bc2'),
        'bd1': tf.constant(bd1, name = 'bd1'),
        'bout': tf.constant(bout, name = 'bout')
    }

    # tf Graph input
    x_store = tf.placeholder(tf.float32, [None, n_length, n_length, 2], name = 'input')
    keep_prob_store = tf.constant(1., tf.float32, name = 'dropout-factor') #dropout (keep probability)
    is_training = tf.constant(False)
    
    # Construct model
    y_store = mod.conv_net(x_store, weights_store, biases_store, keep_prob_store, is_training, generation_phase = True)
    
    sess_2 = tf.Session()
    init_2 = tf.global_variables_initializer();
    sess_2.run(init_2)
    
    # test learned network with random spectrum
    testobj = np.concatenate(((np.random.rand(1, n_length, n_length, 1))*0.001, 2*(np.random.rand(1, n_length, n_length, 1))-0), 3)
    y_pred = sess_2.run(y_store, feed_dict={x_store: testobj, keep_prob_store: 1.})
    plt.plot(y_pred.T)
    plt.show()    

    graph_def = g_2.as_graph_def()
    tf.train.write_graph(graph_def, logs_path, 'expert-graph_CN.pb', as_text=False)

    # Test trained model
    #    y__2 = tf.placeholder("float", [None, 10])
    #    correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y__2, 1))
    #    accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))

    #    print "check accuracy %g" % accuracy_2.eval(
    #{x_2: mnist.test.images, y__2: mnist.test.labels}, sess_2)
    


    
