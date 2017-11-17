#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:12:01 2016

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
"""

import tensorflow as tf
import numpy as np
import h5py 
from matplotlib import pyplot as plt
from scipy import stats
import scipy as scipy
import sklearn.model_selection as sk
#import modeldef_conv as mod

def handle_input_data(read_data, preprocess_data, inputmat_name, outputmat_name):

   #ratio between train and test samples 
    p_train = 0.8
    p_test = 0.2
    
    # number of pixels along X/Y
    n_length = 64

    # choose if onyl a subset of the enitre dataset is used for training/testing    
    input_subset = -1#batch_size


    if read_data:  
        if(0):
        ##load input data; new MATLAB v7.3 Format! 
            mat_input = h5py.File(inputmat_name)
            x_input_hdf5 = mat_input['nninputs']
            x_input_raw = np.array(x_input_hdf5)
            
            #load output data
            mat_output = h5py.File(outputmat_name)
            y_input_hdf5 = mat_output['nnoutputs']
            y_input_raw = np.array(y_input_hdf5)
            print("Data was read-in sucessfully!")
            
            x_input_raw = np.transpose(x_input_raw, [3, 1, 2, 0])
            y_input_raw = np.transpose(y_input_raw, [1, 0])
        else:
            
            # this case is for loading a matfile gerenated with the TF gradient descent routine in "tf_illopt....py"
            x_input_raw = scipy.io.loadmat(inputmat_name).values()[0]
            y_input_raw = scipy.io.loadmat(outputmat_name).values()[1] 
            
            
            
            print("Data was read-in sucessfully!")
            
            
        # data has to be in the form (samples, x_pixel, y_pixel, channels)
    if preprocess_data:
        
        print("Seperate Data...")
    

        # generate train, test and validation datasets 80/10/10
        train_x, test_x, train_y, test_y = sk.train_test_split(x_input_raw, y_input_raw, test_size=0.2, random_state = 42)
        if(0):
            test_x, valid_x, test_y, valid_y = sk.train_test_split(test_x, test_y, test_size=0.5, random_state = 42)
        
        print('Data was seperated sucessfully!')
        print("check if reshape is reversable!")
#        plt.imshow(train_x_imag[0,:].reshape(128, 128), interpolation='nearest', cmap='gray')
#        plt.draw()
#        plt.show()
#        plt.pause(.01)
#        

        
        return train_x, train_y, test_x, test_y#, valid_x, valid_y
    