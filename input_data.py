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
import modeldef_conv as mod

def handle_input_data(read_data, preprocess_data, inputmat_name, outputmat_name):

   #ratio between train and test samples 
    p_train = 0.8
    p_test = 0.2
    
    # number of pixels along X/Y
    n_length = 64

    # choose if onyl a subset of the enitre dataset is used for training/testing    
    input_subset = -1#batch_size


    if read_data:  
        ##load input data; new MATLAB v7.3 Format! 
        mat_input = h5py.File(inputmat_name)
        x_input_hdf5 = mat_input['nninputs']
        x_input_raw = np.array(x_input_hdf5)
        
        #load output data
        mat_output = h5py.File(outputmat_name)
        y_input_hdf5 = mat_output['nnoutputs']
        y_input_raw = np.array(y_input_hdf5)
        print("Data was read-in sucessfully!")
    
    if preprocess_data:
    
        # select subset of main dataset    
        x_input = x_input_raw[:,:,:,0:input_subset]
        
        y_input = y_input_raw[:,0:input_subset]
        
        # shuffle dataset into train & test data
        size_input = x_input.shape
        
        train_idx = np.random.randint(size_input[3], size=int(p_train*(size_input[3])))
        test_idx = np.random.randint(size_input[3], size=int(p_test*(size_input[3])))
        
        train_x, test_x = x_input[:,:,:,train_idx], x_input[:,:,:,test_idx]
        train_y, test_y = y_input[:, train_idx], y_input[:,test_idx]
        
        # onyl needed if validation data is requiered
        if 0: 
            # shuffle dataset into test & valid data
            p,q = test_x.shape
            test_idx = np.random.randint(test_x.shape[0], size=int(0.5*p))
            valid_idx = np.random.randint(test_x.shape[0], size=int(0.5*p))
            
            test_x, valid_x = test_x[test_idx, :], test_x[valid_idx, :]
            test_y, valid_y = test_y[test_idx, :], test_y[valid_idx, :]
    
        

        
        print("Data was seperated sucessfully!")
    
        print("check if reshape is reversable!")
#        plt.imshow(train_x_imag[0,:].reshape(64,64), interpolation='nearest', cmap='gray')
#        plt.draw()
#        plt.show()
#        plt.pause(.01)
#        

        # bring back to 2D shape - due to 1D ordering from MATLAB
        train_x_cplx_2D = a = np.transpose(train_x, [3, 1, 2, 0]) 
        test_x_cplx_2D = a = np.transpose(test_x, [3, 1, 2, 0]) 
        
        train_y = train_y.T
        test_y = test_y.T
        
        return train_x_cplx_2D, train_y, test_x_cplx_2D, test_y
    