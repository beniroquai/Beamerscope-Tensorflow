'''
Author: Benedict Diederich
%% Generate a dataset with complex objects and its corresponding optimized
% illumination shapes using the TCC
% 
% The software is for private use only and gives no guarrantee, that it's
% working as it should! 
% 
%
% Written by Benedict Diederich, benedict.diederich@leibniz-ipht.de
% www.nanoimaging.de
% License: GPL v3 or later.

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
from scipy import stats
import time
import scipy as scipy
from scipy import ndimage
#def contrastperpixel( grayImage ):
## contrastperpixel Contrast per Pixel Calculation from an input image
##   Detailed explanation goes here
#kernel = np.array((-1, -1, -1, -1, 8, -1, -1, -1))/8
#diffImage = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(grayImage, 0),3), kernel, strides=[1, 1, 1], padding='SAME')
#cpp = mean2(diffImage);
#
#end



def getSegment(xopt):
    def cart2pol(x, y):
        theta = np.arctan2(y, x)
        rho = np.hypot(x, y)
        return theta, rho

    xopt = np.random.rand(48,1)    
    kx = np.linspace(-1.5, 1.5, 100)
    ky = kx
    XX, YY  = np.meshgrid(kx, ky)
    
    mr=np.sqrt(XX**2+YY**2)
    mtheta, mr =  cart2pol(XX, YY)
    Po=1.*(mr<=1)
    n_segment = 12;     # number of colour segments
   
    Ic = np.zeros(Po.shape)
    
    for i in range(0, xopt.shape[0]):
        Isegment = np.zeros(Po.shape)
                
        # i-th segment in one of the annuli
        i_segment = np.mod(i, n_segment)
    
        if (np.int16(i/n_segment) == 0):
            NAc_i = 0;
            NAc_o = 0.25;
        elif (np.int16(i/n_segment) == 1):
            NAc_i = 0.25;
            NAc_o = 0.5;
        elif (np.int16(i/n_segment) == 2):
            NAc_i = 0.5;
            NAc_o = .75;
        elif (np.int16(i/n_segment) == 3):
            NAc_i = 0.75;
            NAc_o = 1;
        
          
        # Creating the annullar shape 0,1,2,3
        Isegment= (1.*(mr>=NAc_i) * 1.*(mr<=NAc_o)) #Inner and Outer radius.
            
    
        # scale rotational symmetric ramp 0..1
        segment_area = (mtheta)/np.max(mtheta) * np.round(n_segment/2) + np.round(n_segment/2);
        
        # each segment results from the threshold of the grayvalues
        # filtered by the annular shape of the illumination sector
        # 0,1,2
        
        # this is due to aliasing of the pixelated source, otherwise
        # there will be a gap in the source shape
        if(i_segment == n_segment-1):
            segment_area = 1.*(segment_area >= i_segment) * 1.*(segment_area < (i_segment+1)*1.00001)
        else:
            segment_area = 1.*(segment_area >= i_segment) * 1.*(segment_area < (i_segment+1))
        
        
        
        # get i-th segment and sum it up; weigh it with coefficient
        segment_area = segment_area*Isegment;
        Isegment = segment_area*xopt[i]
        
        Ic = Ic + Isegment;
    return Ic
            
        
def resize_by_axis(image, dim_1, dim_2, ax, is_grayscale):
    resized_list = []
    
    
    if is_grayscale:
        unstack_img_depth_list = [tf.expand_dims(x,2) for x in tf.unstack(image, axis = ax)]
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.squeeze(tf.stack(resized_list, axis=ax))
        print(stack_img.get_shape())
    
    else:
        unstack_img_depth_list = tf.unstack(image, axis = ax)
        for i in unstack_img_depth_list:
            resized_list.append(tf.image.resize_images(i, [dim_1, dim_2],method=0))
        stack_img = tf.stack(resized_list, axis=ax)
    
    return stack_img
    


def tf_normminmax(x):
   # x is your tensor
   
   x = x-tf.reduce_min(x)
   x = x/tf.reduce_max(x)
   return x


def tf_normmax(x):
   # x is your tensor
   

   x = x/tf.reduce_max(x)
   return x


def MeanSquareError(origImg, distImg):
    return tf.reduce_sum(tf.square(origImg - distImg))
    

def tf_fftshift(tensor):
    ndim = len(tensor.shape)
    for i in range(ndim):
        n = tensor.shape[i].value
        p2 = (n+1) // 2
        begin1 = [0] * ndim
        begin1[i] = p2
        size1 = tensor.shape.as_list()
        size1[i] = size1[i] - p2
        begin2 = [0] * ndim
        size2 = tensor.shape.as_list()
        size2[i] = p2
        t1 = tf.slice(tensor, begin1, size1)
        t2 = tf.slice(tensor, begin2, size2)
        tensor = tf.concat([t1, t2], axis=i)
    return tensor


def tf_ifftshift(tensor):
    ndim = len(tensor.shape)
    for i in range(ndim):
        n = tensor.shape[i].value
        p2 = n - (n + 1) // 2
        begin1 = [0] * ndim
        begin1[i] = p2
        size1 = tensor.shape.as_list()
        size1[i] = size1[i] - p2
        begin2 = [0] * ndim
        size2 = tensor.shape.as_list()
        size2[i] = p2
        t1 = tf.slice(tensor, begin1, size1)
        t2 = tf.slice(tensor, begin2, size2)
        tensor = tf.concat([t1, t2], axis=i)
    return tensor


def tv_loss(x):
    #https://github.com/utkarsh2254/compression-artifacts-reduction/blob/6ebf11ff813e4bb64ab8437a56c6ffd2f99b1f7a/baseline/Losses.py
  def total_variation(images):
    pixel_dif1 = images[1:, :] - images[:-1, :]
    pixel_dif2 = images[:, 1:] - images[:, :-1]
    sum_axis = [0, 1]
    tot_var = tf.reduce_sum(tf.abs(pixel_dif1), axis=sum_axis) + \
              tf.reduce_sum(tf.abs(pixel_dif2), axis=sum_axis)
    return tot_var

  loss = tf.reduce_sum(total_variation(x))
  return loss


# logpath for tensorboard
logs_path = "./logs/nn_logs"
  


# Parameters
learning_rate = 0.01

display_step = 25
# num of iterations
num_steps = 150  # 272640

matlab_data = './tf_illoptdata.mat'
object_data = './PreProcessedDataNN.mat'

with tf.device('/gpu:0'):
    
    
     ##load system data; new MATLAB v7.3 Format! 
    mat_matlab_data = h5py.File(matlab_data)
    mat_eigenfunction = np.array(mat_matlab_data['eigenfunction'])
    mat_eigenvalue = np.array(mat_matlab_data['eigenvalue'])
    mat_ill_method = np.array(mat_matlab_data['ill_method'])
        
    
    ##load input data; new MATLAB v7.3 Format! 
    mat_object_data = h5py.File(object_data)
    #mat_complxObject = np.array(mat_object_data['complxObject'])
    mat_complxObject = mat_object_data['complxObject'].value.view(np.double)
    mat_complxObject = mat_complxObject.reshape(mat_complxObject.shape[0], mat_complxObject.shape[1], mat_complxObject.shape[2]/2, 2)
    mat_complxObject = mat_complxObject[:,:,:,0] + 1j*mat_complxObject[:,:,:,1]
    
    
    # session.close()
    tf.reset_default_graph()
    
    # determine system parameters from Matlabs eigenfunction
    n_illpattern, n_eigfct, n_system, m_system = mat_eigenfunction.shape
    
    # determine system parameters from Matlabs eigenfunction
    n_samples, n_object, m_object = mat_complxObject.shape
    
    
    # Finally aerial image can be calculated by equation (3):
    # padsize = round((size(objectspectrum, 2) - size(eigenfunction(:, :, 1),2))/2);
    scalefactor = n_object/n_eigfct;
    
    
    # Convert Matlab to Tensorflow
    n_samples_batch = 25;
    n_kernel = 5;
    
    
    
    tf_object_real = tf.placeholder(dtype=tf.float32, shape=np.real(mat_complxObject[0,:,:]).shape)
    tf_object_imag = tf.placeholder(dtype=tf.float32, shape=np.real(mat_complxObject[0,:,:]).shape)
    tf_object = tf.cast(tf.complex(tf_object_real, tf_object_imag), dtype=tf.complex64)
    tf_eigenfct = tf.constant(mat_eigenfunction)
    tf_eigenval = tf.constant(mat_eigenvalue)
    tf_xopt = tf.Variable(tf.ones([n_illpattern, 1], dtype=tf.float64))
    
    # get spectrum of the object =
    tf_object_FT = tf_fftshift(tf.ifft2d(tf_ifftshift(tf_object)))
    tf_I = tf.zeros([n_object, m_object], dtype=tf.float32)
    
    for j in range(0, n_illpattern):
        print("illpattern: "+str(j))

        tf_eigenfct_i = tf_eigenfct[j,0:n_kernel,:,:]
        tf_eigenfct_i = tf.transpose(tf_eigenfct_i, [1, 2, 0])
        
        tf_eigenfct_i = tf.image.resize_images(tf_eigenfct_i, [n_object, m_object], method=0)
        tf_eigenfct_i = tf.transpose(tf_eigenfct_i, [2, 0, 1])
        tf_aerial = tf_object_FT*tf.complex(tf_eigenfct_i, tf_eigenfct_i*0)
        
        tf_aerial_FT = tf_fftshift(tf.fft2d(tf_ifftshift(tf_aerial)));
        
        
        tf_eigenval_i = tf.expand_dims(tf.expand_dims(tf_xopt[j]*tf.square(tf_eigenval[j, 0:n_kernel]), 1), 2) 
        tf_I = tf_I + tf.reduce_sum(tf_eigenval_i*tf.abs(tf.conj(tf_aerial_FT)*tf_aerial_FT), [0])


    # try to do it in one step
    tf_eigenfct_i = tf_eigenfct[:,0:n_kernel,:,:]
    #tf_eigenfct_i = tf.transpose(tf_eigenfct_i, [1, 2, 0])
    
    tf_eigenfct_i = tf_image_resize_4d(tf_eigenfct_i, n_object, m_object)
    
    
    resized_along_depth = resize_by_axis(tf_eigenfct_i,n_kernel,n_object,3, False)
    resized_along_width = resize_by_axis(resized_along_depth,n_kernel,n_object,2,False)


    
    tf.image.resize_images(tf_eigenfct_i, [n_kernel, n_object, m_object], method=0)
    tf_eigenfct_i = tf.transpose(tf_eigenfct_i, [2, 0, 1])
    tf_aerial = tf_object_FT*tf.complex(tf_eigenfct_i, tf_eigenfct_i*0)
    
    tf_aerial_FT = tf_fftshift(tf.fft2d(tf_ifftshift(tf_aerial)));
    
    
    tf_eigenval_i = tf.expand_dims(tf.expand_dims(tf_xopt[j]*tf.square(tf_eigenval[j, 0:n_kernel]), 1), 2) 
    tf_I = tf_I + tf.reduce_sum(tf_eigenval_i*tf.abs(tf.conj(tf_aerial_FT)*tf_aerial_FT), [0])

    # define the cost function as the mean squared difference between the objects phase and the intensity
    if(0):
        tf_I_iter_norm = tf_normmax(tf_I)
        tf_object_iter_norm = tf_normmax(tf.cast(tf.abs(tf_object),dtype=tf.float64))
        tf_cost = MeanSquareError(tf_object_iter_norm, tf_I_iter_norm)
    if(0):
        tf_cost = tf.image.total_variation(tf.expand_dims(tf_I,0))
    if(0):
        # Contrast per Pixel
        kernel = 1.*np.array([[-1, -1, -1],[-1, 9, -1],[-1, -1, -1]])
        tf_kernel = tf.constant(kernel[ :, :, np.newaxis, np.newaxis])
        #kernel = np.repeat(kernel[:, :, np.newaxis], 3, axis=2)
        
        tf_diffImage = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tf_I, axis=2), axis=3), tf_kernel, strides=[1, 1, 1, 1], padding='SAME')
        tf_cpp = tf.reduce_mean(tf_diffImage)
        tf_cost = tf_cpp
    if(0):
        tf_cost = -tf.reduce_sum(tf.image.total_variation(tf.expand_dims(tf_I,0)))
    if(1):
        tf_cost = -tv_loss(tf_I)
        if(0):
            # debug
            images = tf_I
            pixel_dif1 = images[1:, :] - images[:-1, :] # y
            pixel_dif2 = images[:, 1:] - images[:, :-1] # x
            sum_axis = [0, 1]
            pixel_dif1_, pixel_dif2_, I_test = sess.run([pixel_dif1, pixel_dif2, tf_I], feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})
        
            plt.imshow(pixel_dif2_)
            
            
    # minimize the error
    tf_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(tf_cost)
        
    # Initializing the variables
    init = tf.global_variables_initializer()
    
    # Launch the graph
    sess = tf.InteractiveSession() #with tf.Session() as sess: #
    #sess = tf.Session()
    sess.run(init)
    step = 1
    
    # initialize variables
    loss_object_iter = []
    loss_iter = []
    xopt_iter  = []
    object_rotate_iter = []
    t = time.time()
    
    for object_iter in range(111,n_samples):
        for step in range(0,num_steps):
        
        
            object_i_real = np.real(mat_complxObject[object_iter,:,:])
            object_i_imag = np.imag(mat_complxObject[object_iter,:,:])
            
            sess.run([tf_optimizer], feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})
    
            # debug 
            # I_iter_norm, object_iter_norm, I_i, object_i = sess.run([tf_I_iter_norm, tf_object_iter_norm, tf_I, tf_object], feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})        
            
    
            # restrict optimization parameters to 0..1
            tf_xopt = tf.where(
                tf.less(tf_xopt, tf.zeros_like(tf_xopt)),
                tf.zeros_like(tf_xopt),
                tf_xopt)
            
            tf_xopt = tf.where(
                tf.greater(tf_xopt, tf.ones_like(tf_xopt)),
                tf.ones_like(tf_xopt),
                tf_xopt)
            
    
    
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                I_iter, loss = sess.run([tf_I, tf_cost],  feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})
                print("Object: " + str(object_iter) + "/ " + str(n_samples) + "- Iter " + str(step) + ", Loss= " + "{:.4f}".format(loss))
    
                loss_object_iter.append(loss)
                plt.imshow(I_iter, cmap='gray')
                plt.colorbar()
                plt.show()
                    
                
                
        # compute results
        object_iter_cmplx, I_iter, xopt= sess.run([tf_object, tf_I, tf_xopt], feed_dict = {tf_object_real: object_i_real, tf_object_imag: object_i_imag})
        
        
        # save values for later use
        loss_iter.append(loss_object_iter)
        
        for iAngles in range(0,12):
        
        
            # rotate optimized pattern as well and bring back into 1D-vector
            # shape 
            xopt_shift = np.reshape(np.roll(np.reshape(xopt, [12, 4]), iAngles, 1), [48, 1]);
            
            # rotate object => generates more data!
            object_rotate_real = scipy.ndimage.interpolation.rotate(np.real(object_iter_cmplx), iAngles*22.5, reshape=False);
            object_rotate_imag = scipy.ndimage.interpolation.rotate(np.imag(object_iter_cmplx), iAngles*22.5, reshape=False);
            
            object_rotate = object_rotate_real + 1j*object_rotate_imag
            
            xopt_iter.append(xopt_shift)
            object_rotate_iter.append(object_rotate)
        
        
        
        
                
        # show the image with enhanced contrast 
        print('this is the optimized image')
        plt.imshow(I_iter, cmap='gray')
        plt.colorbar()
        plt.show()
        plt.imshow(np.imag(object_iter_cmplx), cmap='gray')
        plt.colorbar()
        plt.show()
        
        
        plt.imshow(getSegment(xopt), cmap='gray')
        plt.colorbar()
        plt.show()

        
        elapsed = time.time() - t
        t = time.time() 
        print("Elapsed time:")
        print(elapsed)
        
        
        
        
import shelve

filename='/tmp/shelve.out'
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('TypeError shelving: {0}'.format(key))
    except:
        # catches everything else
        print('Generic error shelving: {0}'.format(key))
my_shelf.close()


# To restore:

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()