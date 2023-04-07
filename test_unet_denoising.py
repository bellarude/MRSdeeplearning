from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, \
    Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
# from keras import backend as K
import tensorflow.keras.backend as K

from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperModel, HyperParameters, RandomSearch
from kerastuner.tuners import BayesianOptimization

import time
from util import textMe
from numba import cuda
from models_unet_denoising import newModel

def dataimport(index):
    global y_train, y_val, y_test, X_train, X_val, X_test

    os.environ["KERAS_BACKEND"] = "theano"
    K.set_image_data_format('channels_last')

    dest_folder = 'C:/Users/Rudy/Desktop/datasets/spectra_for_1D_Unet/'  # 'C:/Users/Rudy/Desktop/datasets/dataset_31/'

    data_import_1 = sio.loadmat(dest_folder + 'X_noisy_specta_matRecon_1.mat')
    data_import_2 = sio.loadmat(dest_folder + 'X_noisy_specta_matRecon_2.mat')
    data_import_3 = sio.loadmat(dest_folder + 'X_noisy_specta_matRecon_3.mat')
    data_import_4 = sio.loadmat(dest_folder + 'X_noisy_specta_matRecon_4.mat')

    labels_import_1 = sio.loadmat(dest_folder + 'Y_GT_specta_matRecon_1.mat')
    labels_import_2 = sio.loadmat(dest_folder + 'Y_GT_specta_matRecon_2.mat')
    labels_import_3 = sio.loadmat(dest_folder + 'Y_GT_specta_matRecon_3.mat')
    labels_import_4 = sio.loadmat(dest_folder + 'Y_GT_specta_matRecon_4.mat')

    # cut last 1024 points, where the spectrum is
    dataset = np.concatenate([data_import_1['X_test_matlabRecon'], data_import_2['X_test_matlabRecon'], data_import_3['X_test_matlabRecon'], data_import_4['X_test_matlabRecon']], axis=1)              #, data_import_4['X_test_matlabRecon'][-1024:,:]], axis=1)
    dataset_ft = np.fft.fft(dataset, axis=0)
    dataset_ft_truncated = dataset_ft[-1024:, :]
    labels = np.concatenate([labels_import_1['Y_test_matlabRecon'], labels_import_2['Y_test_matlabRecon'], labels_import_3['Y_test_matlabRecon'], labels_import_4['Y_test_matlabRecon']], axis=1)         #, labels_import_4['Y_test_matlabRecon'][-1024:,:]], axis=1)
    labels_ft = np.fft.fft(labels, axis=0)
    labels_ft_truncated = labels_ft[-1024:, :]

    # here modify the test and train dataset so that I can load them wihtouth issues

    X_train = dataset_ft_truncated[:, 0:18000]
    X_test = dataset_ft_truncated[:, 18000:19000]
    X_val = dataset_ft_truncated[:, 19000:20000]  # check out

    y_train = labels_ft_truncated[:, 0:18000]
    y_test = labels_ft_truncated[:, 18000:19000]
    y_val = labels_ft_truncated[:, 19000:20000]

    '''
    X_train = dataset[0:18000, :]
    X_test = dataset[18000:19000, :]
    X_val = dataset[19000:20000, :]  # check out

    y_train = labels[0:18000, :]
    y_test = labels[18000:19000, :]
    y_val = labels[19000:20000, :]
    '''
    return dataset, labels

datapoints = 1024

# MD
dataimport(1)

modelUnet = newModel(type=0)

X_train_full = np.zeros([18000, 1024, 2])
X_reshaped = np.transpose(X_train, (1, 0))
X_train_full[:,:,0] = np.real(X_reshaped)
X_train_full[:,:,1] = np.imag(X_reshaped)
Y_train_full = np.zeros([18000, 1024, 2])
Y_reshaped = np.transpose(y_train, (1, 0))
Y_train_full[:,:,0] = np.real(Y_reshaped)
Y_train_full[:,:,1] = np.imag(Y_reshaped)

X_val_full = np.zeros([1000, 1024, 2])
X_reshaped = np.transpose(X_val, (1, 0))
X_val_full[:,:,0] = np.real(X_reshaped)
X_val_full[:,:,1] = np.imag(X_reshaped)
Y_val_full = np.zeros([1000, 1024, 2])
Y_reshaped = np.transpose(y_val, (1, 0))
Y_val_full[:,:,0] = np.real(Y_reshaped)
Y_val_full[:,:,1] = np.imag(Y_reshaped)

X_test_full = np.zeros([1000, 1024, 2])
X_reshaped = np.transpose(X_test, (1, 0))
X_test_full[:,:,0] = np.real(X_reshaped)
X_test_full[:,:,1] = np.imag(X_reshaped)
Y_test_full = np.zeros([1000, 1024, 2])
Y_reshaped = np.transpose(y_test, (1, 0))
Y_test_full[:,:,0] = np.real(Y_reshaped)
Y_test_full[:,:,1] = np.imag(Y_reshaped)

# for DL RR quantification
GT_data_ = np.concatenate([Y_train_full, Y_test_full, Y_val_full], axis=0)
X_noisy_data_ = np.concatenate([X_train_full, X_test_full, X_val_full], axis=0)
# store them
#saveDir = 'C:/Users/Rudy/Desktop/toMartyna/toRUDY/'
#np.save(saveDir + 'GT_data.npy', GT_data_)
#np.save(saveDir + 'X_noisy_data.npy', X_noisy_data_)
np.save(saveDir + 'pred_denoised_DL.npy', pred)

output_folder = 'C:/Users/Rudy/Desktop/denoising_unet/'
net_name = "RR_1e3_learRate_fullDataset"   # RR_1e4_learRate_fullDataset.best
'''
RR_1e3_learRate_fullDataset
RR_1e4_learRate_32_batch_fullDataset

'''
checkpoint_path = output_folder + net_name + ".best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

modelUnet.load_weights(checkpoint_path)

loss = modelUnet.evaluate(X_test_full, Y_test_full, verbose=2)

pred = modelUnet.predict(X_noisy_data_)  # X_test_full normalized [0-1] absolute concentrations prediction


#
plt.figure()
plt.plot(Y_test_full[0,:,0])
plt.plot(pred[0,:,0])

plt.figure()
plt.plot(Y_test_full[0,:,1])
plt.plot(pred[0,:,1])

fig, axs = plt.subplots(3)
fig.suptitle('test result')
axs[0].plot(Y_test_full[0,:,0])
axs[1].plot(pred[0,:,0])
axs[2].plot(Y_test_full[0,:,0]-pred[0,:,0])

fig, axs = plt.subplots(3)
fig.suptitle('test result')
axs[0].plot(Y_test_full[0,:,1])
axs[1].plot(pred[0,:,1])
axs[2].plot(Y_test_full[0,:,1]-pred[0,:,1])

