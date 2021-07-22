from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
# import matplotlib.pyplot as plt
# import h5py
# from keras.models import Model, load_model, Sequential
# from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
# #from keras import backend as K
import tensorflow.keras.backend as K

# from keras import layers
# from keras.callbacks import ModelCheckpoint, EarlyStopping
#
# import tensorflow as tf
# import kerastuner as kt
# from kerastuner import HyperModel, HyperParameters, RandomSearch
# from kerastuner.tuners import BayesianOptimization
#
# import time
# import xlsxwriter
#
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score


def dataimport2D(folder, filename):
    data_import = sio.loadmat(folder + filename)
    dataset = data_import['dataset']

    X_train = dataset[0:18000, :, :, :]
    X_val   = dataset[18000:20000, :, :, :]
    # X_test  = dataset[18000:20000, :, :, :]

    return X_train, X_val

def dataimport1D(folder, filename):
    data_import = sio.loadmat(folder + filename)
    dataset = data_import['dataset_spectra']

    X_train = dataset[0:18000, :]
    X_val = dataset[18000:20000, :]
    # X_test = dataset[19000:20000, :]  # unused

    # reshaping
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    # X_test_rs = np.transpose(X_test, (0, 2, 1))

    return X_train, X_val

def labelsimport(folder, filename):
    labels_import = sio.loadmat(folder + filename)
    labels = labels_import['labels_c'] * 64.5
    y_train = labels[0:18000, :]
    y_val = labels[18000:20000, :]
    # y_test = labels[18000:20000, :]

    return y_train, y_val

def dataNorm(dataset):
    dataset_norm = np.empty(dataset.shape)
    for i in range(dataset.shape[0]):
      # M = np.amax(np.abs(dataset[i,:,:,:]))
      M = 2500
      dataset_norm[i,0,:,:] = dataset[i,0,:,:] / M
      dataset_norm[i,1,:,:] = dataset[i,1,:,:] / M
    return dataset_norm


def labelsNorm(labels):
    labels_norm = np.empty(labels.shape)
    weights = np.empty([labels.shape[0], 1])
    for i in range(labels.shape[1]):
        w = np.amax(labels[:, i])
        labels_norm[:, i] = labels[:, i] / w
        weights[i] = w

    return labels_norm, weights

def ilabelsNorm(labels_norm, weights):
    ilabels = np.empty(labels_norm.shape)
    for i in range(labels_norm.shape[1]):
        ilabels[:, i] = labels_norm[:, i] * weights[i]

    return ilabels

def inputConcat2D(input):
  i_concat = np.empty((input.shape[0], input.shape[1], input.shape[2]*2, 1))
  i_real = input[:,:,:,0]
  i_imag = input[:,:,:,1]
  i_concat[:,:,:,0] = np.concatenate((i_real, i_imag), axis=2)

  return i_concat


def inputConcat1D(dataset1D):
    cinput = np.zeros([dataset1D.shape[0], dataset1D.shape[1]*2, 1])
    cinput[:, :, 0] = np.concatenate((dataset1D[:, :, 0], dataset1D[:, :, 1]), axis=1)

    return cinput
