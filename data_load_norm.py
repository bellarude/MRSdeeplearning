from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import h5py
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
def dataimport2Dhres(folder, filename, keyname):
    filepath = folder + filename
    data_import = {}
    f = h5py.File(filepath, 'r')
    for k, v in f.items():
        data_import[k] = np.array(v)

    # data_import = sio.loadmat(folder + filename)
    dataset = data_import[keyname]
    dataset = np.transpose(dataset, (3, 2, 1, 0))

    X_train = dataset[0:35000, :, :, :]
    X_val = dataset[35000:40000, :, :, :]

    return X_train, X_val

def dataimport2D(folder, filename, keyname):

    # dataset20: 18000 training, 2000 validation
    # dataset24, 25, 26,27,30: 20000 training, 5000 validation
    # dataset29 and 32: 35 training, 5000 validation

    data_import = sio.loadmat(folder + filename)
    dataset = data_import[keyname]

    X_train = dataset[0:35000, :, :, :]
    X_val   = dataset[35000:40000, :, :, :]

    return X_train, X_val

def dataimport2D_md(folder, filenames, keyname):

    ds = []
    for i in range(len(filenames)):
        data_import = sio.loadmat(folder + filenames[i])
        ds.append(data_import[keyname])

    dataset = np.concatenate((ds[0], ds[1], ds[2], ds[3]), axis=0) * 5500



    X_train = dataset[0:16000, :, :, :]
    X_val = dataset[16000:18000, :, :, :]
    X_test = dataset[18000:20000, :, :, :]  # unused

    return X_train, X_val, X_test

def dataimport1D(folder, filename, keyname):
    data_import = sio.loadmat(folder + filename)
    dataset = data_import[keyname]

    X_train = dataset[0:18000, :]
    X_val = dataset[18000:20000, :]
    # X_test = dataset[19000:20000, :]  # unused

    # reshaping
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    # X_test_rs = np.transpose(X_test, (0, 2, 1))

    return X_train, X_val

def labelsimport(folder, filename, keyname):
    labels_import = sio.loadmat(folder + filename)
    labels = labels_import[keyname] * 64.5
    y_train = labels[0:35000, :]
    y_val = labels[35000:40000, :]
    # y_test = labels[18000:20000, :]

    return y_train, y_val

def labelsimport_md(folder, filenames, keyname):
    ls = []
    for i in range(len(filenames)):
        labels_import = sio.loadmat(folder + filenames[i])

        ll = labels_import[keyname]
        # check up if Martyna gives me only 16 labels (water missing)
        if ll.shape[0] < 17:
            ll = np.concatenate((ll, np.ones((1, ll.shape[1]))), axis=0)

        ls.append(np.transpose(ll*64.5, (1,0)))

    labels = np.concatenate((ls[0], ls[1], ls[2], ls[3]), axis=0)

    y_train = labels[0:16000, :]
    y_val = labels[16000:18000, :]
    y_test = labels[18000:20000, :]

    return y_train, y_val, y_test

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
    weights = np.empty([labels.shape[1], 1])
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
