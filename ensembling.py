from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
#from keras import backend as K
import tensorflow.keras.backend as K

from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperModel, HyperParameters, RandomSearch
from kerastuner.tuners import BayesianOptimization

import time
import xlsxwriter

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from data_load_norm import dataNorm, labelsNorm, ilabelsNorm, inputConcat2D, inputConcat1D
from models import newModel

dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'
def testimport():
    global dataset2D, dataset1D, nlabels, w_nlabels

    data2D_import = sio.loadmat(dest_folder + 'dataset_spgram_TEST.mat')
    data1D_import = sio.loadmat(dest_folder + 'dataset_spectra_TEST.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_c_TEST_abs.mat')

    dataset2D = data2D_import['output']
    dataset1D = data1D_import['dataset_spectra']
    labels = labels_import['labels_c']*64.5

    #reshaping
    # dataset2D_rs = np.transpose(dataset2D, (0, 3, 1, 2))
    dataset1D = np.transpose(dataset1D, (0, 2, 1))
    labels = np.transpose(labels,(1,0))

    nlabels, w_nlabels = labelsNorm(labels)

    return dataset2D, dataset1D, nlabels, w_nlabels


testimport()
dataset1D_flat = inputConcat1D(dataset1D)

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"


net_name1 = "ShallowELU_hp2"
checkpoint_path1 = outpath + folder + net_name1 + ".best.hdf5"
model1 = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
model1.load_weights(checkpoint_path1)

net_name2 = "ResNet_fed"
checkpoint_path2 = outpath + folder + net_name2 + ".best.hdf5"
model2 = newModel(dim='1D', type='ResNet', subtype='ResNet_fed')
model2.load_weights(checkpoint_path2)

loss1 = model1.evaluate(dataset2D, nlabels, verbose=2)
loss2 = model2.evaluate(dataset1D_flat, nlabels, verbose=2)

