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
from models_to_basis import newModel
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
from util import textMe


import kerastuner as kt
from kerastuner import HyperModel, HyperParameters, RandomSearch
from kerastuner.tuners import BayesianOptimization

import time

# interval to train metabolites, help in debug or developing phase:
# full range of metabolites is defined in [0, len(metnames)=17] = [0,1,2,3,4,...,16]
met2train_start = 0
met2train_stop = 1

same = 1 #same == 1 means having same network architecture for all metabolites
#NB 01.01.2022: same = 0 is not supported for dualpath_net: dualpath architecture has not yet been optimized
dualpath_net = 1


def dataimport(index):
    global y_train, y_val, y_test, X_train, X_val, X_test

    os.environ["KERAS_BACKEND"] = "theano"
    K.set_image_data_format('channels_last')


    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/'

    data_import = sio.loadmat(dest_folder + 'spectra_kor_wat.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_kor_' + str(index) + '_NOwat.mat')


    dataset = data_import['spectra_kor']
    scaling = np.max(dataset)

    labels = labels_import['labels_kor_' + str(index)]
    scaling_labels = np.max(labels)

    dataset = dataset/scaling
    X_train = dataset[0:18000, :]
    X_val = dataset[18000:20000, :]
    X_test = dataset[19000:20000, :]  # unused

    #normalization try
    labels = labels/scaling_labels
    y_train = labels[0:18000, :]
    y_val = labels[18000:20000, :]
    y_test = labels[19000:20000, :]

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']
order = [8, 10, 1, 11, 12, 2, 13, 14, 15, 4, 16, 9, 5, 17, 3, 6, 7]


for idx in range(met2train_start, met2train_stop ):
    dataimport(order[idx])

    if same:
        if dualpath_net:
            model = newModel(type=1)
            spec = '_dualpath_elu_nonorm'
        else:
            model = newModel(type=0)
            spec = '_220126'
    else:
        model = newModel(met=metnames[idx], type=0)
        spec = '_doc'

    times2train = 1
    output_folder = 'C:/Users/Rudy/Desktop/DL_models/'
    subfolder = "net_type/unet/"
    net_name = "UNet_" + metnames[idx] + spec + "_NOwat"


    for i in range(times2train):
        checkpoint_path = output_folder + subfolder + net_name + ".best.hdf5"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        # selected channel 0 to keep only Re(spectrogram)
        history = model.fit(X_train, y_train,
                              epochs=200,
                              batch_size=50,
                              shuffle=True,
                              validation_data=(X_val, y_val),
                              validation_freq=1,
                              callbacks=[es, mc],
                              verbose=1)

    #textMe('UNet training for ' + metnames[idx] + ' is done')
    fig = plt.figure(figsize=(10, 10))
    # summarize history for loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model losses')
    plt.xlabel('epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()
    print('loss: ' + str(history.history['loss'][-1]))
    print('val_loss:' + str(history.history['val_loss'][-1]))

    pred_train = model.predict(X_train)

    for idplt in [0,10,100]:
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(np.flip(X_train[idplt, :]))
        plt.subplot(3, 1, 2)
        plt.plot(np.flip(y_train[idplt, :]))
        plt.plot(np.flip(pred_train[idplt, :, 0]))
        plt.subplot(3, 1, 3)
        plt.plot(np.flip(np.exp(pred_train[idplt, :, 1])))