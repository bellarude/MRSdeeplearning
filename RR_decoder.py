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
import pickle
from util import textMe
import xlsxwriter

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from data_load_norm import dataimport2D, labelsimport, labelsNorm, ilabelsNorm, inputConcat2D, dataimport2D_md, labelsimport_md, dataimport2Dhres
from models import newModel

md_input = 0
flat_input = 0
resize_input = 0
hres = 0

if md_input == 0:
    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'
    dataname = 'dataset_spgram.mat'
    labelsname = 'labels_c.mat'

    if hres:
        X_train, X_val = dataimport2Dhres(folder, dataname, 'dataset')
        y_train, y_val = labelsimporthres(folder, labelsname, 'labels_c')
    else:
        X_train, X_val = dataimport2D(folder, dataname, 'dataset')
        y_train, y_val = labelsimport(folder, labelsname, 'labels_c')



    # nX_train_rs = dataNorm(X_train_rs)
    # nX_val_rs = dataNorm(X_val_rs)

else:
    #Martyna's noisy - denoised - GT dataset

    # pred --> output
    # datasetX --> output_noisy
    # labelsY --> output_gt

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33/'
    filenames = ['zoomedSpgram_datasetX_1.mat',
                 'zoomedSpgram_datasetX_2.mat',
                 'zoomedSpgram_datasetX_3.mat',
                 'zoomedSpgram_datasetX_4.mat']
    keyname = 'output_noisy'

    X_train, X_val, X_test = dataimport2D_md(folder, filenames, keyname)

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33/labels/'
    filenames = ['labels_c_1.mat',
                 'labels_c_2.mat',
                 'labels_c_3.mat',
                 'labels_c_4.mat']
    keyname = 'labels_c'

    y_train, y_val, y_test = labelsimport_md(folder, filenames, keyname)

ny_train, w_y_train = labelsNorm(y_train)
ny_val, w_y_val = labelsNorm(y_val)

#RR test only 17.02.2022
ny_train = y_train
ny_val = y_val

if flat_input:
    X_train = inputConcat2D(X_train)
    X_val = inputConcat2D(X_val)

if resize_input:
    X_train = tf.image.resize(X_train, (224, 224))
    X_val = tf.image.resize(X_val, (224, 224))


times2train = 1
outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"
subfolder = ""
net_name = "ShallowNet-2D2c-hp-Original_labels"

if times2train == 1:

    model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
    # checkpoint_path = "/content/drive/My Drive/RR/nets models/waterNOwater/RRdecoder_ESMRMB1_d31_" + str(i) + ".best.hdf5"
    checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"

    # model.load_weights(checkpoint_path)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                         save_weights_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Reduce learning rate when a metric has stopped improving
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # selected channel 0 to keep only Re(spectrogram)
    history = model.fit(X_train, ny_train,
                        epochs=200,
                        batch_size=50,
                        shuffle=True,
                        validation_data=(X_val, ny_val),
                        validation_freq=1,
                        callbacks=[es, mc],
                        verbose=1)

    fig = plt.figure(figsize=(10, 10))
    # summarize history for loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model losses')
    plt.xlabel('epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


    history_filename = outpath + folder + subfolder + net_name + "_history"
    open_file = open(history_filename, "wb")
    pickle.dump(history.history, open_file)
    open_file.close()

else: #times2train > 1 automatic training + email message

    def training():
        from keras.callbacks import ReduceLROnPlateau
        tf.debugging.set_log_device_placement(True)
        for i in range(times2train):
            start = time.time()

            model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
            # checkpoint_path = "/content/drive/My Drive/RR/nets models/waterNOwater/RRdecoder_ESMRMB1_d31_" + str(i) + ".best.hdf5"
            checkpoint_path = outpath + folder + subfolder + net_name + "-" + str(i) + ".best.hdf5"

            # model.load_weights(checkpoint_path)
            checkpoint_dir = os.path.dirname(checkpoint_path)
            mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='min')
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

            # Reduce learning rate when a metric has stopped improving
            # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

            # selected channel 0 to keep only Re(spectrogram)
            history = model.fit(X_train, ny_train,
                                  epochs=200,
                                  batch_size=50,
                                  shuffle=True,
                                  validation_data=(X_val, ny_val),
                                  validation_freq=1,
                                  callbacks=[es, mc],
                                  verbose=1)

            fig = plt.figure(figsize=(10, 10))
            # summarize history for loss
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.title('model losses')
            plt.xlabel('epoch')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            end = time.time()
            elapsedtime = (end - start) / 3600  # in hours

            # from util import save_pickle
            # history_filename = outpath + folder + subfolder + net_name + "_history"
            # save_pickle(history_filename, history)
            textMe(str(i) + '. training DONE, time -> ' + '{0:.2f}'.format(elapsedtime) + 'h, val loss->' + '{0:.5f}'.format(history.history['val_loss'][-1]) + ', epochs->' + '{0:.0f}'.format(len(history.epoch)))

            # otherwise times2train it stops the loop ove
            if times2train == 1:
                plt.show()


    training()
