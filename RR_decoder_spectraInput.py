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
from data_load_norm import dataimport1D, labelsimport, labelsNorm, ilabelsNorm, inputConcat1D, labelsimport_md
from models import newModel

md_input = 1

if md_input == 0:
    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/'
    dataname = 'dataset_spectra.mat'
    X_train, X_val = dataimport1D(folder, dataname, 'dataset_spectra')

    labelsname = 'labels_c.mat'
    y_train, y_val = labelsimport(folder, labelsname, 'labels_c')
    # nX_train_rs = dataNorm(X_train_rs)
    # nX_val_rs = dataNorm(X_val_rs)

    ny_train, w_y_train = labelsNorm(y_train)
    ny_val, w_y_val = labelsNorm(y_val)

    X_train_flat = inputConcat1D(X_train)
    X_val_flat = inputConcat1D(X_val)
else:
    # Martyna's noisy - denoised - GT dataset

    # pred --> output
    # datasetX --> output_noisy
    # labelsY --> output_gt

    folder = 'C:/Users/Rudy/Desktop/toMartyna/toRUDY/'
    # data = np.load(folder + 'X_noisy_data.npy')
    data = np.load(folder + 'GT_data.npy')
    # data = np.load(folder + 'pred_denoised_DL.npy')
    scaling = np.max(data)
    data = data / scaling
    X_train = data[0:17000, :, :]
    X_val = data[17000:19000, :, :]
    X_test = data[19000:20000, :, :]  # unused

    folder = 'C:/Users/Rudy/Desktop/toMartyna/toRUDY/labels/'
    filenames = ['labels_c_1.mat',
                 'labels_c_2.mat',
                 'labels_c_3.mat',
                 'labels_c_4.mat']
    keyname = 'labels_c'

    y_train, y_val, y_test = labelsimport_md(folder, filenames, keyname)
    ny_train, w_y_train = labelsNorm(y_train)
    ny_val, w_y_val = labelsNorm(y_val)



def training():
  times2train = 1
  outpath = 'C:/Users/Rudy/Desktop/DL_models/'
  folder = "net_type/"
  subfolder = 'decoder_1D_denoising_quant/'
  net_name = "Inception-Net-1D2c-v0_trained_on_GT"

  from keras.callbacks import ReduceLROnPlateau

  tf.debugging.set_log_device_placement(True)
  for i in range(times2train):
      model = newModel(dim='1D', type='InceptionNet_1D2c', subtype='v0')
      checkpoint_path = outpath + folder + subfolder + net_name + "_iter_" + str(i) + ".best.hdf5"
      # checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"
      # checkpoint_dir = os.path.dirnamename(checkpoint_path)
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

      # fig = plt.figure(figsize=(10, 10))
      # # summarize history for loss
      # plt.plot(history.history['loss'], label='loss')
      # plt.plot(history.history['val_loss'], label='val_loss')
      # plt.title('model losses')
      # plt.xlabel('epoch')
      # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
      # plt.show()


training()