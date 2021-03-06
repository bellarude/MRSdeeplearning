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
from data_load_norm import dataimport1D, labelsimport, labelsNorm, ilabelsNorm, inputConcat1D
from models import newModel


folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'
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

def training():
  times2train = 1
  outpath = 'C:/Users/Rudy/Desktop/DL_models/'
  folder = "net_type/"
  subfolder = 'typology/'
  net_name = "InceptionNet1D"

  from keras.callbacks import ReduceLROnPlateau

  tf.debugging.set_log_device_placement(True)
  for i in range(times2train):
      model = newModel(dim='1D', type='InceptionNet', subtype='InceptionNet-1D')
      # checkpoint_path = outpath + folder + subfolder + net_name + str(i+3) + ".best.hdf5"
      checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"
      # checkpoint_dir = os.path.dirnamename(checkpoint_path)
      mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                           save_weights_only=True, mode='min')
      es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

      # Reduce learning rate when a metric has stopped improving
      # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

      # selected channel 0 to keep only Re(spectrogram)
      history = model.fit(X_train_flat, ny_train,
                          epochs=200,
                          batch_size=50,
                          shuffle=True,
                          validation_data=(X_val_flat, ny_val),
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
      plt.show()


training()