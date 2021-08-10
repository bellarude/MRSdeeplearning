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


def dataimport():
    global y_train, y_val, y_test, X_train, X_val, X_test

    os.environ["KERAS_BACKEND"] = "theano"
    K.set_image_data_format('channels_last')


    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/'

    data_import = sio.loadmat(dest_folder + 'spectra_kor_wat.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_kor_5_NOwat.mat')

    dataset = data_import['spectra_kor']
    labels = labels_import['labels_kor_5']

    X_train = dataset[0:18000, :]
    X_val = dataset[18000:20000, :]
    X_test = dataset[19000:20000, :]  # unused

    y_train = labels[0:18000, :]
    y_val = labels[18000:20000, :]
    y_test = labels[19000:20000, :]

datapoints = 1406
channels = 1 #number of channels
input_shape = (datapoints, channels)
inputs = Input(shape=input_shape)
channel_axis = 1 if K.image_data_format() == 'channels_first' else 2

# --- Define kwargs dictionary
kwargs = {
    'strides': (1),
    'padding': 'same'}

# --- Define poolargs dictionary
poolargs = {
    'pool_size': (2),
    'strides': (2)}

# -----------------------------------------------------------------------------
# Define lambda functions
# -----------------------------------------------------------------------------

conv = lambda x, kernel_size, filters : layers.Conv1D(filters=filters, kernel_size=kernel_size, **kwargs)(x)
conv_s = lambda x, strides, filters : layers.Conv1D(filters=filters, kernel_size=3, strides=strides, padding='same')(x)
# --- Define stride-1, stride-2 blocks
conv1 = lambda filters, x : relu(norm(conv_s(x, filters=filters, strides=1)))
conv2 = lambda filters, x : relu(norm(conv_s(x, filters=filters, strides=2)))
# --- Define single transpose
tran = lambda x, filters, strides : layers.Conv1DTranspose(filters=filters, strides=strides, kernel_size=3, padding='same')(x)
# --- Define transpose block
tran1 = lambda filters, x : relu(norm(tran(x, filters, strides=1)))
tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=2)))

norm = lambda x : layers.BatchNormalization(axis=channel_axis)(x)
normD = lambda x : layers.BatchNormalization(axis=1)(x)
relu = lambda x : layers.ReLU()(x)
maxP = lambda x, pool_size, strides : layers.MaxPooling1D(pool_size=pool_size, strides=strides)(x)

flatten = lambda x : layers.Flatten()(x)
dense = lambda units, x : layers.Dense(units=units)(x)

convBlock = lambda x, kernel_size, filters : relu(norm(conv(x, kernel_size, filters)))
convBlock2 = lambda x, kernel_size, filters : convBlock(convBlock(x, kernel_size, filters), kernel_size, filters)

convBlock_lin = lambda x, kernel_size, filters : norm(conv(x, kernel_size, filters))
convBlock2_lin = lambda x, kernel_size, filters : convBlock(convBlock(x, kernel_size, filters), kernel_size, filters)

concat = lambda a, b : layers.Concatenate(axis=channel_axis)([a, b])

def concatntimes(x, n):
  output = concat(x, x)
  for i in range(n-1):
    output = concat(output, output)
  return output

add = lambda x, y: layers.Add()([x, y])
ResidualBlock = lambda x, y: relu(add(x,y))
dropout = lambda x, percentage, size : layers.Dropout(percentage, size)(x)


hp = HyperParameters()
def build_model(hp):
    pad = 9
    # -----------------------------------------------------------------------------
    # RR-Unet 2xconv1
    # -----------------------------------------------------------------------------
    # --- Define contracting layers

    units1 = hp.Int('units1', min_value=10, max_value=100, step=10, default=30)
    units2 = hp.Int('units2', min_value=10, max_value=100, step=10, default=30)
    units3 = hp.Int('units3', min_value=20, max_value=200, step=10, default=60)
    units4 = hp.Int('units3', min_value=20, max_value=200, step=10, default=60)
    units5 = hp.Int('units5', min_value=40, max_value=400, step=30, default=120)
    units6 = hp.Int('units6', min_value=40, max_value=400, step=30, default=120)
    units7 = hp.Int('units7', min_value=80, max_value=800, step=50, default=240)
    units8 = hp.Int('units8', min_value=80, max_value=800, step=50, default=240)
    units9 = hp.Int('units9', min_value=160, max_value=1600, step=80, default=480)

    lrate = hp.Float('lrate', min_value=2e-6, max_value=2e-2, default=1e-3)


    l1 = conv1(units1*2, conv1(units1, tf.keras.layers.ZeroPadding1D(padding=(pad))(inputs)))
    l2 = conv1(units3*2, conv1(units3, conv2(units2, l1)))
    l3 = conv1(units5*2, conv1(units5, conv2(units4, l2)))
    l4 = conv1(units7*2, conv1(units7, conv2(units6, l3)))
    l5 = conv1(units9*2, conv1(units9, conv2(units8, l4)))

    # --- Define expanding layers
    l6 = tran2(units8, l5)

    # --- Define expanding layers
    l7 = tran2(units6, tran1(units7, tran1(units7*2, concat(l4, l6))))
    l8 = tran2(units4, tran1(units5, tran1(units5*2, concat(l3, l7))))
    l9 = tran2(units2, tran1(units3, tran1(units3*2, concat(l2, l8))))
    l10 = conv1(units1, conv1(units1*2, l9))

    # --- Create logits
    outputs = tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=1))

    # --- Create model
    modelRR = Model(inputs=inputs, outputs=outputs)

    # --- Compile model
    modelRR.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
        loss = tf.keras.losses.MeanSquaredError(),
        experimental_run_tf_function=False
        )
    return modelRR
# print(modelRR.summary())

dataimport()

# RR210714: here normalization brings up problems (see notes)
# nX_train, ny_train = dataNormKoreanSet(X_train, y_train)
# nX_val, ny_val = dataNormKoreanSet(X_val, y_val)

nX_train = X_train
nX_val = X_val
ny_train = y_train
ny_val = y_val

# ny_train = dataHighlight(y_train, 10)
# ny_val = dataHighlight(y_val, 10)

outpath = 'C:/Users/Rudy/Desktop/Dl_models/'
tuner = BayesianOptimization(build_model,
                          objective = 'val_loss', #what u want to track
                          max_trials = 50, #how many randoms picking do we want to have
                          executions_per_trial=1, #number of time you train each dynamic version (see details below)
                          directory=outpath+'BayesianSearch/',
                          project_name= f"project_{int(time.time())}")

tf.debugging.set_log_device_placement(True)
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
tuner.search(nX_train, ny_train,
            epochs=100,
            batch_size=100,
            shuffle=True,
            validation_data=(nX_val, ny_val),
            validation_freq=1,
            callbacks=[es]
             )

# When search is over, you can retrieve the best model(s):
#print("------------------------")
#models = tuner.get_best_models(num_models=2)
#print("------------------------")
# Or print a summary of the results:
#tuner.results_summary()

print("------------------------")
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

#
# times2train = 1
# output_folder = 'C:/Users/Rudy/Desktop/DL_models/'
# subfolder = "net_type/"
# net_name = "UNet_mI_NOwat"
#
# fig = plt.figure()
# plt.plot(ny_val[0,:])
# plt.show()
#
# for i in range(times2train):
#     checkpoint_path = output_folder + subfolder + net_name + ".best.hdf5"
#     checkpoint_dir = os.path.dirname(checkpoint_path)
#     mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
#                          save_weights_only=True, mode='min')
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#
#     # selected channel 0 to keep only Re(spectrogram)
#     history = modelRR.fit(nX_train, ny_train,
#                           epochs=100,
#                           batch_size=50,
#                           shuffle=True,
#                           validation_data=(nX_val, ny_val),
#                           validation_freq=1,
#                           callbacks=[es, mc],
#                           verbose=1)
#
#     fig = plt.figure(figsize=(10, 10))
#     # summarize history for loss
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.title('model losses')
#     plt.xlabel('epoch')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#     plt.show()
#     # print('loss: ' + str(history.history['loss'][-1]))
#     # print('val_loss:' + str(history.history['val_loss'][-1]))

import smtplib

def textMe(string):
    # needs to abilitate less secure app: https://www.google.com/settings/security/lesssecureapps
    content = (string)
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.ehlo()
    mail.starttls()
    address = 'rudy.rizzo.tv@gmail.com'
    mail.login('amsmdeepmrs@gmail.com', 'amsmdeepmrs20')
    mail.sendmail('amsmdeepmrs@gmail.com', address, content)
    mail.close()
    print(">>> sent E-mail @" + address)

textMe('UNet hp is done')