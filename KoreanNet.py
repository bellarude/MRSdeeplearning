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

    data_import = sio.loadmat(dest_folder + 'spectra_kor.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_kor_2.mat')


    dataset = data_import['spectra_kor']
    labels = labels_import['labels_kor_2']


    X_train = dataset[0:18000, :]
    X_val = dataset[18000:20000, :]
    X_test = dataset[19000:20000, :]  # unused

    y_train = labels[0:18000, :]
    y_val = labels[18000:20000, :]
    y_test = labels[19000:20000, :]


def dataNormKoreanSet(dataset, labels):
    dataset_norm = np.empty(dataset.shape)
    labels_norm = np.empty(labels.shape)
    # M = np.amax(np.abs(dataset[:]))
    M = 25000
    for i in range(dataset.shape[0]):
        # M = np.amax(np.abs(dataset[i, :]))

        dataset_norm[i, :] = dataset[i, :] / M
        labels_norm[i, :] = labels[i,:] / M
    return dataset_norm, labels_norm

def dataHighlight(labels, factor):
    # dataset_h = dataset
    labels_h = labels
    for i in range(labels.shape[0]):
        # M = np.amax(np.abs(dataset[i, :]))

        # dataset_h[i, 250:1406] = dataset[i, 250:1406] * factor
        labels_h[i, 250:1406] = labels[i, 250:1406] * factor
    return labels_h

# input image dimensions
ref2tCr = 1
if ref2tCr:
    datapoints = 1308
else:
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

kornet = 0
unet = 1

if kornet:
    naa_k = [199, 78, 8, 65, 8, 5, 4, 5, 3]
    naa_f = [9, 36, 46, 77, 25, 60, 42, 30, 85]
    naa_d = 5910
    naa_lr = 0.005944

    gaba_k = [33, 24, 17, 57, 19, 6, 6, 3, 3]
    gaba_f = [49, 33, 66, 22, 16, 50, 51, 97, 86]
    gaba_d = 6476
    gaba_lr = 0.007486

    scy_k = [30, 297, 116, 39, 13, 15, 7, 4, 3]
    scy_f = [19, 50, 12, 78, 57, 25, 14, 20, 80]
    scy_d = 6000
    scy_lr = 0.006479

    ks = {'naa': naa_k, 'gaba': gaba_k, 'scy': scy_k}  # kernel size
    nf = {'naa': naa_f, 'gaba': gaba_f, 'scy': scy_f}  # number of filters
    nd = {'naa': naa_d, 'gaba': gaba_d, 'scy': scy_d}  # neuron dense layer
    lr = {'naa': naa_lr, 'gaba': gaba_lr, 'scy': scy_lr}  # learning rate

    met = 'gaba'

    # -----------------------------------------------------------------------------
    # Korean Decoder NET
    # -----------------------------------------------------------------------------
    l0 = maxP(convBlock2(inputs, ks[met][0], nf[met][0]), **poolargs)
    l1 = maxP(convBlock2(l0, ks[met][1], nf[met][1]), **poolargs)
    l2 = maxP(convBlock2(l1, ks[met][2], nf[met][2]), **poolargs)
    l3 = maxP(convBlock2(l2, ks[met][3], nf[met][3]), **poolargs)
    l4 = maxP(convBlock2(l3, ks[met][4], nf[met][4]), **poolargs)
    l5 = maxP(convBlock2(l4, ks[met][5], nf[met][5]), **poolargs)
    l6 = maxP(convBlock2(l5, ks[met][6], nf[met][6]), **poolargs)
    l7 = maxP(convBlock2(l6, ks[met][7], nf[met][7]), **poolargs)
    l8 = maxP(convBlock2(l7, ks[met][8], nf[met][8]), **poolargs)
    l9 = relu(dense(nd[met], flatten(l8)))
    outputs = dense(datapoints, l9)
    lrate = lr[met]

    # -----------------------------------------------------------------------------
    # Korean Decoder NET - linear
    # -----------------------------------------------------------------------------
    # l0 = maxP(convBlock2_lin(inputs, ks[met][0], nf[met][0]), **poolargs)
    # l1 = maxP(convBlock2_lin(l0, ks[met][1], nf[met][1]), **poolargs)
    # l2 = maxP(convBlock2_lin(l1, ks[met][2], nf[met][2]), **poolargs)
    # l3 = maxP(convBlock2_lin(l2, ks[met][3], nf[met][3]), **poolargs)
    # l4 = maxP(convBlock2_lin(l3, ks[met][4], nf[met][4]), **poolargs)
    # l5 = maxP(convBlock2_lin(l4, ks[met][5], nf[met][5]), **poolargs)
    # l6 = maxP(convBlock2_lin(l5, ks[met][6], nf[met][6]), **poolargs)
    # l7 = maxP(convBlock2_lin(l6, ks[met][7], nf[met][7]), **poolargs)
    # l8 = maxP(convBlock2_lin(l7, ks[met][8], nf[met][8]), **poolargs)
    # l9 = dense(nd[met], flatten(l8))
    # outputs = dense(datapoints, l9)
    # lrate = lr[met]

    # -----------------------------------------------------------------------------

if unet:

    if ref2tCr:
        pad = 2
    else:
        pad = 9
    # -----------------------------------------------------------------------------
    # RR-Unet
    # -----------------------------------------------------------------------------
    # --- Define contracting layers

    # l1 = conv1(32, tf.keras.layers.ZeroPadding1D(padding=(pad))(inputs))
    # l2 = conv1(64, conv2(32, l1))
    # l3 = conv1(128, conv2(48, l2))
    # l4 = conv1(256, conv2(64, l3))
    # l5 = conv1(512, conv2(80, l4))
    #
    # # --- Define expanding layers
    # l6 = tran2(256, l5)
    #
    # # --- Define expanding layers
    # l7 = tran2(128, tran1(64, concat(l4, l6)))
    # l8 = tran2(64, tran1(48, concat(l3, l7)))
    # l9 = tran2(32,  tran1(32, concat(l2, l8)))
    # l10 = conv1(32, l9)
    #
    # # --- Create logits
    # outputs = tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=1))
    # lrate = 1e-3

    # -----------------------------------------------------------------------------
    # RR-Unet 2xconv1
    # -----------------------------------------------------------------------------
    # --- Define contracting layers

    l1 = conv1(64, conv1(32, tf.keras.layers.ZeroPadding1D(padding=(pad))(inputs)))
    l2 = conv1(128, conv1(64, conv2(32, l1)))
    l3 = conv1(256, conv1(128, conv2(48, l2)))
    l4 = conv1(512, conv1(256, conv2(64, l3)))
    l5 = conv1(256, conv1(512, conv2(80, l4)))

    # --- Define expanding layers
    l6 = tran2(256, l5)

    # --- Define expanding layers
    l7 = tran2(128, tran1(64, tran1(64, concat(l4, l6))))
    l8 = tran2(64, tran1(48, tran1(48, concat(l3, l7))))
    l9 = tran2(32, tran1(32, tran1(32, concat(l2, l8))))
    l10 = conv1(32, conv1(32, l9))

    # --- Create logits
    outputs = tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=1))
    lrate = 1e-3

# --- Create model
modelRR = Model(inputs=inputs, outputs=outputs)

# --- Compile model
modelRR.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
    loss = tf.keras.losses.MeanSquaredError(),
    #experimental_run_tf_function=False
    )

print(modelRR.summary())

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

times2train = 1
output_folder = 'C:/Users/Rudy/Desktop/DL_models/'
subfolder = "net_type/"
net_name = "UNet_gaba_conv2x"

fig = plt.figure()
plt.plot(ny_val[0,:])
plt.show()

for i in range(times2train):
    checkpoint_path = output_folder + subfolder + net_name + ".best.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                         save_weights_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # selected channel 0 to keep only Re(spectrogram)
    history = modelRR.fit(nX_train, ny_train,
                          epochs=100,
                          batch_size=50,
                          shuffle=True,
                          validation_data=(nX_val, ny_val),
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
    # print('loss: ' + str(history.history['loss'][-1]))
    # print('val_loss:' + str(history.history['val_loss'][-1]))