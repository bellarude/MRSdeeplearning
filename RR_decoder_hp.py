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
    global y_train, y_val, y_test, X_train_rs, X_val_rs, X_test_rs

    os.environ["KERAS_BACKEND"] = "theano"
    K.set_image_data_format('channels_first')

    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'

    data_import   = sio.loadmat(dest_folder + 'dataset_spgram.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_c_abs.mat')

    dataset = data_import['dataset']
    labels  = labels_import['labels_c']*64.5

    X_train = dataset[0:18000, :, :, :]
    X_val   = dataset[18000:20000, :, :, :]
    # X_test  = dataset[18000:20000, :, :, :]

    y_train = labels[0:18000, :]
    y_val   = labels[18000:20000, :]
    # y_test  = labels[18000:20000, :]

    #reshaping
    X_train_rs = np.transpose(X_train, (0, 3, 1, 2))
    X_val_rs = np.transpose(X_val, (0, 3, 1, 2))
    # X_test_rs = np.transpose(X_test, (0, 3, 1, 2))


def dataNorm(dataset):
    dataset_norm = np.empty(dataset.shape)
    for i in range(dataset.shape[0]):
        M = np.amax(np.abs(dataset[i, :, :, :]))
        dataset_norm[i, 0, :, :] = dataset[i, 0, :, :] / M
        dataset_norm[i, 1, :, :] = dataset[i, 1, :, :] / M
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

def inputConcat(input_rs):
  i_conc = np.empty((input_rs.shape[0], 1, input_rs.shape[2], input_rs.shape[3]*2))
  i_real = input_rs[:,0,:,:]
  i_imag = input_rs[:,1,:,:]
  i_conc[:,0,:,:] = np.concatenate((i_real,i_imag),axis=2)

  return i_conc

dataimport()
nX_train_rs = dataNorm(X_train_rs)
nX_val_rs = dataNorm(X_val_rs)
# nX_test_rs = dataNorm(X_test_rs)

ny_train, w_y_train = labelsNorm(y_train)
ny_val, w_y_val = labelsNorm(y_val)
# ny_test, w_y_test = labelsNorm(y_test)


# --- Define kwargs dictionary
kwargs = {
    'kernel_size': (3, 3),  # was 2,2 before
    'padding': 'same'}

# --- Define poolargs dictionary
poolargs = {
    'pool_size': (2, 2),
    'strides': (2, 2)}

# --- Define lambda functions
conv = lambda x, filters, strides: layers.Conv2D(filters=filters, strides=strides, **kwargs)(x)
norm = lambda x: layers.BatchNormalization(axis=1)(x)
relu = lambda x: layers.ReLU()(x)
lin = lambda x: tf.keras.activations.linear(x)
elu = lambda x: tf.keras.activations.elu(x)
tanh = lambda x: tf.keras.activations.tanh(x)
maxP = lambda x, pool_size, strides: layers.MaxPooling2D(pool_size=pool_size, strides=strides)(x)
# ConvPool = lambda filters, x : conv(x, filters, strides=2)
add = lambda x, y: layers.Add()([x, y])

conv_ks = lambda x, filters, strides, kernel_size: layers.Conv2D(filters=filters, strides=strides,
                                                                 kernel_size=kernel_size, padding='same')(x)

flatten = lambda x: layers.Flatten()(x)
dense = lambda units, x: layers.Dense(units=units)(x)
maxP1D = lambda x: tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="same",
                                                data_format="channels_first")(x)

convBlock = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
convBlock2 = lambda filters, x: convBlock(filters, convBlock(filters, x))
# convBlock4 = lambda filters, x : convBlock2(filters, convBlock2(filters, x))

convBlock_ks = lambda filters, ks, x: relu(norm(conv_ks(x, filters, strides=1, kernel_size=ks)))
convBlock_ks_lin = lambda filters, ks, x: lin(norm(conv_ks(x, filters, strides=1, kernel_size=ks)))
convBlock_ks_elu = lambda filters, ks, x: elu(conv_ks(x, filters, strides=1, kernel_size=ks))
convBlock_2x_ks_elu = lambda filters, ks, x: convBlock_ks_elu(filters, ks, convBlock_ks_elu(filters, ks, x))
convBlock_ks_tanh = lambda filters, ks, x: tanh(norm(conv_ks(x, filters, strides=1, kernel_size=ks)))

# --- Concatenate
concat = lambda a, b: layers.Concatenate(axis=1)([a, b])

# --- Dropout
sDrop = lambda x, rate: layers.SpatialDropout2D(rate, data_format='channels_first')(x)
drop = lambda x, rate: layers.Dropout(rate, noise_shape=None, seed=None)(x)

# Federico's blocks
bottleneck = lambda filters_input, filters_output, kernel_size, x: norm(conv_ks(relu(norm(
    conv_ks(relu(norm(conv_ks(x, filters_input, strides=1, kernel_size=1))), filters_input, strides=1,
            kernel_size=kernel_size))), filters_output, strides=1, kernel_size=1))
ResConvBlock = lambda filters_input, filters_output, kernel_size, x: relu(
    add(bottleneck(filters_input, filters_output, kernel_size, x),
        norm(conv_ks(x, filters_output, strides=1, kernel_size=1))))
IdeConvBlock = lambda filters_input, filters_output, kernel_size, x: relu(
    add(bottleneck(filters_input, filters_output, kernel_size, x), norm(x)))
ResidualBlock = lambda filters_input, filters_output, kernel_size, x: maxP(
    IdeConvBlock(filters_input, filters_output, kernel_size[2], maxP(
        IdeConvBlock(filters_input, filters_output, kernel_size[1],
                     maxP(ResConvBlock(filters_input, filters_output, kernel_size[0], x), **poolargs)),
        **poolargs)), **poolargs)

# ResNet paper's block
# NB2: Residual Blocks from Federico's design differ from Residual Blocks on the
# ResNet original paper.

conv2id = lambda filters, x: relu(add(norm(conv(convBlock(filters, x), filters, strides=1)), norm(x)))
conv2conv = lambda filters, x: relu(
    add(norm(conv(convBlock(filters, x), filters, strides=1)), norm(conv(x, filters, strides=1))))
DRB1 = lambda filters, x: conv2id(filters, conv2id(filters, conv2id(filters, x)))



# reference for tuning
# https://www.youtube.com/watch?v=vvC15l4CY1Q&ab_channel=sentdex

hp = HyperParameters()
def build_model(hp):
    shallowCNN = 1
    deepCNN = 0
    ResNet = 0

    if shallowCNN:
        img_rows, img_cols = 128, 32
        channels = 2
        input_shape = (channels, img_rows, img_cols)
        inputs = Input(shape=input_shape)

        # -----------------------------------------------------------------
        # ShallowRELU
        # -----------------------------------------------------------------
        # l1 = maxP(convBlock_ks(50, (11, 11), inputs), **poolargs)
        # l2 = sDrop(maxP(convBlock_ks(100, (9, 9), l1), **poolargs), 0.1)
        # l3 = sDrop(maxP(convBlock_ks(150, (5, 5), l2), **poolargs), 0.15)
        # outputs = lin(norm(dense(17, flatten(l3))))

        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # ShallowRELU_ks3
        # -----------------------------------------------------------------
        # l1 = maxP(convBlock_ks(50, (3, 3), inputs), **poolargs)
        # l2 = sDrop(maxP(convBlock_ks(100, (3, 3), l1), **poolargs), 0.1)
        # l3 = sDrop(maxP(convBlock_ks(150, (3, 3), l2), **poolargs), 0.15)
        # outputs = lin(norm(dense(17, flatten(l3))))
        # -----------------------------------------------------------------

        # l1 = ConvPool(convBlock_ks(50, (11,11), inputs), 25)
        # l2 = sDrop(ConvPool(convBlock_ks(100, (9,9), l1), 50), 0.1)
        # l3 = sDrop(ConvPool(convBlock_ks(150, (5,5), l2), 75), 0.15)

        # -----------------------------------------------------------------
        # shallow ELU-CNN 2xConv
        # -----------------------------------------------------------------
        units1 = hp.Int('units1', min_value=20, max_value=400, step=10, default=50)
        units2 = hp.Int('units2', min_value=40, max_value=400, step=10, default=100)
        units3 = hp.Int('units3', min_value=80, max_value=400, step=10, default=150)

        ksize1 = hp.Int('ks1', min_value=3, max_value=11, step=2, default=9)
        ksize2 = hp.Int('ks2', min_value=3, max_value=11, step=2, default=5)
        ksize3 = hp.Int('ks3', min_value=3, max_value=11, step=2, default=3)

        drop1 = hp.Float('drop1', min_value=0.05, max_value=0.5, step=0.05, default=0.1)
        drop2 = hp.Float('drop2', min_value=0.05, max_value=0.5, step=0.05, default=0.15)

        lrate = hp.Float('lrate', min_value=2e-6, max_value=2e-2, default=2e-4)

        input_shape = (channels, img_rows, img_cols)
        inputs = Input(shape=input_shape)


        l1 = maxP(convBlock_2x_ks_elu(units1, (ksize1,ksize1), inputs), **poolargs)
        l2 = sDrop(maxP(convBlock_2x_ks_elu(units2, (ksize2,ksize2), l1), **poolargs), drop1)
        l3 = sDrop(maxP(convBlock_2x_ks_elu(units3, (ksize3,ksize3), l2), **poolargs), drop2)
        outputs = lin(norm(dense(17, flatten(l3))))
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # shallow ELU-CNN from ESMRMB abstract
        # -----------------------------------------------------------------

        # units1 = hp.Int('units1', min_value=20, max_value=400, step=10, default=50)
        # units2 = hp.Int('units2', min_value=40, max_value=400, step=10, default=100)
        # units3 = hp.Int('units3', min_value=80, max_value=400, step=10, default=150)
        #
        # ksize1 = hp.Int('ks1', min_value=3, max_value=11, step=2, default=9)
        # ksize2 = hp.Int('ks2', min_value=3, max_value=11, step=2, default=5)
        # ksize3 = hp.Int('ks3', min_value=3, max_value=11, step=2, default=3)
        #
        # drop1 = hp.Float('drop1', min_value=0.05, max_value=0.5, step=0.05, default=0.1)
        # drop2 = hp.Float('drop2', min_value=0.05, max_value=0.5, step=0.05, default=0.15)
        #
        # lrate = hp.Float('lrate', min_value=2e-6, max_value=2e-2, default=2e-4)
        #
        # input_shape = (channels, img_rows, img_cols)
        # inputs = Input(shape=input_shape)
        #
        # l1 = maxP(convBlock_ks_elu(units1, (ksize1, ksize1), inputs), **poolargs)
        # l2 = sDrop(maxP(convBlock_ks_elu(units2, (ksize2, ksize2), l1), **poolargs), drop1)
        # l3 = sDrop(maxP(convBlock_ks_elu(units3, (ksize3, ksize3), l2), **poolargs), drop2)
        # outputs = lin(norm(dense(17, flatten(l3))))

        # -----------------------------------------------------------------

    if deepCNN:
        img_rows, img_cols = 128, 64
        channels = 1
        input_shape = (channels, img_rows, img_cols)
        inputs = Input(shape=input_shape)

        # -----------------------------------------------------------------
        # deepCNN_2D_ks
        # -----------------------------------------------------------------
        l1 = maxP(convBlock_ks(8, (13, 13), inputs), **poolargs)
        l2 = maxP(convBlock_ks(16, (11, 11), l1), **poolargs)
        l3 = maxP(convBlock_ks(32, (9, 9), l2), **poolargs)
        l4 = maxP(convBlock_ks(64, (7, 7), l3), **poolargs)
        l5 = maxP(convBlock_ks(128, (5, 5), l4), **poolargs)
        l6 = maxP(convBlock_ks(256, (3, 3), l5), **poolargs)
        l7 = drop(relu(norm(dense(400, flatten(l6)))), 0.1)
        outputs = lin(dense(17, l7))
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # deepCNN_2D
        # -----------------------------------------------------------------
        # l1 = maxP(convBlock_ks(8, (3,3), inputs), **poolargs)
        # l2 = maxP(convBlock_ks(16, (3,3), l1), **poolargs)
        # l3 = maxP(convBlock_ks(32, (3,3), l2), **poolargs)
        # l4 = maxP(convBlock_ks(64, (3,3), l3), **poolargs)
        # l5 = maxP(convBlock_ks(128, (3,3), l4), **poolargs)
        # l6 = maxP(convBlock_ks(256, (3,3), l5), **poolargs)
        # l7 = drop(relu(norm(dense(400, flatten(l6)))), 0.1)
        # outputs = lin(dense(17,l7))
        # -----------------------------------------------------------------

    if ResNet:
        img_rows, img_cols = 128, 64
        channels = 1
        input_shape = (channels, img_rows, img_cols)
        inputs = Input(shape=input_shape)

        # general comment: we started from Federico's design but due to the input size we
        # cannot use 3 Residual block (too many MaxP and dimension reduction) and the
        # 2 conv layers at the beginning must be skipped to keep 2 residual blocks.
        # on ResNet_2D_deep a design with these 2 layers is kept, but maxP is skipped
        # on them.

        # NB2: Residual Blocks from Federico's design differ from Residual Blocks on the
        # ResNet original paper.

        # -----------------------------------------------------------------
        # ResNet_2D_ks
        # -----------------------------------------------------------------
        l1 = ResidualBlock(32, 128, [13, 11, 9], inputs)
    l2 = ResidualBlock(64, 256, [7, 5, 3], l1)
    l3 = drop(relu(norm(dense(400, flatten(l2)))), 0.1)
    outputs = lin(dense(17, l3))

    # -----------------------------------------------------------------
    # ResNet_2D
    # -----------------------------------------------------------------
    # l1 = ResidualBlock(32, 128, [3,3,3], inputs)
    # l2 = ResidualBlock(64, 256, [3,3,3],l1)
    # l3 = drop(relu(norm(dense(400, flatten(l2)))),0.1)
    # outputs = lin(dense(17, l3))

    # -----------------------------------------------------------------
    # ResNet_2D_deep
    # -----------------------------------------------------------------
    # l1 = convBlock(8, inputs)
    # l2 = convBlock(16, l1)
    # l3 = ResidualBlock(32, 128, [13,11,9], l2)
    # l4 = ResidualBlock(64, 256, [7,5,3],l3)
    # l6 = drop(relu(norm(dense(400, flatten(l4)))),0.1)
    # outputs = lin(dense(17, l6))

# --- Create model
model = Model(inputs=inputs, outputs=outputs)

# --- Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),  # 2e-4 as starter
    loss=tf.keras.losses.MeanSquaredError(),
    experimental_run_tf_function=False)

# print(model.summary())
return model



# # reference for tuning
# # https://www.youtube.com/watch?v=vvC15l4CY1Q&ab_channel=sentdex
#
# hp = HyperParameters()
# def build_model(hp):
#     units1 = hp.Int('units1', min_value=150, max_value=400, step=5, default=200)
#     units2 = hp.Int('units2', min_value=200, max_value=400, step=5, default=290)
#     units3 = hp.Int('units3', min_value=100, max_value=400, step=5, default=160)
#
#     ksize1 = hp.Int('ks1', min_value=3, max_value=15, step=2, default=5)
#     ksize2 = hp.Int('ks2', min_value=3, max_value=15, step=2, default=13)
#     ksize3 = hp.Int('ks3', min_value=3, max_value=21, step=2, default=15)
#
#     #drop1 = hp.Float('drop1', min_value=0.05, max_value=0.5, step=0.05, default=0.1)
#     #drop2 = hp.Float('drop2', min_value=0.05, max_value=0.5, step=0.05, default=0.15)
#
#     #lrate = hp.Float('lrate', min_value=2e-6, max_value=2e-2, default=2e-4)
#
#     input_shape = (channels, img_rows, img_cols)
#     inputs = Input(shape=input_shape)
#
#     l1 = maxP(convBlock_ks_lin(units1, (ksize1,ksize1), inputs), **poolargs)
#     l2 = sDrop(maxP(convBlock_ks_lin(units2, (ksize2,ksize2), l1), **poolargs), 0.1)
#     l3 = sDrop(maxP(convBlock_ks_lin(units3, (ksize3,ksize3), l2), **poolargs), 0.15)
#
#     outputs = lin(dense(17, flatten(l3)))
#
#     # --- Create model
#     model = Model(inputs=inputs, outputs=outputs)
#
#     # --- Compile model
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),  #2e-4 as starter
#         loss = tf.keras.losses.MeanSquaredError(),
#         experimental_run_tf_function=False)
#     return model

outpath = 'C:/Users/Rudy/Desktop/Dl_models/dataset_20/'
tuner = BayesianOptimization(build_model,
                          objective = 'val_loss', #what u want to track
                          max_trials = 50, #how many randoms picking do we want to have
                          executions_per_trial=1, #number of time you train each dynamic version (see details below)
                          directory=outpath+'BayesianSearch/',
                          project_name= f"project_{int(time.time())}")

# NB: executions_per_trial:
# =1: when you shoot in the dark (you are simply looking  for a model that works at all, that learns)
# =3, =5 or even more: when you are keen to the best performance of a model that works

tf.debugging.set_log_device_placement(True)
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
tuner.search(X_train_rs, ny_train,
            epochs=100,
            batch_size=100,
            shuffle=True,
            validation_data=(X_val_rs, ny_val),
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
print(f"""
The hyperparameter search is complete. 
opt. units1 = {best_hps.get('units1')} /n opt. units2 = {best_hps.get('units2')} /n opt. units 3 = {best_hps.get('units3')} /n 
opt. ks1 = {best_hps.get('ks1')} /n opt ks2 = {best_hps.get('ks2')} /n opt. ks3 = {best_hps.get('ks3')} /n
opt. drop1 = {best_hps.get('drop1')} /n opt. drop2 = {best_hps.get('drop2')} /n opt. lrate = {best_hps.get('lrate')} /n
is.""")
