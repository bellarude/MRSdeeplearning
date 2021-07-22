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
    K.set_image_data_format('channels_last')
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'

    data_import = sio.loadmat(dest_folder + 'dataset_spectra.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_c.mat')

    dataset = data_import['dataset_spectra']
    labels = labels_import['labels_c'] * 64.5

    X_train = dataset[0:18000, :]
    X_val = dataset[18000:20000, :]
    X_test = dataset[19000:20000, :]  # unused

    y_train = labels[0:18000, :]
    y_val = labels[18000:20000, :]
    y_test = labels[19000:20000, :]

    # reshaping
    X_train_rs = np.transpose(X_train, (0, 2, 1))
    X_val_rs = np.transpose(X_val, (0, 2, 1))
    X_test_rs = np.transpose(X_test, (0, 2, 1))


def dataNorm(dataset):
    dataset_norm = np.empty(dataset.shape)
    for i in range(dataset.shape[0]):
        M = np.amax(np.abs(dataset[i, :, :]))
        dataset_norm[i, :, 0] = dataset[i, :, 0] / M
        dataset_norm[i, :, 1] = dataset[i, :, 1] / M
    return dataset_norm

def oneChannelDataset(dataset):
    OneChannel = np.zeros([dataset.shape[0], dataset.shape[1]*2, 1])
    OneChannel[:,:,0] = np.concatenate((dataset[:,:,0], dataset[:,:,1] ),axis=1)
    return OneChannel

def labelsNorm(labels):
  labels_norm = np.empty(labels.shape)
  weights = np.empty([labels.shape[0],1])
  for i in range(labels.shape[1]):
    w = np.amax(labels[:,i])
    labels_norm[:,i] = labels[:,i] / w
    weights[i] = w

  return labels_norm, weights

def ilabelsNorm(labels_norm, weights):
  ilabels = np.empty(labels_norm.shape)
  for i in range(labels_norm.shape[1]):
    ilabels[:,i] = labels_norm[:,i] * weights[i]

  return ilabels

# --- Define kwargs dictionary
kwargs = {
    'kernel_size': (3),
    'padding': 'same'}

# --- Define poolargs dictionary
poolargs = {
    'pool_size': (2),
    'strides': (2)}

# -----------------------------------------------------------------------------
# Define lambda functions
# -----------------------------------------------------------------------------
os.environ["KERAS_BACKEND"] = "theano"
K.set_image_data_format('channels_last')
channel_axis = 1 if K.image_data_format() == 'channels_first' else 2

conv = lambda filters, x, strides : layers.Conv1D(filters=filters, strides=strides, **kwargs)(x)
conv_ks = lambda filters, x, strides, kernel_size : layers.Conv1D(filters=filters, strides=strides, kernel_size = kernel_size, padding = 'same')(x)

norm = lambda x : layers.BatchNormalization(axis=channel_axis)(x)
normD = lambda x : layers.BatchNormalization(axis=1)(x)
relu = lambda x : layers.ReLU()(x)
lin = lambda x : tf.keras.activations.linear(x)
tanh = lambda x : tf.keras.activations.tanh(x)
elu = lambda x : tf.keras.layers.ELU(alpha=1.0)(x)
maxP = lambda x, pool_size, strides : layers.MaxPooling1D(pool_size=pool_size, strides=strides)(x)
ConvPool = lambda x, filters : conv(filters, x, strides=2)

flatten = lambda x : layers.Flatten()(x)
dense = lambda units, x : layers.Dense(units=units)(x)

convBlock_ks = lambda filters, ks, x : relu(norm(conv_ks(filters, x, strides=1, kernel_size = ks)))
convBlock_ks_lin = lambda filters, ks, x : lin(norm(conv_ks(filters, x, strides=1, kernel_size = ks)))
convBlock_ks_tanh = lambda filters, ks, x : tanh(norm(conv_ks(filters, x, strides=1, kernel_size = ks)))

convBlock = lambda filters, x : relu(norm(conv(filters, x, strides=1)))
convBlockELU = lambda filters, x : elu(conv(filters, x, strides=1))
convBlock2 = lambda filters, x : convBlock(filter, convBlock(filters, x))
convBlock3 = lambda filters, x : convBlock(filters, convBlock2(filters, x))
convBlock4 = lambda filters, x : convBlock2(filters, convBlock2(filters, x))

convShortcut = lambda filters, x : norm(conv(filters, x, strides=1))

concat = lambda a, b : layers.Concatenate(axis=2)([a, b])

def concatntimes(x, n):
  output = concat(x, x)
  for i in range(n-1):
    output = concat(output, output)
  return output

add = lambda x, y: layers.Add()([x, y])


#dropout = lambda x, percentage, size : layers.Dropout(percentage, size)(x)
drop = lambda x, rate: layers.Dropout(rate, noise_shape=None, seed=None)(x)

#Federico's blocks
bottleneck = lambda filters_input, filters_output, x :  norm(conv_ks(filters_output, relu(norm(conv_ks(filters_input, relu(norm(conv_ks(filters_input, x, strides=1, kernel_size=1))), strides=1, kernel_size=3))), strides=1, kernel_size=1))
ResConvBlock = lambda filters_input, filters_output, x : relu(add(bottleneck(filters_input, filters_output, x) , norm(conv_ks(filters_output,x,strides=1, kernel_size=1)) ))
IdeConvBlock = lambda filters_input, filters_output, x : relu(add(bottleneck(filters_input, filters_output, x), norm(x) ))
ResidualBlock = lambda filters_input, filters_output, x : maxP(IdeConvBlock( filters_input, filters_output, maxP( IdeConvBlock( filters_input, filters_output, maxP( ResConvBlock( filters_input, filters_output,x ),**poolargs )),**poolargs ) ), **poolargs)


#ResNet paper's block
conv2id = lambda filters, x : relu( add( norm(conv( filters, convBlock(filters, x), strides=1 )), norm(x) ) )
conv2conv = lambda filters, x : relu( add( norm(conv( filters, convBlock(filters, x), strides=2 )), norm(conv(filters,x,strides=1) ) ) )
DRB1 = lambda filters, x : conv2id( filters, conv2id( filters, conv2id(filters,x) ) )


hp = HyperParameters()
def build_model(hp):
    # input image dimensions

    # datapoints = 1024
    # channels = 2

    ResidualNet = 1  # if activate applies a residual network strategy
    DeepConvolution = 0  # if activate the pooling with convolution
    Variational = 0

    if ResidualNet:
        # -----------------------------------------------------------------
        # ResNet_fed
        # -----------------------------------------------------------------

        units1 = hp.Int('units1', min_value=5, max_value=40, step=10, default=8)
        units2 = hp.Int('units2', min_value=10, max_value=60, step=10, default=16)
        units3 = hp.Int('units3', min_value=20, max_value=100, step=10, default=32)
        units4 = hp.Int('units4', min_value=70, max_value=200, step=10, default=128)
        units5 = hp.Int('units5', min_value=50, max_value=150, step=10, default=64)
        units6 = hp.Int('units6', min_value=170, max_value=330, step=10, default=256)
        units7 = hp.Int('units7', min_value=100, max_value=200, step=10, default=128)
        units8 = hp.Int('units8', min_value=400, max_value=600, step=10, default=512)
        unitsDense = hp.Int('unitsDense', min_value=100, max_value=1000, step=10, default=400)

        # ksize1 = hp.Int('ks1', min_value=3, max_value=11, step=2, default=9)
        # ksize2 = hp.Int('ks2', min_value=3, max_value=11, step=2, default=5)
        # ksize3 = hp.Int('ks3', min_value=3, max_value=11, step=2, default=3)

        drop1 = hp.Float('drop1', min_value=0.1, max_value=0.8, step=0.05, default=0.1)
        # drop2 = hp.Float('drop2', min_value=0.05, max_value=0.5, step=0.05, default=0.15)

        lrate = hp.Float('lrate', min_value=2e-5, max_value=2e-2, default=1e-3)


        datapoints = 2048
        channels = 1

        input_shape = (datapoints, channels)
        inputs = Input(shape=input_shape)

        l1 = maxP(convBlock(units1, inputs), **poolargs)
        l2 = maxP(convBlock(units2, l1), **poolargs)
        l3 = ResidualBlock(units3, units4, l2)
        l4 = ResidualBlock(units5, units6, l3)
        l5 = ResidualBlock(units7, units8, l4)
        l6 = drop(relu(normD(dense(unitsDense, flatten(l5)))), drop1)
        outputs = lin(dense(17, l6))
        # -----------------------------------------------------------------

        # #-----------------------------------------------------------------
        # # ResNet_orig
        # #-----------------------------------------------------------------
        # datapoints = 2048
        # channels = 1
        #
        # input_shape = (datapoints, channels)
        # inputs = Input(shape=input_shape)
        # l1 = maxP(convBlock_ks(64, 7, inputs), **poolargs)
        # l2 = DRB1(64, l1)
        # l3 = DRB1(128, conv2conv(128, l2))
        # l4 = conv2id(256, conv2id(256, DRB1(256, conv2conv(256, l3))))
        # l5 = maxP(conv2id(512, conv2id(512, conv2conv(512, l4))), **poolargs)
        # l6 = drop(relu(norm(dense(1000, flatten(l5)))), 0.1)
        # outputs = lin(dense(17, l6))
        # #-----------------------------------------------------------------

    if DeepConvolution:
        # -----------------------------------------------------------------
        # deepCNN_fed
        # -----------------------------------------------------------------

        datapoints = 2048
        channels = 1

        input_shape = (datapoints, channels)
        inputs = Input(shape=input_shape)

        l1 = maxP(convBlock(units1, inputs), **poolargs)
        l2 = maxP(convBlock(units2, l1), **poolargs)
        l3 = maxP(convBlock(units3, l2), **poolargs)
        l4 = maxP(convBlock(units4, l3), **poolargs)
        l5 = maxP(convBlock(units5, l4), **poolargs)
        l6 = maxP(convBlock(units6, l5), **poolargs)
        l7 = drop(relu(normD(dense(unitsDense, flatten(l6)))), drop1)
        outputs = lin(dense(17, l7))
        # -----------------------------------------------------------------

    if Variational:
        train_size = X_train_rs.shape[0]
        l1 = maxP(varBlock(8, inputs), **poolargs)
        l2 = maxP(varBlock(16, l1), **poolargs)
        l3 = maxP(varBlock(32, l2), **poolargs)
        l4 = maxP(varBlock(64, l3), **poolargs)
        l5 = maxP(varBlock(128, l4), **poolargs)
        l6 = maxP(varBlock(256, l5), **poolargs)
        l7 = drop(relu(dense(400, flatten(l6))), 0.1)

    # else:
    #   l1 = maxP(convBlock_ks(50, (3), inputs), **poolargs)
    #   #l2 = drop(maxP(convBlock_ks_lin(100, (5), l1), **poolargs), 0.1)
    #   #l3 = drop(maxP(convBlock_ks_lin(150, (3), l2), **poolargs), 0.15)
    #   l2 = drop(maxP(convBlock_ks(100, (3), l1), **poolargs),0.1)
    #   l3 = drop(maxP(convBlock_ks(150, (3), l2), **poolargs),0.15)
    #   outputs = lin(norm(dense(17,flatten(l3))))

    # outputs = relu(dense(4096, dense(512, flatten(l3))))
    # l7 = drop(relu(dense(500, flatten(l5))),0.1)
    # l8 = drop(relu(dense(1000, flatten(l7))),0.2)
    # l7 = relu(dense(248, l6))

    # --- Create model
    modelRR = Model(inputs=inputs, outputs=outputs)

    # --- Compile model
    modelRR.compile(
        # optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), #optimization from Pruthvi
        optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
        loss=tf.keras.losses.MeanSquaredError(),
        experimental_run_tf_function=False
    )

    print(modelRR.summary())
    return modelRR

dataimport()

# nX_train_rs_flat = oneChannelDataset(dataNorm(X_train_rs))
# nX_val_rs_flat = oneChannelDataset(dataNorm(X_val_rs))
# nX_test_rs_flat = oneChannelDataset(dataNorm(X_test_rs))

nX_train_rs_flat = oneChannelDataset(X_train_rs)
nX_val_rs_flat = oneChannelDataset(X_val_rs)

ny_train, w_y_train = labelsNorm(y_train)
ny_val, w_y_val = labelsNorm(y_val)

outpath = 'C:/Users/Rudy/Desktop/Dl_models/'
tuner = BayesianOptimization(build_model,
                           objective='val_loss',  # what u want to track
                           max_trials=50,  # how many randoms picking do we want to have
                           executions_per_trial=1,
                           # number of time you train each dynamic version (see details below)
                           directory=outpath + 'BayesianSearch/',
                           project_name=f"project_{int(time.time())}")

# NB: executions_per_trial:
# =1: when you shoot in the dark (you are simply looking  for a model that works at all, that learns)
# =3, =5 or even more: when you are keen to the best performance of a model that works

tf.debugging.set_log_device_placement(True)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
tuner.search(nX_train_rs_flat, ny_train,
           epochs=100,
           batch_size=100,
           shuffle=True,
           validation_data=(nX_val_rs_flat, ny_val),
           validation_freq=1,
           callbacks=[es]
           )

# When search is over, you can retrieve the best model(s):
# print("------------------------")
# models = tuner.get_best_models(num_models=2)
# print("------------------------")
# Or print a summary of the results:
# tuner.results_summary()

print("------------------------")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

  
