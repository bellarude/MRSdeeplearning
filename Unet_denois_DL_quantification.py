import glob
import csv
import os
import pandas as pd
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
from keras import backend as K
#import tensorflow.keras.backend as K

from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import scipy.io as sio

os.environ["KERAS_BACKEND"] = "theano"
K.set_image_data_format('channels_first')

import math
import seaborn as snc
#import xlsxwriter
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
regr = linear_model.LinearRegression()
import pandas as pd


import sys
sys.path.append('/content/drive/My Drive/EEG/DeepLearning_MRS/forFig5')

#cd "/content/drive/My Drive/DeepLearning_MRS/forFig5/"
#import MDv2_quantification_shallowELU
#import MD_quantification_shallowELU

# -------------------------------------------------------------------------------------------------------
#                               DATA LOAD
# -------------------------------------------------------------------------------------------------------

dest_folder = 'C:/Users/Rudy/Desktop/toMartyna/DeepLearning_MRS/forFig5/zoomedData/'
resultsPath = 'C:/Users/Rudy/Desktop/toMartyna/DeepLearning_MRS/forFig5/results/'
maxValue = sio.loadmat('C:/Users/Rudy/Desktop/toMartyna/DeepLearning_MRS/forFig5/zoomedData/labels/max_norm_Global_16met_wat_spgram_normGlobal_-1_1.mat')


data_import4 = sio.loadmat(dest_folder + 'zoomedSpgram_datasetX_4.mat')
labels_import4 = sio.loadmat(dest_folder + 'labels/labels_c_4.mat')
gt4 = np.transpose(labels_import4['labels_c']* 64.5)                    # (labels_import4['labels_c']* 64.5, (1,0))
labels_x = gt4                                                                  # np.concatenate((gt1,gt2,gt3,gt4), axis=0)
denoised4 = sio.loadmat(dest_folder + 'zoomedSpgram_pred_4.mat')

dataset4 = data_import4['output_noisy']
dataDenoised4 = denoised4['output']
maxVal = maxValue['mydata'][0][0];

# ------------------------------------------------------------------------------------------------------------
X_test  = dataset4[2999:4000, :, :, :]*maxVal             # (5000, 128, 32, 2)
y_test  = labels_x[2999:4000, :]                          # (1001, 16)
X_denoised =  dataDenoised4[2999:4000, :, :, :]*maxVal    # (5000, 128, 32, 2)

X_test_rs = np.transpose(X_test, (0, 3, 1, 2))
X_denoised_rs = np.transpose(X_denoised, (0, 3, 1, 2))

print('Dataset:'); print(X_test.shape)
print('Labels set:'); print(y_test.shape)


# -------------------------
# -------------------------
# -------------------------
labels_import4 = sio.loadmat(dest_folder + 'labels/labels_c_4.mat')
data_denoised4   = sio.loadmat(dest_folder + 'zoomedSpgram_pred_4.mat')

dataSet = data_import4['output_noisy']
labelsSet = labels_import4['labels_c']
denoisedSet = data_denoised4['output']

labels_x = np.transpose(labelsSet* 64.5)
dataset_noisy = dataSet
data_denoised = denoisedSet
maxVal = maxValue['mydata'][0][0];


# ----------------------------------------------
#      RE_TRAIN  MODEL
# ----------------------------------------------
data_import1 = sio.loadmat(dest_folder + 'zoomedSpgram_datasetX_1.mat'); data_import2   = sio.loadmat(dest_folder + 'zoomedSpgram_datasetX_2.mat')
data_import3 = sio.loadmat(dest_folder + 'zoomedSpgram_datasetX_3.mat'); data_import4   = sio.loadmat(dest_folder + 'zoomedSpgram_datasetX_4.mat')

labels_import1 = sio.loadmat(dest_folder + 'labels/labels_c_1.mat');  labels_import2 = sio.loadmat(dest_folder + 'labels/labels_c_2.mat')
labels_import3 = sio.loadmat(dest_folder + 'labels/labels_c_3.mat');  labels_import4 = sio.loadmat(dest_folder + 'labels/labels_c_4.mat')

data_denoised1   = sio.loadmat(dest_folder + 'zoomedSpgram_pred_1.mat');  data_denoised2   = sio.loadmat(dest_folder + 'zoomedSpgram_pred_2.mat')
data_denoised3   = sio.loadmat(dest_folder + 'zoomedSpgram_pred_3.mat');  data_denoised4   = sio.loadmat(dest_folder + 'zoomedSpgram_pred_4.mat')

dataSet = np.concatenate((data_import1['output_noisy'], data_import2['output_noisy'], data_import3['output_noisy'], data_import4['output_noisy']), axis=0)
labelsSet = np.concatenate((labels_import1['labels_c'], labels_import2['labels_c'], labels_import3['labels_c'], labels_import4['labels_c']), axis=1)
denoisedSet = np.concatenate((data_denoised1['output'], data_denoised2['output'], data_denoised3['output'], data_denoised4['output']), axis=0)

labels_x = np.transpose(labelsSet* 64.5)
dataset_noisy = dataSet
data_denoised = denoisedSet
maxVal = maxValue['mydata'][0][0];

# ------------------------------------------------------------------------------------------------------------
X_train_noisy = dataset_noisy[:18000, :, :, :]*maxVal
X_train_denoised = data_denoised[:18000, :, :, :]*maxVal
y_train = labels_x[:18000, :]

X_test_noisy = dataset_noisy[18000:19000, :, :, :]*maxVal
X_test_denoised = data_denoised[18000:19000, :, :, :]*maxVal
y_test = labels_x[18000:19000, :]

X_val_noisy = dataset_noisy[19000:, :, :, :]*maxVal
X_val_denoised = data_denoised[19000:, :, :, :]*maxVal
y_val = labels_x[19000:, :]

X_train_noisy_rs = np.transpose(X_train_noisy, (0, 3, 1, 2))
X_test_noisy_rs = np.transpose(X_test_noisy, (0, 3, 1, 2))

X_train_denoised_rs = np.transpose(X_train_denoised, (0, 3, 1, 2))
X_test_denoised_rs = np.transpose(X_test_denoised, (0, 3, 1, 2))

X_val_noisy_rs = np.transpose(X_val_noisy, (0, 3, 1, 2))
X_val_denoised_rs = np.transpose(X_val_denoised, (0, 3, 1, 2))

print('Dataset:'); print(X_test.shape)
print('Labels set:'); print(y_test.shape)

# --------------------------------------------------------------------------------------------------------
#                   TRAIN MODEL
#---------------------------------------------------------------------------------------------------------

output_folder = 'C:/Users/Rudy/Desktop/toMartyna/DeepLearning_MRS/forFig5/modelRR_quantification/NEW_results/'
net_name = "MD_zoomedQuant_Xnoisy"   #"RR_default"
checkpoint_path = output_folder + net_name + ".best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                     save_weights_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# selected channel 0 to keep only Re(spectrogram)
history = modelRR.fit(X_train_noisy_rs, y_train,
                      epochs=200,
                      batch_size=50,
                      shuffle=True,
                      validation_data=(X_val_noisy_rs, y_val),
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

pred_noisy = modelRR.predict(X_test_noisy_rs)



# -----------------------------------------------------------------------------------------------------------------
#                   LOAD DL MODEL
# -----------------------------------------------------------------------------------------------------------------
def newModel():
    # --- Define kwargs dictionary
    kwargs = {
        'kernel_size': (3,3), #was 2,2 before
        'padding': 'same'}

    # --- Define poolargs dictionary
    poolargs = {
        'pool_size': (2,2),
        'strides': (2,2)}

    # --- Define lambda functions
    conv = lambda x, filters, strides : layers.Conv2D(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization(axis=1)(x)
    normD = lambda x : layers.BatchNormalization(axis=1)(x)
    relu = lambda x : layers.ReLU()(x)
    lin = lambda x : tf.keras.activations.linear(x)
    elu = lambda x : tf.keras.activations.elu(x)
    tanh = lambda x : tf.keras.activations.tanh(x)
    maxP = lambda x, pool_size, strides : layers.MaxPooling2D(pool_size=pool_size, strides=strides)(x)
    #ConvPool = lambda filters, x : conv(x, filters, strides=2)
    add = lambda x, y: layers.Add()([x, y])
    avgP = lambda x, pool_size, strides: layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding = 'same')(x)

    conv_ks = lambda x, filters, strides, kernel_size : layers.Conv2D(filters=filters, strides=strides, kernel_size = kernel_size, padding = 'same')(x)

    flatten = lambda x : layers.Flatten()(x)
    dense = lambda units, x : layers.Dense(units=units)(x)
    maxP1D = lambda x : tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="same", data_format="channels_first")(x)

    convBlock = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    convBlock2 = lambda filters, x : convBlock(filters, convBlock(filters, x))
    #convBlock4 = lambda filters, x : convBlock2(filters, convBlock2(filters, x))

    convBlock_ks = lambda filters, ks, x : relu(norm(conv_ks(x, filters, strides=1, kernel_size = ks)))
    convBlock_ks_lin = lambda filters, ks, x : lin(norm(conv_ks(x, filters, strides=1, kernel_size = ks)))
    convBlock_ks_elu = lambda filters, ks, x : elu(conv_ks(x, filters, strides=1, kernel_size = ks))
    convBlock_ks_elu_s = lambda filters, ks, s, x : elu(conv_ks(x, filters, strides=s, kernel_size = ks))
    convBlock_2x_ks_elu = lambda filters, ks, x : convBlock_ks_elu(filters, ks, convBlock_ks_elu(filters, ks, x))
    convBlock_ks_tanh = lambda filters, ks, x : tanh(norm(conv_ks(x, filters, strides=1, kernel_size = ks)))

    # --- Concatenate
    concat = lambda a, b : layers.Concatenate(axis=1)([a, b])

    # --- Dropout
    sDrop = lambda x, rate : layers.SpatialDropout2D(rate, data_format='channels_first')(x)
    drop = lambda x, rate: layers.Dropout(rate, noise_shape=None, seed=None)(x)

    #Federico's blocks
    # NB: residual block from Federico's differ from the original one in ResNet paper (see ResNet50)
    bottleneck = lambda filters_input, filters_output, kernel_size, x :  norm(conv_ks(relu(norm(conv_ks(relu(norm(conv_ks(x, filters_input, strides=1, kernel_size=1))), filters_input, strides=1, kernel_size=kernel_size))), filters_output, strides=1, kernel_size=1))
    ResConvBlock = lambda filters_input, filters_output, kernel_size, x : relu(add(bottleneck(filters_input, filters_output, kernel_size, x) , norm(conv_ks(x, filters_output, strides=1, kernel_size=1)) ))
    IdeConvBlock = lambda filters_input, filters_output, kernel_size, x : relu(add(bottleneck(filters_input, filters_output, kernel_size, x), norm(x) ))
    ResidualBlock = lambda filters_input, filters_output, kernel_size, x : maxP(IdeConvBlock( filters_input, filters_output,  kernel_size[2], maxP( IdeConvBlock( filters_input, filters_output, kernel_size[1], maxP( ResConvBlock( filters_input, filters_output, kernel_size[0], x ),**poolargs )),**poolargs ) ), **poolargs)

    img_rows, img_cols = 128, 32
    channels = 2
    input_shape = (channels, img_rows, img_cols)
    inputs = Input(shape=input_shape)
    #-----------------------------------------------------------------
    # shallow ELU-CNN from ESMRMB abstract
    #-----------------------------------------------------------------
    # l1 = maxP(convBlock_ks_elu(50, (9,9), inputs), **poolargs)
    # l2 = sDrop(maxP(convBlock_ks_elu(100, (5,5), l1), **poolargs), 0.1)
    # l3 = sDrop(maxP(convBlock_ks_elu(150, (3,3), l2), **poolargs), 0.15)
    # outputs = lin(norm(dense(17, flatten(l3))))

    # lrate = 2e-4
    # [hp]
    l1 = maxP(convBlock_ks_elu(300, (7,7), inputs), **poolargs)
    l2 = sDrop(maxP(convBlock_ks_elu(100, (9,9), l1), **poolargs), 0.35)
    l3 = sDrop(maxP(convBlock_ks_elu(150, (5,5), l2), **poolargs), 0.35)
    outputs = lin(norm(dense(17, flatten(l3))))

    lrate = 5.3954e-3
   #-----------------------------------------------------------------
    # --- Create model
    modelRR = Model(inputs=inputs, outputs=outputs)
    # --- Compile model
    modelRR.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
        loss = tf.keras.losses.MeanSquaredError(),
        experimental_run_tf_function=False)

    print(modelRR.summary())
    return modelRR

modelRR = newModel()
