from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, \
    Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
# from keras import backend as K
import tensorflow.keras.backend as K

from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperModel, HyperParameters, RandomSearch
from kerastuner.tuners import BayesianOptimization

import time
from util import textMe
from numba import cuda
from models_unet_denoising import newModel, newModel_deep

def dataimport(index):
    global y_train, y_val, y_test, X_train, X_val, X_test

    os.environ["KERAS_BACKEND"] = "theano"
    K.set_image_data_format('channels_last')

    dest_folder = 'C:/Users/Rudy/Desktop/datasets/spectra_for_1D_Unet/'  # 'C:/Users/Rudy/Desktop/datasets/dataset_31/'

    data_import_1 = sio.loadmat(dest_folder + 'X_noisy_specta_matRecon_1.mat')
    data_import_2 = sio.loadmat(dest_folder + 'X_noisy_specta_matRecon_2.mat')
    data_import_3 = sio.loadmat(dest_folder + 'X_noisy_specta_matRecon_3.mat')
    data_import_4 = sio.loadmat(dest_folder + 'X_noisy_specta_matRecon_4.mat')

    labels_import_1 = sio.loadmat(dest_folder + 'Y_GT_specta_matRecon_1.mat')
    labels_import_2 = sio.loadmat(dest_folder + 'Y_GT_specta_matRecon_2.mat')
    labels_import_3 = sio.loadmat(dest_folder + 'Y_GT_specta_matRecon_3.mat')
    labels_import_4 = sio.loadmat(dest_folder + 'Y_GT_specta_matRecon_4.mat') #-- corrupted file, unpack again

    # cut last 1024 points, where the spectrum is
    dataset = np.concatenate([data_import_1['X_test_matlabRecon'], data_import_2['X_test_matlabRecon'], data_import_3['X_test_matlabRecon'],data_import_4['X_test_matlabRecon']], axis=1)              #, data_import_4['X_test_matlabRecon'][-1024:,:]], axis=1)
    dataset_ft = np.fft.fft(dataset, axis=0)
    dataset_ft_truncated = dataset_ft[-1024:, :]
    labels = np.concatenate([labels_import_1['Y_test_matlabRecon'], labels_import_2['Y_test_matlabRecon'], labels_import_3['Y_test_matlabRecon'], labels_import_4['Y_test_matlabRecon']], axis=1)         #, labels_import_4['Y_test_matlabRecon'][-1024:,:]], axis=1)
    labels_ft = np.fft.fft(labels, axis=0)
    labels_ft_truncated = labels_ft[-1024:, :]

    # here modify the test and train dataset so that I can load them wihtouth issues

    X_train = dataset_ft_truncated[:, 0:18000]
    X_test = dataset_ft_truncated[:, 18000:19000]
    X_val = dataset_ft_truncated[:, 19000:20000]  # check out

    y_train = labels_ft_truncated[:, 0:18000]
    y_test = labels_ft_truncated[:, 18000:19000]
    y_val = labels_ft_truncated[:, 19000:20000]


    return dataset, labels

datapoints = 1024

# MD
dataimport(1)

#modelUnet = newModel_deep(type=0)

X_train_full = np.zeros([18000, 1024, 2])
X_reshaped = np.transpose(X_train, (1, 0))
X_train_full[:,:,0] = np.real(X_reshaped)
X_train_full[:,:,1] = np.imag(X_reshaped)
Y_train_full = np.zeros([18000, 1024, 2])
Y_reshaped = np.transpose(y_train, (1, 0))
Y_train_full[:,:,0] = np.real(Y_reshaped)
Y_train_full[:,:,1] = np.imag(Y_reshaped)

X_val_full = np.zeros([1000, 1024, 2])
X_reshaped = np.transpose(X_val, (1, 0))
X_val_full[:,:,0] = np.real(X_reshaped)
X_val_full[:,:,1] = np.imag(X_reshaped)
Y_val_full = np.zeros([1000, 1024, 2])
Y_reshaped = np.transpose(y_val, (1, 0))
Y_val_full[:,:,0] = np.real(Y_reshaped)
Y_val_full[:,:,1] = np.imag(Y_reshaped)

X_test_full = np.zeros([1000, 1024, 2])
X_reshaped = np.transpose(X_test, (1, 0))
X_test_full[:,:,0] = np.real(X_reshaped)
X_test_full[:,:,1] = np.imag(X_reshaped)
Y_test_full = np.zeros([1000, 1024, 2])
Y_reshaped = np.transpose(y_test, (1, 0))
Y_test_full[:,:,0] = np.real(Y_reshaped)
Y_test_full[:,:,1] = np.imag(Y_reshaped)


modelUnet = newModel_deep(type=0)   # newModel(type=0)
output_folder = 'C:/Users/Rudy/Desktop/denoising_unet/'
net_name = "MD_64_1e4_learRate_32batch_fullDataset"   #"RR_default"
checkpoint_path = output_folder + net_name + ".best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                     save_weights_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# selected channel 0 to keep only Re(spectrogram)
history = modelUnet.fit(X_train_full, Y_train_full,
                      epochs=200,
                      batch_size=32,
                      shuffle=True,
                      validation_data=(X_val_full, Y_val_full),
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

pred_train = modelUnet.predict(X_test_full)










# channels = 1  # number of channels
# input_shape = (datapoints, channels)
# inputs = Input(shape=input_shape)
# channel_axis = 1 if K.image_data_format() == 'channels_first' else 2
#
# # --- Define kwargs dictionary
# kwargs = {
#     'strides': (1),
#     'padding': 'same'}
#
# # --- Define poolargs dictionary
# poolargs = {
#     'pool_size': (2),
#     'strides': (2)}
#
# # -----------------------------------------------------------------------------
# # Define lambda functions
# # -----------------------------------------------------------------------------
#
# conv = lambda x, kernel_size, filters: layers.Conv1D(filters=filters, kernel_size=kernel_size, **kwargs)(x)
# conv_s = lambda x, strides, filters: layers.Conv1D(filters=filters, kernel_size=3, strides=strides, padding='same')(x)
# # --- Define stride-1, stride-2 blocks
# conv1 = lambda filters, x: relu(norm(conv_s(x, filters=filters, strides=1)))
# conv1_lin = lambda filters, x: norm(conv_s(x, filters=filters, strides=1))
# conv2 = lambda filters, x: relu(norm(conv_s(x, filters=filters, strides=2)))
# # --- Define single transpose
# tran = lambda x, filters, strides: layers.Conv1DTranspose(filters=filters, strides=strides, kernel_size=3,
#                                                           padding='same')(x)
# # --- Define transpose block
# tran1 = lambda filters, x: relu(norm(tran(x, filters, strides=1)))
# tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=2)))
#
# norm = lambda x: layers.BatchNormalization(axis=channel_axis)(x)
# normD = lambda x: layers.BatchNormalization(axis=1)(x)
# relu = lambda x: layers.ReLU()(x)
# maxP = lambda x, pool_size, strides: layers.MaxPooling1D(pool_size=pool_size, strides=strides)(x)
#
# flatten = lambda x: layers.Flatten()(x)
# dense = lambda units, x: layers.Dense(units=units)(x)
#
# convBlock = lambda x, kernel_size, filters: relu(norm(conv(x, kernel_size, filters)))
# convBlock2 = lambda x, kernel_size, filters: convBlock(convBlock(x, kernel_size, filters), kernel_size, filters)
#
# convBlock_lin = lambda x, kernel_size, filters: norm(conv(x, kernel_size, filters))
# convBlock2_lin = lambda x, kernel_size, filters: convBlock(convBlock(x, kernel_size, filters), kernel_size, filters)
#
# concat = lambda a, b: layers.Concatenate(axis=channel_axis)([a, b])
#
#
# def concatntimes(x, n):
#     output = concat(x, x)
#     for i in range(n - 1):
#         output = concat(output, output)
#     return output
#
#
# add = lambda x, y: layers.Add()([x, y])
# ResidualBlock = lambda x, y: relu(add(x, y))
# dropout = lambda x, percentage, size: layers.Dropout(percentage, size)(x)
#
# hp = HyperParameters()

#
# def build_model(hp):
#     pad = 9
#     # -----------------------------------------------------------------------------
#     # RR-Unet 2xconv1
#     # -----------------------------------------------------------------------------
#     # --- Define contracting layers
#
#     units1 = hp.Int('units1', min_value=10, max_value=50, step=10, default=30)
#     units2 = hp.Int('units2', min_value=10, max_value=50, step=10, default=30)
#     units3 = hp.Int('units3', min_value=20, max_value=100, step=10, default=60)
#     units4 = hp.Int('units4', min_value=20, max_value=100, step=10, default=60)
#     units5 = hp.Int('units5', min_value=40, max_value=200, step=10, default=120)
#     units6 = hp.Int('units6', min_value=40, max_value=200, step=10, default=120)
#     units7 = hp.Int('units7', min_value=80, max_value=400, step=10, default=240)
#     units8 = hp.Int('units8', min_value=80, max_value=400, step=10, default=240)
#     units9 = hp.Int('units9', min_value=160, max_value=800, step=10, default=480)
#
#     lrate = hp.Float('lrate', min_value=2e-6, max_value=2e-2, default=1e-3)
#
#     l1 = conv1(units1 * 2, conv1(units1, tf.keras.layers.ZeroPadding1D(padding=(pad))(inputs)))
#     l2 = conv1(units3 * 2, conv1(units3, conv2(units2, l1)))
#     l3 = conv1(units5 * 2, conv1(units5, conv2(units4, l2)))
#     l4 = conv1(units7 * 2, conv1(units7, conv2(units6, l3)))
#     l5 = conv1(units9 * 2, conv1(units9, conv2(units8, l4)))
#
#     # --- Define expanding layers
#     l6 = tran2(units8, l5)
#
#     # --- Define expanding layers
#     l7 = tran2(units6, tran1(units7, tran1(units7 * 2, concat(l4, l6))))
#     l8 = tran2(units4, tran1(units5, tran1(units5 * 2, concat(l3, l7))))
#     l9 = tran2(units2, tran1(units3, tran1(units3 * 2, concat(l2, l8))))
#     l10 = conv1_lin(units1, conv1(units1 * 2, l9))
#
#     # --- Create logits
#     outputs = tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=1))
#
#     # --- Create model
#     modelRR = Model(inputs=inputs, outputs=outputs)
#
#     # --- Compile model
#
#     modelRR.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
#         loss=tf.keras.losses.MeanSquaredError(),
#         experimental_run_tf_function=False
#     )
#
#     return modelRR

# modelRR = build_model(X_train)
# print(modelRR.summary())



#
#
# # ----------------------------------------------------------
# #               U-net denoising
# # ----------------------------------------------------------
#
# from numpy import mean
# from numpy import std
# from numpy import dstack
# from pandas import read_csv
# from matplotlib import pyplot
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.layers.convolutional import Conv1D
# from keras.layers.convolutional import MaxPooling1D
# #from keras.utils import to_categorical
#

# MD
# #accuracY_model = evaluate_model(X_reshaped_1D, Y_reshaped_1D, X_test_reshaped_1D, Y_test_reshaped_1D)
#
# accuracY_model = evaluate_model(X_reshaped_1D, Y_reshaped_1D, X_test_reshaped_1D, Y_test_reshaped_1D)
#
#
# # fit and evaluate a model
# def evaluate_model(trainX, trainy, testX, testy):
#     verbose, epochs, batch_size = 0, 10, 32
#     n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
#     #n_timesteps, n_features, n_outputs = trainX.shape[0], trainX.shape[1], trainy.shape[0]
#     print('n_timesteps: ', n_timesteps, 'n_features: ', n_features, 'n_outputs: ', n_outputs)
#     input_shape = (n_timesteps, n_features)
#     print('input shape: ', input_shape)
#
#     model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu'))
#     model.add(Dense(n_outputs, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     model.summary()
#     # fit network
#     model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
#     # evaluate model
#     _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
#     return accuracy
#
#
# # summarize scores
# def summarize_results(scores):
#     print(scores)
#     m, s = mean(scores), std(scores)
#     print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
#
#
# # run an experiment
# def run_experiment(repeats=10):
#     # load data
#     trainX, trainy, testX, testy = load_dataset()
#     # repeat experiment
#     scores = list()
#     for r in range(repeats):
#         score = evaluate_model(trainX, trainy, testX, testy)
#     score = score * 100.0
#     print('>#%d: %.3f' % (r + 1, score))
#     scores.append(score)
#     # summarize results
#     summarize_results(scores)
#
#
# # run the experiment
# run_experiment()
#
#
# def unet_LeakyReLU(pretrained_weights = None,input_size = (128,128,2), ksize = 3, size_filter_in = 16, dropVal = 0.7):
#     #size filter input
#     #size_filter_in = 16
#     #normal initialization of weights
#     kernel_init = None
#     #To apply leaky relu after the conv layer
#     activation_layer = None
#     inputs = Input(input_size)
#     conv1 = Conv2D(size_filter_in, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(inputs)
#     conv1 = LeakyReLU()(conv1)
#     conv1 = Conv2D(size_filter_in, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv1)
#     conv1 = LeakyReLU()(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(size_filter_in*2, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(pool1)
#     conv2 = LeakyReLU()(conv2)
#     conv2 = Conv2D(size_filter_in*2, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv2)
#     conv2 = LeakyReLU()(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(size_filter_in*4, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(pool2)
#     conv3 = LeakyReLU()(conv3)
#     conv3 = Conv2D(size_filter_in*4, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv3)
#     conv3 = LeakyReLU()(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(size_filter_in*8, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(pool3)
#     conv4 = LeakyReLU()(conv4)
#     conv4 = Conv2D(size_filter_in*8, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv4)
#     conv4 = LeakyReLU()(conv4)
#     drop4 = Dropout(dropVal)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(size_filter_in*16, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(pool4)
#     conv5 = LeakyReLU()(conv5)
#     conv5 = Conv2D(size_filter_in*16, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv5)
#     conv5 = LeakyReLU()(conv5)
#     drop5 = Dropout(dropVal)(conv5)
#
#     up6 = Conv2D(size_filter_in*8, 2, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(drop5))
#     up6 = LeakyReLU()(up6)
#     merge6 = concatenate([drop4,up6], axis = 3)
#     conv6 = Conv2D(size_filter_in*8, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(merge6)
#     conv6 = LeakyReLU()(conv6)
#     conv6 = Conv2D(size_filter_in*8, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv6)
#     conv6 = LeakyReLU()(conv6)
#     up7 = Conv2D(size_filter_in*4, 2, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv6))
#     up7 = LeakyReLU()(up7)
#     merge7 = concatenate([conv3,up7], axis = 3)
#     conv7 = Conv2D(size_filter_in*4, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(merge7)
#     conv7 = LeakyReLU()(conv7)
#     conv7 = Conv2D(size_filter_in*4, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv7)
#     conv7 = LeakyReLU()(conv7)
#     up8 = Conv2D(size_filter_in*2, 2, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv7))
#     up8 = LeakyReLU()(up8)
#     merge8 = concatenate([conv2,up8], axis = 3)
#     conv8 = Conv2D(size_filter_in*2, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(merge8)
#     conv8 = LeakyReLU()(conv8)
#     conv8 = Conv2D(size_filter_in*2, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv8)
#     conv8 = LeakyReLU()(conv8)
#
#     up9 = Conv2D(size_filter_in, 2, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv8))
#     up9 = LeakyReLU()(up9)
#     merge9 = concatenate([conv1,up9], axis = 3)
#     conv9 = Conv2D(size_filter_in, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(merge9)
#     conv9 = LeakyReLU()(conv9)
#     conv9 = Conv2D(size_filter_in, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv9)
#     conv9 = LeakyReLU()(conv9)
#     conv9 = Conv2D(2, kernel_size= ksize, activation = activation_layer, padding = 'same', kernel_initializer = kernel_init)(conv9)
#     conv9 = LeakyReLU()(conv9)
#     #conv10 = Conv2D(1, 1, activation = 'tanh')(conv9)
#
#     model = Model(inputs,conv9)
#
#     model.compile(optimizer = tf.keras.optimizers.Adam(), loss = tf.keras.losses.Huber(), metrics = ['mse']) ## here check out what was the default learning rate
#     # 3e-4; default 1e-3
#
#     model.summary()
#
#     if(pretrained_weights):
#     	model.load_weights(pretrained_weights)
#
#     return model
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # ----------------------------------------------------------
# #               OLD RUN
# # ----------------------------------------------------------
# '''
# metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
#             'Tau', 'Water']
#
# order = [8, 10, 1, 11, 12, 2, 13, 14, 15, 4, 16, 9, 5, 17, 3, 6, 7]
#
# for idx in range(15, len(metnames)):
#     start = time.time()
#     dataimport(order[idx])
#
#     outpath = 'C:/Users/Rudy/Desktop/Dl_models/'
#     tuner = BayesianOptimization(build_model,
#                                  objective='val_loss',  # what u want to track
#                                  max_trials=50,  # how many randoms picking do we want to have
#                                  executions_per_trial=1,
#                                  # number of time you train each dynamic version (see details below)
#                                  directory=outpath + 'BayesianSearch/',
#                                  project_name='project_Unet_' + metnames[idx] + '_t' + str(int(time.time())) + '_NOwat')
#
#     tf.debugging.set_log_device_placement(True)
#     # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
#     tuner.search(X_train, y_train,
#                  epochs=100,
#                  batch_size=100,
#                  shuffle=True,
#                  validation_data=(X_val, y_val),
#                  validation_freq=1,
#                  callbacks=[es]
#                  )
#
#     end = time.time()
#     elapsedtime = (end - start) / 3600  # in hours
#
#     textMe(str(idx) + '. DONE UNet-hp ' + metnames[idx] + ', time -> ' + '{0:.2f}'.format(elapsedtime) + 'h')
#     # device = cuda.get_current_device()
#     # device.reset()
# '''