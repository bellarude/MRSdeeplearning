from __future__ import print_function
import os
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
import tensorflow.keras.backend as K
from keras import layers
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from models import newModel


K.set_image_data_format('channels_last')
channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

# mu sigma of use in case we want to consider a full matrix instead of a diagonal one
def mu_sigma(output):
    p = 17  # y_train.shape[1]
    # output = np.mean(output, axis=0) # account batch normalization
    #
    # mu = output[0:p]
    # T1 = output[p:2 * p]
    # T2 = output[2 * p:]
    mu = output[0][0:p]
    T1 = output[0][p:2 * p]
    T2 = output[0][2 * p:]

    ones = tf.ones((p, p), dtype=tf.float32)
    mask_a = tf.linalg.band_part(ones, 0, -1)
    mask_b = tf.linalg.band_part(ones, 0, 0)
    mask = tf.subtract(mask_a, mask_b)
    zero = tf.constant(0, dtype=tf.float32)
    non_zero = tf.not_equal(mask, zero)
    indices = tf.where(non_zero)
    T2 = tf.sparse.SparseTensor(indices, T2, dense_shape=tf.cast((p, p),
                                                                 dtype=tf.int64))
    T2 = tf.sparse.to_dense(T2)
    T1 = tf.linalg.diag(T1)
    sigma = T1 + T2
    return mu, sigma

model = newModel(dim='2D', type='dualpath_net', subtype='ShallowELU')

from data_load_norm import dataimport2D, labelsimport, labelsNorm, ilabelsNorm, inputConcat2D, dataimport2D_md, labelsimport_md, dataimport2Dhres

md_input = 1
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
    filenames = ['zoomedSpgram_labelsY_1.mat',
                 'zoomedSpgram_labelsY_2.mat',
                 'zoomedSpgram_labelsY_3.mat',
                 'zoomedSpgram_labelsY_4.mat']
    keyname = 'output_gt'

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

if flat_input:
    X_train = inputConcat2D(X_train)
    X_val = inputConcat2D(X_val)

if resize_input:
    X_train = tf.image.resize(X_train, (224, 224))
    X_val = tf.image.resize(X_val, (224, 224))


outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"
subfolder = ""
net_name = "ShallowNet-2D2c-ESMRMB_dualPath_wmse_noiseless"

checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"

# model.load_weights(checkpoint_path)
checkpoint_dir = os.path.dirname(checkpoint_path)
mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                     save_weights_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

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

