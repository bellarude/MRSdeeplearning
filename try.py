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

# from keras import layers
# from keras.callbacks import ModelCheckpoint, EarlyStopping
#
# import tensorflow as tf
#
# # pip install tf-nightly-gpu
# # import tensorflow as tf
# #
# lin = lambda x: tf.keras.activations.linear(x)
# norm = lambda x: layers.BatchNormalization(axis=-1)(x)
# dense = lambda units, x: layers.Dense(units=units)(x)
# flatten = lambda x: layers.Flatten()(x)
# #
#
# from nfnets_keras import NFNetF3
#
# model = NFNetF3(include_top = True, num_classes = 10)
# model.compile('adam', 'categorical_crossentropy')
# # IMG_SHAPE = (224, 224, 3)
# IMG_SHAPE = (224, 224, 2)
# model0 = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights=None)
# outputs = lin(norm(dense(17, flatten(model0.output))))

import time

start = time.time()


# tf.keras.utils.plot_model(model0) # to draw and visualize
# model.summary() # to see the list of layers and parameters

from data_load_norm import dataimport2D, labelsimport, labelsNorm, ilabelsNorm, inputConcat2D, dataimport2D_md, labelsimport_md
from models import newModel

md_input = 0
flat_input = 0

if md_input == 0:
    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'
    # dataname = 'dataset_spgram.mat'
    # X_train, X_val = dataimport2D(folder, dataname, 'dataset')

    labelsname = 'labels_c.mat'
    y_train, y_val = labelsimport(folder, labelsname, 'labels_c')
    # nX_train_rs = dataNorm(X_train_rs)
    # nX_val_rs = dataNorm(X_val_rs)

else:
    #Martyna's noisy - denoised - GT dataset

    # pred --> output
    # datasetX --> output_noisy
    # labelsY --> output_gt

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33_gauss/'
    filenames = ['zoomedSpgram_labelsY_1.mat',
                 'zoomedSpgram_labelsY_2.mat',
                 'zoomedSpgram_labelsY_3.mat',
                 'zoomedSpgram_labelsY_4.mat']
    keyname = 'output_gt'

    X_train_gt, X_val_gt, X_test_gt = dataimport2D_md(folder, filenames, keyname)

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33_gauss/'
    filenames = ['zoomedSpgram_datasetX_1.mat',
                 'zoomedSpgram_datasetX_2.mat',
                 'zoomedSpgram_datasetX_3.mat',
                 'zoomedSpgram_datasetX_4.mat']
    keyname = 'output_noisy'

    X_train, X_val, X_test = dataimport2D_md(folder, filenames, keyname)

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33_gauss/'
    filenames = ['zoomedSpgram_pred_1.mat',
                 'zoomedSpgram_pred_2.mat',
                 'zoomedSpgram_pred_3.mat',
                 'zoomedSpgram_pred_4.mat']
    keyname = 'output'

    X_train_pred, X_val_pred, X_test_pred = dataimport2D_md(folder, filenames, keyname)

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33_gauss/labels/'
    filenames = ['labels_c_1.mat',
                 'labels_c_2.mat',
                 'labels_c_3.mat',
                 'labels_c_4.mat']
    keyname = 'labels_c'

    y_train, y_val, y_test = labelsimport_md(folder, filenames, keyname)

ny_train, w_y_train = labelsNorm(y_train)
ny_val, w_y_val = labelsNorm(y_val)

folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33_gauss/labels/'
filename = 'spectrum16m_shim_gt_1.mat'
spectra_import = sio.loadmat(folder + filename)
spectra_gt = spectra_import['spectra_gt']

filename = 'spectrum16m_shim_snr_mmbl_1.mat'
spectra_import = sio.loadmat(folder + filename)
spectra = spectra_import['spectra']

# toplot = 15001
# fig = plt.figure()
# fig.add_subplot(131)
# plt.imshow(X_train[toplot,:,:,0])
# fig.add_subplot(132)
# plt.imshow(X_train_gt[toplot,:,:,0])
# fig.add_subplot(133)
# plt.imshow(X_train_pred[toplot,:,:,0])
#
# fig = plt.figure()
# fig.add_subplot(121)
# plt.plot(spectra[:,toplot])
# fig.add_subplot(122)
# plt.plot(spectra_gt[:,toplot])



# w = np.power((ny_val[:,0]-0.5), 2)
#
# fig = plt.figure()
# plt.histt(w)
#
# fig.add_subplot(121)
# plt.hist(y_val[:,0], 20)
# fig.add_subplot(122)
# plt.hist(wy_val,20)

# import smtplib
#
# def textMe(string):
#     # needs to abilitate less secure app: https://www.google.com/settings/security/lesssecureapps
#     content = (string)
#     mail = smtplib.SMTP('smtp.gmail.com', 587)
#     mail.ehlo()
#     mail.starttls()
#     address = 'rudy.rizzo.tv@gmail.com'
#     mail.login('amsmdeepmrs@gmail.com', 'amsmdeepmrs20')
#     mail.sendmail('amsmdeepmrs@gmail.com', address, content)
#     mail.close()
#     print(">>> sent E-mail @" + address)
#
# textMe('prova finita #2')



end = time.time()

elapsed = end - start
print('{0:.2f}'.format(elapsed))

i = 2

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']
idx = 5
from util import textMe
print(str(idx) + '. DONE: UNet-hp ' + metnames[idx] + ' - time: {0:.2f}'.format(elapsed))
# textMe('ciao')
textMe(str(idx) + '. DONE UNet-hp ' + metnames[idx] + ', time -> {0:.2f}'.format(elapsed) + "sec")