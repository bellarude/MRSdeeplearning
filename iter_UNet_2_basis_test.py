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
from models_to_basis import newModel
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperModel, HyperParameters, RandomSearch
from kerastuner.tuners import BayesianOptimization
from util_plot import jointregression, blandAltmann_Shim, blandAltmann_SNR, sigma_distro, sigma_vs_gt
import time
import pickle

# interval to train metabolites, help in debug or developing phase:
# full range of metabolites is defined in [0, len(metnames)=17] = [0,1,2,3,4,...,16]
met2train_start = 0
met2train_stop = 17

same = 0 #same == 1 means having same network architecture for all metabolites
#NB 01.01.2022: same = 0 is not supported for dualpath_net: dualpath architecture has not yet been optimized
dualpath_net = 0

doSNR=0
doShim=0
doSavings=0

os.environ["KERAS_BACKEND"] = "theano"
K.set_image_data_format('channels_last')

dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'
def dataimport(dest_folder, index):
    labels_import = sio.loadmat(dest_folder + 'labels_kor_' + str(index) + '_TEST_NOwat.mat')
    labels = labels_import['labels_kor_' + str(index)]
    return labels
#
#
data_import = sio.loadmat(dest_folder + 'spectra_kor_TEST_wat.mat')
conc_import = sio.loadmat(dest_folder + 'labels_c_TEST.mat')
snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')
#
X_test = data_import['spectra_kor']
conc = conc_import['labels_c'] * 64.5  # mM
snr_v = snr_v['snr_v']
shim_v = readme_SHIM['shim_v']

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']

order = [8, 10, 1, 11, 12, 2, 13, 14, 15, 4, 16, 9, 5, 17, 3, 6, 7] #to order them
ordernames = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5] #to plot them in the order we want

tf.debugging.set_log_device_placement(True)

loss = []
pred = []
labels_list = []
# X_test = dataset
for idx in range(met2train_start, met2train_stop):
    labels = dataimport(dest_folder, order[idx])
    # dataimport(order[idx])
    if same:
        if dualpath_net:
            model = newModel(type=1)
            spec = '_dualpath_elu_nonorm'
        else:
            model = newModel(type=0)
            spec = ''

    else:
        model = newModel(met=metnames[idx], type=0)
        spec = '_doc'

    y_test = labels
    output_folder = 'C:/Users/Rudy/Desktop/DL_models/'
    subfolder = "net_type/unet/"
    net_name = "UNet_" + metnames[idx] + spec + "_NOwat"

    checkpoint_path = output_folder + subfolder + net_name + ".best.hdf5"
    model.load_weights(checkpoint_path)
    # a = model.evaluate(X_test, y_test, verbose=0)
    # b = model.predict(X_test)

    loss.append(model.evaluate(X_test, y_test, verbose=0))
    pred.append(model.predict(X_test))  # normalized [0-1] absolute concentrations prediction
    labels_list.append(y_test)


# pred_sum = np.empty((X_test.shape[0], 17))
# labels_sum = np.empty((X_test.shape[0], 17))
# for idx in range(met2train_start, met2train_stop):
#     for smp in range(0, X_test.shape[0]):
#         pred_sum[smp, idx] = np.sum(pred[idx][smp, :, 0])
#         labels_sum[smp, idx] = np.sum(labels_list[idx][smp, :])

# check areas for only 1 network (ie 1 met)!
pred_sum = np.empty((X_test.shape[0], 17))
labels_sum = np.empty((X_test.shape[0], 17))
for mm in range(0, 17):
    for smp in range(0, X_test.shape[0]):
        pred_sum[smp, mm] = np.sum(pred[mm][smp, :, 0])
        labels_sum[smp, mm] = np.sum(labels_list[mm][smp, :])

#file_name = output_folder + subfolder + "new_pred_list.pkl"

#open_file = open(file_name, "wb")
#pickle.dump(pred_sum, open_file)
#open_file.close()
#
# loss_train = model.evaluate(X_train, y_train, verbose=0)
# pred_train = model.predict(X_train)


# fig=plt.figure()
# plt.subplot(2,1,1)
# plt.plot(np.flip(X_train[100,:]))
# plt.subplot(2,1,2)
# plt.plot(np.flip(y_train[100,:]))
# plt.plot(np.flip(pred_train[100,:,0]))

fig=plt.figure()
plt.subplot(2,1,1)
plt.plot(np.flip(X_test[100,:]))
plt.subplot(2,1,2)
plt.plot(np.flip(y_test[100,:]))
plt.plot(np.flip(pred[0][100,:,0]))

# we need to scale them in 0-1 otherwise plotting is a pain
def norm_sum(input):
    output = np.empty(input.shape)
    for idx in range(0, input.shape[1]):
        output[:, idx] = input[:, idx] / np.max(input[:,idx])
    return output

npred_sum = norm_sum(pred_sum)
nlabels_sum = norm_sum(labels_sum)

fig = plt.figure()
jointregression(fig, nlabels_sum, npred_sum, metnames[0],
                        snr_v=snr_v)

def rel2wat(input):
    output = np.empty(input.shape)
    for idx in range(0,input.shape[1]):
        output[:, idx] = np.divide(input[:, idx], input[:, 16])
    return output

# this is a work around to convert it back to mM. We exploit S_labels using eventually as refernce
# labels in mM coming from matlab directly!!!
labels_mM = rel2wat(labels_sum)*64.5
pred_mM = rel2wat(pred_sum)*64.5 / labels_mM * conc

if (met2train_stop - met2train_start < 17):
    # plot unreferenced predictions
    # 1. I assume not knowing the water, since it needs a dedicated network
    # 2. I do not care of ordernames[] variable since there might be no order on plotting
    for idx in range(met2train_start, met2train_stop):
        fig = plt.figure()
        jointregression(fig, npred_sum[:, idx], nlabels_sum[:, idx], metnames[idx],
                        snr_v=snr_v)

    if doShim:
        for idx in range(met2train_start, met2train_stop):
            fig = plt.figure()
            blandAltmann_Shim(fig, npred_sum[:, idx], nlabels_sum[:, idx], shim_v=shim_v, metname=metnames[idx], snr_v=snr_v)

    if doSNR:
        for idx in range(met2train_start, met2train_stop):
            fig = plt.figure()
            blandAltmann_SNR(fig, npred_sum[:, idx], nlabels_sum[:, idx], metnames[idx], snr_v=snr_v)

else:
    def plotREGR2x4fromindex(i):
        fig = plt.figure(figsize=(40, 10))

        widths = 2 * np.ones(4)
        heights = 2 * np.ones(2)
        spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                height_ratios=heights)

        if doSNR:
            for row in range(2):
                for col in range(4):
                    ax = fig.add_subplot(spec[row, col])
                    if (i == 0) or (i == 8):
                        jointregression(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=snr_v,
                                        outer=spec[row, col], sharey=1)
                    elif (i == 4) or (i == 12):
                        jointregression(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=snr_v,
                                        outer=spec[row, col], sharex=1, sharey=1)
                    elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                        jointregression(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=snr_v,
                                        outer=spec[row, col], sharex=1)
                    else:
                        jointregression(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=snr_v,
                                        outer=spec[row, col])

                    i += 1

        else:
            for row in range(2):
                for col in range(4):
                    ax = fig.add_subplot(spec[row, col])
                    if (i == 0) or (i == 8):
                        jointregression(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=[],
                                        outer=spec[row, col], sharey=1)
                    elif (i == 4) or (i == 12):
                        jointregression(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=[],
                                        outer=spec[row, col], sharex=1, sharey=1)
                    elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                        jointregression(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=[],
                                        outer=spec[row, col], sharex=1)
                    else:
                        jointregression(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=[],
                                        outer=spec[row, col])

                    i += 1

    plotREGR2x4fromindex(0)
    plotREGR2x4fromindex(8)

    if doShim:
        def plotSHIM2x4fromindex(i):
            fig = plt.figure(figsize=(40, 20))

            widths = 2 * np.ones(4)
            heights = 2 * np.ones(2)
            spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                    height_ratios=heights)

            if doSNR:
                for row in range(2):
                    for col in range(4):
                        ax = fig.add_subplot(spec[row, col])

                        if (i == 0) or (i == 8):
                            blandAltmann_Shim(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], shim_v=shim_v,
                                              metname=metnames[ordernames[i]], snr_v=snr_v,
                                              outer=spec[row, col], sharey=1)
                        elif (i == 4) or (i == 12):
                            blandAltmann_Shim(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], shim_v=shim_v,
                                              metname=metnames[ordernames[i]], snr_v=snr_v, outer=spec[row, col], sharex=1,
                                              sharey=1)
                        elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                            blandAltmann_Shim(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], shim_v=shim_v,
                                              metname=metnames[ordernames[i]], snr_v=snr_v, outer=spec[row, col], sharex=1)
                        else:
                            blandAltmann_Shim(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], shim_v=shim_v,
                                              metname=metnames[ordernames[i]], snr_v=snr_v, outer=spec[row, col])

                        i += 1
            else:
                for row in range(2):
                    for col in range(4):
                        ax = fig.add_subplot(spec[row, col])

                        if (i == 0) or (i == 8):
                            blandAltmann_Shim(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], shim_v=shim_v,
                                              metname=metnames[ordernames[i]], snr_v=[],
                                              outer=spec[row, col], sharey=1)
                        elif (i == 4) or (i == 12):
                            blandAltmann_Shim(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], shim_v=shim_v,
                                              metname=metnames[ordernames[i]], snr_v=[], outer=spec[row, col], sharex=1,
                                              sharey=1)
                        elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                            blandAltmann_Shim(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], shim_v=shim_v,
                                              metname=metnames[ordernames[i]], snr_v=[], outer=spec[row, col], sharex=1)
                        else:
                            blandAltmann_Shim(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], shim_v=shim_v,
                                              metname=metnames[ordernames[i]], snr_v=[], outer=spec[row, col])

                        i += 1

        plotSHIM2x4fromindex(0)
        plotSHIM2x4fromindex(8)

    if doSNR:
        def plotSNR2x4fromindex(i):
            fig = plt.figure(figsize=(40, 20))

            widths = 2 * np.ones(4)
            heights = 2 * np.ones(2)
            spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                    height_ratios=heights)
            for row in range(2):
                for col in range(4):
                    ax = fig.add_subplot(spec[row, col])

                    if (i == 0) or (i == 8):
                        blandAltmann_SNR(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=snr_v,
                                         outer=spec[row, col], xlabel='noise', sharey=1)

                    elif (i == 4) or (i == 12):
                        blandAltmann_SNR(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=snr_v,
                                         outer=spec[row, col], xlabel='noise', sharex=1, sharey=1)
                    elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                        blandAltmann_SNR(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=snr_v,
                                         outer=spec[row, col], xlabel='noise', sharex=1)
                    else:
                        blandAltmann_SNR(fig, conc[:, ordernames[i]], pred_mM[:, ordernames[i]], metnames[ordernames[i]], snr_v=snr_v,
                                         outer=spec[row, col], xlabel='noise')

                    i += 1

        plotSNR2x4fromindex(0)
        plotSNR2x4fromindex(8)

if doSavings:
    # ----- savings of scores
    from util import save_scores_tab

    filename = 'evalTOT' + spec
    filepath = output_folder + subfolder
    save_scores_tab(filename, filepath, conc, pred_mM)

#
#
# fig = plt.figure()
# plt.plot(np.flip(X_test[0, :]))
#
# fig = plt.figure()
# plt.plot(np.flip(labels_list[2][0,:]))
#
# fig = plt.figure()
# plt.plot(np.flip(pred[2][0,:]))
#
# fig = plt.figure()
# plt.plot(np.flip(labels_list[2][0,:] - pred[2][0,:,0]), 'r')
# plt.ylim(-1000, 9000)

import pandas as pd
filename = "UNet_label"
outpath = "C:/Users/Rudy/Desktop/DL_models/"
folder = "net_type/"
subfolder = "typology/"
filepath = outpath + folder + subfolder

excelname = filename + ".xlsx"
#workbook = xlsxwriter.Workbook(filepath + excelname)
#worksheet = workbook.add_worksheet()

df = pd.DataFrame(conc)
df.to_excel(excel_writer=filepath + excelname)