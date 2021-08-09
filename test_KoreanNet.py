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
    global dataset, labels, conc, snr_v

    os.environ["KERAS_BACKEND"] = "theano"
    K.set_image_data_format('channels_last')

    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'
    # dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/'

    data_import = sio.loadmat(dest_folder + 'spectra_kor_TEST_wat.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_kor_5_TEST_NOwat.mat')
    conc_import = sio.loadmat(dest_folder + 'labels_c_TEST.mat')
    snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')

    dataset = data_import['spectra_kor']
    labels = labels_import['labels_kor_5']
    conc = conc_import['labels_c']*64.5 #mM
    snr_v = snr_v['snr_v']

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
ref2tCr = 0
NOwat = 1
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

met = 'naa'
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

    # -----------------------------------------------------------------------------
    # Korean Decoder NET
    # -----------------------------------------------------------------------------
    # l0 = maxP(convBlock2(inputs, ks[met][0], nf[met][0]), **poolargs)
    # l1 = maxP(convBlock2(l0, ks[met][1], nf[met][1]), **poolargs)
    # l2 = maxP(convBlock2(l1, ks[met][2], nf[met][2]), **poolargs)
    # l3 = maxP(convBlock2(l2, ks[met][3], nf[met][3]), **poolargs)
    # l4 = maxP(convBlock2(l3, ks[met][4], nf[met][4]), **poolargs)
    # l5 = maxP(convBlock2(l4, ks[met][5], nf[met][5]), **poolargs)
    # l6 = maxP(convBlock2(l5, ks[met][6], nf[met][6]), **poolargs)
    # l7 = maxP(convBlock2(l6, ks[met][7], nf[met][7]), **poolargs)
    # l8 = maxP(convBlock2(l7, ks[met][8], nf[met][8]), **poolargs)
    # l9 = relu(dense(nd[met], flatten(l8)))
    # outputs = dense(datapoints, l9)
    # lrate = lr[met]

    # -----------------------------------------------------------------------------
    # Korean Decoder NET - linear
    # -----------------------------------------------------------------------------
    l0 = maxP(convBlock2_lin(inputs, ks[met][0], nf[met][0]), **poolargs)
    l1 = maxP(convBlock2_lin(l0, ks[met][1], nf[met][1]), **poolargs)
    l2 = maxP(convBlock2_lin(l1, ks[met][2], nf[met][2]), **poolargs)
    l3 = maxP(convBlock2_lin(l2, ks[met][3], nf[met][3]), **poolargs)
    l4 = maxP(convBlock2_lin(l3, ks[met][4], nf[met][4]), **poolargs)
    l5 = maxP(convBlock2_lin(l4, ks[met][5], nf[met][5]), **poolargs)
    l6 = maxP(convBlock2_lin(l5, ks[met][6], nf[met][6]), **poolargs)
    l7 = maxP(convBlock2_lin(l6, ks[met][7], nf[met][7]), **poolargs)
    l8 = maxP(convBlock2_lin(l7, ks[met][8], nf[met][8]), **poolargs)
    l9 = dense(nd[met], flatten(l8))
    outputs = dense(datapoints, l9)
    lrate = lr[met]

    # -----------------------------------------------------------------------------

if unet:
    # -----------------------------------------------------------------------------
    # RR-Unet
    # -----------------------------------------------------------------------------
    # --- Define contracting layers

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
    # l9 = tran2(32, tran1(32, concat(l2, l8)))
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss = tf.keras.losses.MeanSquaredError(),
    #experimental_run_tf_function=False
    )

print(modelRR.summary())


tf.debugging.set_log_device_placement(True)
dataimport()

# nX_test, ny_test = dataNormKoreanSet(dataset, labels)
nX_test = dataset
ny_test = labels
# ny_test = dataHighlight(labels, 10)

output_folder = 'C:/Users/Rudy/Desktop/DL_models/'
subfolder = "net_type/"
net_name = "UNet_mI_NOwat"
checkpoint_path = output_folder + subfolder + net_name + ".best.hdf5"

modelRR.load_weights(checkpoint_path)
# loss = modelRR.evaluate(nX_test, ny_test, verbose=2)

pred = modelRR.predict(nX_test)  # normalized [0-1] absolute concentrations prediction

est = np.empty((pred.shape[0],1))
gt = np.empty((pred.shape[0],1))


# 780-840 for Cr_CH3 peak
if met == 'gaba':
    if ref2tCr:
        igr = [350, 650]
    else:
        igr = [500,1000]
        igr = [400, 1000] #without even water
    cidx = 5
    spincorr = 1
if met == 'naa':
    igr = [540, 620]
    cidx = 2
    spincorr = 3
if met == 'scy':
    if ref2tCr:
        igr = [860, 950]
    else:
        igr = [980, 1040]

    cidx = 14
    spincorr = 6

if NOwat:
    igr = [0, 1405]

if ref2tCr:
    iref = [760, 860]
    cref = 4
else:
    iref = [0, 200]
    cref = 16

for idx in np.arange(pred.shape[0]):
    #tcr = 780-840
    #water = 0-200
    est[idx] = np.sum(pred[idx, igr[0]:igr[1]]) / np.sum(pred[idx, iref[0]:iref[1]])
    gt[idx] = np.sum(ny_test[idx, igr[0]:igr[1]]) / np.sum(ny_test[idx, iref[0]:iref[1]])



from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()

def subplotconcentration(est,gt):
    # ----------------------------------------------
    x = gt
    y = est
    regr.fit(x, y)
    lin = regr.predict(np.arange(np.min(x), np.max(x), 0.01).reshape(-1, 1))
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)
    # ----------------------------------------------
    plt.scatter(x, y, c=snr_v)
    m = np.max(x)
    plt.plot(np.arange(np.min(x), m, 0.01), lin, linewidth=4)
    ident = [np.min(x), m]
    plt.plot(ident, ident, '--', linewidth=3, color='k')
    plt.title(r' - Coefficients: ' + str(np.round(regr.coef_, 3)) + ' - $R^2$ score: ' + str(
        np.round(r_sq, 3)) + ' - mse: ' + str(np.round(mse, 3)) + ' - std: ' + str(np.round( np.sqrt(mse),3))), plt.xlabel('GT'), plt.ylabel('estimates')

    return regr.coef_[0], r_sq, mse

subplotconcentration(est,gt)

def draw_text(ax, string, loc):
    """
    Draw two text-boxes, anchored by different corners to the upper-left
    corner of the figure.
    """
    from matplotlib.offsetbox import AnchoredText
    if loc == 'ul':
        at = AnchoredText(string,
                          loc='upper left', prop=dict(size=8), frameon=True,
                          )
    elif loc == 'ur':
        at = AnchoredText(string,
                          loc='upper right', prop=dict(size=8), frameon=True,
                          )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)


idx_plot = [312,496] #289,100

#concentrations: naa =2, tcr = 4, gaba = 5 sI = 14
fig = plt.figure()
fig.add_subplot(421)
plt.plot(np.flip(nX_test[idx_plot[0],:]))
plt.title('input')

fig.add_subplot(423)
plt.plot(np.flip(ny_test[idx_plot[0],:]))
plt.title('GT')
ax = plt.gca()
draw_text(ax, met + ':' + str(round(np.sum(ny_test[idx_plot[0], igr[0]:igr[1]]), 2))
          + '\nwater:' + str(round(np.sum(ny_test[idx_plot[0], iref[0]:iref[1]]), 2))
          + '\nR_conc:' + str(round(gt[idx_plot[0]][0], 2)), 'ul')
draw_text(ax, met + '_c:' + str(round(conc[idx_plot[0], cidx], 2))
          + '\nwater_c:' + str(round(conc[idx_plot[0], cref], 2)), 'ur')

fig.add_subplot(425)
plt.plot(np.flip(pred[idx_plot[0],:]))
plt.title('pred')
ax = plt.gca()
draw_text(ax, met + ':' + str(round(np.sum(pred[idx_plot[0], igr[0]:igr[1]]),2))
          + '\nwater:' + str(round(np.sum(pred[idx_plot[0], iref[0]:iref[1]]),2))
          + '\nR_conc:' + str(round(est[idx_plot[0]][0], 2)), 'ul')

fig.add_subplot(427)
# plt.plot(pred[0,:,0])
if kornet:
    plt.plot(np.flip(ny_test[idx_plot[0], :] - pred[idx_plot[0], :]))
elif unet:
    plt.plot(np.flip(ny_test[idx_plot[0],:] - pred[idx_plot[0],:,0]))
plt.title('GT-pred')

fig.add_subplot(422)
plt.plot(np.flip(nX_test[idx_plot[1],:]))
plt.title('input')

fig.add_subplot(424)
plt.plot(np.flip(ny_test[idx_plot[1],:]))
plt.title('GT')
ax = plt.gca()
draw_text(ax, met + ':' + str(round(np.sum(ny_test[idx_plot[1], igr[0]:igr[1]]),2))
          + '\nwater:' + str(round(np.sum(ny_test[idx_plot[1], iref[0]:iref[1]]),2))
          + '\nR_conc:' + str(round(gt[idx_plot[1]][0], 2)), 'ul')
draw_text(ax, met + '_c:' + str(round(conc[idx_plot[1], cidx], 2))
          + '\n water_c:' + str(round(conc[idx_plot[1], cref], 2)), 'ur')

fig.add_subplot(426)
plt.plot(np.flip(pred[idx_plot[1],:]))
plt.title('pred')
ax = plt.gca()
draw_text(ax, met + ':' + str(round(np.sum(pred[idx_plot[1], igr[0]:igr[1]]),2))
          + '\nwater:' + str(round(np.sum(pred[idx_plot[1], iref[0]:iref[1]]),2))
          + '\nR_conc:' + str(round(est[idx_plot[1]][0], 2)), 'ul')

fig.add_subplot(428)
if kornet:
    plt.plot(np.flip(ny_test[idx_plot[1], :] - pred[idx_plot[1], :]))
elif unet:
    plt.plot(np.flip(ny_test[idx_plot[1],:] - pred[idx_plot[1],:,0]))
plt.title('GT-pred')

##
# naa = np.sum(pred[:, 350:640], axis=1)
# tcr = np.sum(pred[:, 780:840], axis=1)
# tcr_good = tcr[tcr>7.5]
# tcr_idx = np.nonzero(tcr>7.5)
#
# gtg = np.empty((tcr_good.shape[0],1))
# estg = np.empty((tcr_good.shape[0],1))
#
# gtg[:,0] = gt[tcr_idx[0],0]
# estg[:,0] = est[tcr_idx[0],0]
# fig = plt.figure()
# subplotconcentration(estg,gtg)
#
# ##
# missing_value = set(range(0, 2500)) - set(tcr_idx[:][0])
# fig = plt.figure()
# fig.add_subplot(211)
# plt.plot(gt-est)
# plt.plot(list(missing_value), np.ones((len(missing_value), 1))*50, 'o')
# fig.add_subplot(212)
# plt.plot(1/tcr)
# # fig.add_subplot(313)
# # plt.plot(tcr)

##
naa_p = np.empty((pred.shape[0], 1))
tcr_p = np.empty((pred.shape[0], 1))
naa_gt = np.empty((pred.shape[0], 1))
tcr_gt = np.empty((pred.shape[0], 1))

if kornet:
    naa_p[:, 0] = np.sum(pred[:, igr[0]:igr[1]], axis=1)
    tcr_p[:, 0] = np.sum(pred[:, iref[0]:iref[1]], axis=1)
elif unet:
    naa_p[:,0] = np.sum(pred[:, igr[0]:igr[1],0], axis=1)
    tcr_p[:,0] = np.sum(pred[:, iref[0]:iref[1],0], axis=1)
naa_gt[:,0] = np.sum(ny_test[:, igr[0]:igr[1]], axis=1)
tcr_gt[:,0] = np.sum(ny_test[:, iref[0]:iref[1]], axis=1)

fig = plt.figure()
fig.add_subplot(211)
subplotconcentration(naa_p,naa_gt)
fig.add_subplot(212)
subplotconcentration(tcr_p,tcr_gt)

# fig = plt.figure()
# fig.add_subplot(211)
# plt.plot(naa_p, naa_gt, 'o')
# fig.add_subplot(212)
# plt.plot(tcr_p, tcr_gt, 'o')


fig = plt.figure()
plt.plot(ny_test[idx_plot[0],:])
plt.title('GT')


# conc_met= np.empty((2500, 1))
# conc_wat= np.empty((2500, 1))
# conc_met[:,0] = conc[:,cidx]
# conc_wat[:,0] = conc[:,16]
# S_wat = np.empty((2500, 1))
# S_met = np.empty((2500, 1))
# for ii in range(2500):
#     S_wat[ii,0] = np.sum(ny_test[ii,0:200])
#     S_met[ii,0] = np.sum(ny_test[ii,igr[0]:igr[1]])
#
#
# coeff = (S_met/S_wat)/(conc_met/conc_wat)


pred_mM = est*64.5/(spincorr/2)
gt_mM = gt*64.5/(spincorr/2)

pred_mM_new = (pred_mM - np.min(gt_mM)) / (np.max(gt_mM) - np.min(gt_mM)) * (np.max(conc[:,cidx]) - np.min(conc[:,cidx])) + np.min(conc[:,cidx])
gt_mM_new = (gt_mM - np.min(gt_mM)) / (np.max(gt_mM) - np.min(gt_mM)) * (np.max(conc[:,cidx]) - np.min(conc[:,cidx])) + np.min(conc[:,cidx])


fig = plt.figure()
subplotconcentration(pred_mM_new,gt_mM_new)

# fig = plt.figure()
# for iplt in range(10):
#     fig.add_subplot(311)
#     plt.plot(naa_p[iplt], naa_gt[iplt], 'o')
#     fig.add_subplot(312)
#     plt.plot(tcr_p[iplt], tcr_gt[iplt], 'o')
#     fig.add_subplot(313)
#     plt.plot(naa_p[iplt]/tcr_p[iplt], naa_gt[iplt]/tcr_gt[iplt], 'o')
#     # plt.show()
#     plt.pause(5)
#     # input("press enter")
