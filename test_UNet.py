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

os.environ["KERAS_BACKEND"] = "theano"
K.set_image_data_format('channels_last')

dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'
def dataimport(dest_folder, index):
    labels_import = sio.loadmat(dest_folder + 'labels_kor_' + str(index) + '_TEST_NOwat.mat')
    labels = labels_import['labels_kor_' + str(index)]
    return labels


data_import = sio.loadmat(dest_folder + 'spectra_kor_TEST_wat.mat')
conc_import = sio.loadmat(dest_folder + 'labels_c_TEST.mat')
snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')

dataset = data_import['spectra_kor']
conc = conc_import['labels_c'] * 64.5  # mM
snr_v = snr_v['snr_v']
shim_v = readme_SHIM['shim_v']


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

pad = 9

def newModel():
    # -----------------------------------------------------------------------------
    # RR-Unet 2xconv1
    # -----------------------------------------------------------------------------
    # --- Define contracting layers
    #
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
        experimental_run_tf_function=False
        )

    print(modelRR.summary())
    return modelRR

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']

order = [8, 10, 1, 11, 12, 2, 13, 14, 15, 4, 16, 9, 5, 17, 3, 6, 7] #to order them
ordernames = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5] #to plot them in the order we want

tf.debugging.set_log_device_placement(True)

loss = []
pred = []
labels_list = []
X_test = dataset
for idx in range(0, len(metnames)):
    labels = dataimport(dest_folder,order[idx])
    modelRR = newModel()
    y_test = labels

    output_folder = 'C:/Users/Rudy/Desktop/DL_models/'
    subfolder = "net_type/unet/"
    net_name = "UNet_" + metnames[idx] +"_NOwat"
    checkpoint_path = output_folder + subfolder + net_name + ".best.hdf5"
    modelRR.load_weights(checkpoint_path)
    a = modelRR.evaluate(X_test, y_test, verbose=0)
    b = modelRR.predict(X_test)
    loss.append(modelRR.evaluate(X_test, y_test, verbose=0))
    pred.append(modelRR.predict(X_test))  # normalized [0-1] absolute concentrations prediction
    labels_list.append(y_test)

pred_sum = np.empty((X_test.shape[0], 17))
labels_sum = np.empty((X_test.shape[0], 17))
for idx in range(0,len(metnames)):
    for smp in range(0, X_test.shape[0]):
        pred_sum[smp, idx] = np.sum(pred[idx][smp, :, 0])
        labels_sum[smp, idx] = np.sum(labels_list[idx][smp, :])

# we need to scale them in 0-1 otherwise plotting is a pain
def norm_sum(input):
    output = np.empty(input.shape)
    for idx in range(0, input.shape[1]):
        output[:, idx] = input[:, idx] / np.max(input[:,idx])
    return output

npred_sum = norm_sum(pred_sum)
nlabels_sum = norm_sum(labels_sum)

#
# -------------------------------------------------------------
# plot regression 4x4
# -------------------------------------------------------------
fig = plt.figure(figsize = (12,12))

widths = 2*np.ones(4)
heights = 2*np.ones(4)
spec = fig.add_gridspec(ncols=4, nrows=4, width_ratios=widths,
                          height_ratios=heights)

i=0
for row in range(4):
    for col in range(4):
        ax = fig.add_subplot(spec[row,col])
        subplotconcentration(i)
        i += 1

# -------------------------------------------------------------
# plot joint distribution of regression
# -------------------------------------------------------------
def jointregression(index, gt, pred, met, outer=None, sharey = 0, sharex = 0):
    from matplotlib import gridspec
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter

    # fig = plt.figure()
    if outer == None:
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = outer, width_ratios=[3, 1], height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)
    # ----------------------------------------------
    x = gt[:, index].reshape(-1, 1)
    y = pred[:, index]
    regr.fit(x, y)
    lin = regr.predict(np.arange(0, np.max(gt[:, index]), 0.01).reshape(-1, 1))
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)

    # ----------------------------------------------

    ax2 = plt.subplot(gs[2])
    p1 = ax2.scatter(gt[:, index], pred[:, index], c=snr_v, cmap='summer', label = 'observation')
    m = np.max(gt[:, index])
    ax2.plot(np.arange(0, m, 0.01), lin, color='tab:olive', linewidth=3)
    ident = [0.0, m]
    ax2.plot(ident, ident, '--', linewidth=3, color='k')
    # ax1 = plt.subplot(gs[1])

    if outer == None:
        cbaxes = inset_axes(ax2, width="30%", height="3%", loc=2)
        plt.colorbar(p1 ,cax=cbaxes, orientation ='horizontal')

    if outer != None:
        if sharex :
            ax2.set_xlabel('Ground Truth [mM]')
        if sharey:
            ax2.set_ylabel('Predictions [mM]')

    # ax2.plot(np.arange(0, m, 0.01), lin - np.sqrt(mse), color = 'tab:orange', linewidth=3)
    # ax2.plot(np.arange(0, m, 0.01), lin + np.sqrt(mse), color = 'tab:orange', linewidth=3)

    mP = np.min(y)
    MP = np.max(y)
    ax2.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax2.set_ylim(mP - (0.05*MP), MP + (0.05*MP))

    ax0 = plt.subplot(gs[0])
    ax0.set_title(met, fontweight="bold")
    sns.distplot(gt[:, index], ax=ax0, color='tab:olive')
    ax0.set_xlim(-0.250,m+0.250)
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax0.set_xlim(0 - (0.05 * m), m + (0.05 * m))

    ax3 = plt.subplot(gs[3])
    sns.distplot(y, ax=ax3, vertical=True, color='tab:olive')
    ax3.set_ylim(mP - (0.05 * MP), MP + (0.05 * MP))
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    # ax3.hist(y, bins=20, orientation =u'horizontal')

    regr.coef_[0], r_sq, mse
    # text
    textstr = '\n'.join((
        r'$a=%.2f$' % (regr.coef_[0],),
        r'$q=%.2f$' % (regr.intercept_,),
        r'$R^{2}=%.2f$' % (r_sq,),
        r'$\sigma=%.2f$' % (np.sqrt(mse),)))
    ax1 = plt.subplot(gs[1])
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    patch_t1 = mpatches.Patch(facecolor='w', label=r'$a=%.3f$' % (regr.coef_[0],))
    patch_t2 = mpatches.Patch(facecolor='w', label=r'$q=%.3f$' % (regr.intercept_,))
    patch_t3 = mpatches.Patch(facecolor='w', label=r'$R^{2}=%.3f$' % (r_sq,))
    patch_t4 = mpatches.Patch(facecolor='w', label=r'$std.=%.3f$ [mM]' % (np.sqrt(mse),))
    patch2 = mpatches.Patch(facecolor='tab:red', label='$y=ax+q$', linestyle='-')
    patch3 = mpatches.Patch(facecolor='k', label = '$y=x$', linestyle='--')
    patch4 = mpatches.Patch(facecolor = 'tab:orange', label = '$y=\pm std. \dot x$', linestyle='-')

    # ax1.legend(handles = [p1, patch2, patch3, patch4, patch_t1, patch_t2, patch_t3, patch_t4],bbox_to_anchor=(0.5, 0.3, 0.5, 0.5))

    ax1.axis('off')
    # gs.tight_layout()

def plotREGR2x4fromindex(i, labels, pred):
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row,col])
            if (i==0) or (i==8):
                jointregression(ordernames[i], labels, pred, metnames[ordernames[i]], spec[row,col], sharey=1)
            elif (i==4) or (i==12):
                jointregression(ordernames[i], labels, pred, metnames[ordernames[i]], spec[row,col], sharex=1, sharey=1)
            elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
                jointregression(ordernames[i], labels, pred, metnames[ordernames[i]], spec[row, col], sharex=1)
            else:
                jointregression(ordernames[i], labels, pred, metnames[ordernames[i]], spec[row, col])

            i += 1
plotREGR2x4fromindex(0, nlabels_sum, npred_sum)
plotREGR2x4fromindex(8, nlabels_sum, npred_sum)

def rel2wat(input):
    output = np.empty(input.shape)
    for idx in range(0,input.shape[1]):
        output[:, idx] = np.divide(input[:, idx], input[:, 16])
    return output

# this is a work around to convert it back to mM. We exploit S_labels using eventually as refernce
# labels in mM coming from matlab directly!!!
labels_mM = rel2wat(labels_sum)*64.5
pred_mM = rel2wat(pred_sum)*64.5 / labels_mM * conc

plotREGR2x4fromindex(0, conc, pred_mM)
plotREGR2x4fromindex(8, conc, pred_mM)

def blandAltmann_Shim(index, gt, pred, met, outer=None, sharey = 0, sharex = 0):
    from matplotlib import gridspec
    from matplotlib.ticker import FormatStrFormatter
    x = gt[:, index]
    diff = pred[:, index] - x
    shim = shim_v[:,0]

    idx_s = np.argsort(shim)
    sort = np.sort(shim)

    s_diff = diff[idx_s]
    std_diff = np.empty((gt.shape[0], 1))
    bsize = 125

    nbins = np.int(gt.shape[0] / bsize)

    m_bin = np.empty((nbins, 1))
    m_std = np.empty((nbins, 1))
    vlines = np.empty((nbins, 1))
    for i in range(nbins):
        bin = idx_s[i * bsize:((i + 1) * bsize) - 1]
        std_diff[i * bsize:((i + 1) * bsize)] = np.std(diff[bin])

        m_bin[i] = (np.max(sort[i * bsize:((i + 1) * bsize)]) - np.min(sort[i * bsize:((i + 1) * bsize)])) / 2 + np.min(sort[i * bsize:((i + 1) * bsize)])
        vlines[i] = np.max(sort[i * bsize:((i + 1) * bsize)])
        m_std[i] = np.std(diff[bin])

    if outer == None:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3],
                          wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer, height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)

    ax0 = plt.subplot(gs[0])
    ax0.plot(sort, std_diff[:, 0], 'lightgray', linewidth=0.5)
    ax0.plot(m_bin[:, 0], m_std[:, 0], 'tab:green')
    ax0.scatter(m_bin[:, 0], m_std[:, 0], c='tab:green', s=10)
    for i in np.arange(vlines.shape[0]):
        ax0.axvline(vlines[i, 0], 0, 1, color='gray', alpha=0.5, linewidth=0.5)
    ax0.xaxis.set_visible(False)
    ax0.set_title(met, fontweight="bold")

    mm = np.mean(std_diff[:,0])
    ax0.set_ylim(mm - 0.8*mm, mm + 0.8*mm)

    ax1 = plt.subplot(gs[1])
    ax1.scatter(sort, s_diff, c=snr_v, cmap='summer')
    ax1.plot(sort, np.zeros((len(sort))), 'k--')


    if outer != None:
        if sharex:
            ax1.set_xlabel('shim [Hz]')
        if sharey:
            ax0.set_ylabel('$\sigma (\Delta )$ [mM]')
            ax1.set_ylabel('$\Delta$ [mM]')
    else:
        ax1.set_xlabel('shim [Hz]')
        ax0.set_ylabel('$\sigma (\Delta )$ [mM]')
        ax1.set_ylabel('$\Delta$ [mM]')

    ax0.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

def blandAltmann_SNR(index, gt, pred, met, outer=None, xlabel='noise', sharey = 0, sharex = 0):
    from matplotlib import gridspec
    from matplotlib.ticker import FormatStrFormatter
    x = gt[:, index]
    diff = pred[:, index] - x
    # rel_noise = np.multiply(1/x, 1/snr_v[:,0])

    snr = snr_v[:,0]
    noise = 1/snr_v[:,0]
    noise_over_gt = noise/x

    if xlabel=='noise':
        idx_s = np.argsort(noise)
        sort = np.sort(noise)
    elif xlabel == 'snr':
        idx_s = np.argsort(snr)
        sort = np.sort(snr)
    else:
        idx_s = np.argsort(noise_over_gt)
        sort = np.sort(noise_over_gt)

    s_diff = diff[idx_s]
    std_diff = np.empty((gt.shape[0],1))
    bsize = 125
    nbins = np.int(gt.shape[0]/bsize)

    m_bin = np.empty((nbins,1))
    m_std = np.empty((nbins,1))
    vlines = np.empty((nbins,1))
    for i in range(nbins):
        bin = idx_s[i*bsize:((i+1)*bsize)]
        m_bin[i] = (np.max(sort[i*bsize:((i+1)*bsize)]) - np.min(sort[i*bsize:((i+1)*bsize)]))/2 + np.min(sort[i*bsize:((i+1)*bsize)])
        vlines[i] = np.max(sort[i*bsize:((i+1)*bsize)])
        # m_bin[i] = np.mean(sort[i * bsize:((i + 1) * bsize)])
        std_diff[i*bsize:((i+1)*bsize)] = np.std(diff[bin])
        m_std[i] = np.std(diff[bin])

    if outer == None:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3],
                          wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer, height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)

    ax0 = plt.subplot(gs[0])
    ax0.plot(sort, std_diff[:, 0], 'lightgray', linewidth=0.5)
    ax0.plot(m_bin[:,0], m_std[:,0], 'tab:green')
    ax0.scatter(m_bin[:, 0], m_std[:, 0], c='tab:green', s=10)
    for i in np.arange(len(m_bin)):
        ax0.axvline(vlines[i,0], 0, 1, color='gray', alpha=0.5, linewidth=0.5)
    ax0.xaxis.set_visible(False)
    ax0.set_title(met, fontweight="bold")


    ax1 = plt.subplot(gs[1])
    ax1.scatter(sort, s_diff, c=snr_v, cmap='summer')
    ax1.plot(sort, np.zeros((len(sort))), 'k--')

    if outer != None:
        if sharex:
            if xlabel == 'noise':
                ax1.set_xlabel('1/SNR')
            elif xlabel == 'snr':
                ax1.set_xlabel('SNR')
            else:
                ax1.set_xlabel('1/(SNR*GT)')
        if sharey:
            ax0.set_ylabel('$\sigma (\Delta )$ [mM]')
            ax1.set_ylabel('$\Delta$ [mM]')
    else:
        if xlabel == 'noise':
            ax1.set_xlabel('noise')
        elif xlabel == 'snr':
            ax1.set_xlabel('SNR')
        else:
            ax1.set_xlabel('noise/GT')
        ax0.set_ylabel('$\sigma (\Delta )$ [mM]')
        ax1.set_ylabel('$\Delta$ [mM]')

    ax0.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

def plotSNR2x4fromindex(i, labels, pred):
    fig = plt.figure(figsize = (40,20))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row,col])

            if (i==0) or (i==8):
                blandAltmann_SNR(ordernames[i], labels, pred, metnames[ordernames[i]], outer=spec[row, col], xlabel='noise', sharey=1)
            elif (i == 4) or (i == 12):
                blandAltmann_SNR(ordernames[i], labels, pred, metnames[ordernames[i]], outer=spec[row, col], xlabel='noise', sharex=1, sharey=1)
            elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                blandAltmann_SNR(ordernames[i], labels, pred, metnames[ordernames[i]], outer=spec[row, col], xlabel='noise', sharex=1)
            else:
                blandAltmann_SNR(ordernames[i], labels, pred, metnames[ordernames[i]], outer=spec[row, col], xlabel='noise')

            i += 1

plotSNR2x4fromindex(0, conc, pred_mM)
plotSNR2x4fromindex(8, conc, pred_mM)


def plotSHIM2x4fromindex(i, labels, pred):
    fig = plt.figure(figsize=(40, 20))

    widths = 2 * np.ones(4)
    heights = 2 * np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                            height_ratios=heights)

    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row, col])

            if (i == 0) or (i == 8):
                blandAltmann_Shim(ordernames[i], labels, pred, metnames[ordernames[i]], outer=spec[row, col], sharey=1)
            elif (i == 4) or (i == 12):
                blandAltmann_Shim(ordernames[i], labels, pred, metnames[ordernames[i]], outer=spec[row, col], sharex=1, sharey=1)
            elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                blandAltmann_Shim(ordernames[i], labels, pred, metnames[ordernames[i]], outer=spec[row, col], sharex=1)
            else:
                blandAltmann_Shim(ordernames[i], labels, pred, metnames[ordernames[i]], outer=spec[row, col])

            i += 1


plotSHIM2x4fromindex(0, conc, pred_mM)
plotSHIM2x4fromindex(8, conc, pred_mM)

# savings


from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import xlsxwriter
regr = linear_model.LinearRegression()


def eval(index, labels, pred):
    # ----------------------------------------------
    x = labels[:, index].reshape(-1, 1)
    y = pred[:, index]
    regr.fit(x, y)
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)

    return regr.coef_[0], r_sq, mse

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/unet/"
workbook = xlsxwriter.Workbook(outpath + folder + '/eval.xlsx')
worksheet = workbook.add_worksheet()

for i in range(16):
    c, r, m = eval(i, conc, pred_mM)
    s = 'A' + str(i * 3 + 1)
    worksheet.write(s, c)
    s = 'A' + str(i * 3 + 2)
    worksheet.write(s, r)
    s = 'A' + str(i * 3 + 3)
    worksheet.write(s, m)

workbook.close()
print('xlsx SAVED')