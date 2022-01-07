from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from data_load_norm import dataNorm, labelsNorm, ilabelsNorm, inputConcat1D, inputConcat2D, dataimport2D_md, labelsNormREDdataset, labelsimport_md
from models import newModel
from util_plot import jointregression, blandAltmann_Shim, blandAltmann_SNR, sigma_distro, sigma_vs_gt

import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 15

input1d = 0
md_input = 1
flat_input = 0
test_diff_conc_bounds = 0

if md_input == 0:
    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/test dataset/'
    # dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'

    def datatestimport():
        global dataset1D, dataset2D, nlabels, w_nlabels, snr_v, shim_v


        # labels = np.transpose(labels,(1,0))
        if test_diff_conc_bounds == 0:
            snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
            readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')
            labels_import = sio.loadmat(dest_folder + 'labels_c_TEST.mat')

            labels = labels_import['labels_c'] * 64.5
            snr_v = snr_v['snr_v']
            shim_v = readme_SHIM['shim_v']

            nlabels, w_nlabels = labelsNorm(labels)
        else:
            snr_v = sio.loadmat(dest_folder + 'snr_v_TEST_0208')
            readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST_0mu.mat')
            labels_import = sio.loadmat(dest_folder + 'labels_c_TEST_0mu.mat')

            labels = labels_import['labels_c'] * 64.5
            snr_v = snr_v['snr_v']
            shim_v = readme_SHIM['shim_v']

            labels_import_orig = sio.loadmat(dest_folder + 'labels_c_TEST.mat')
            labels_orig = labels_import_orig['labels_c'] * 64.5
            nlabels, w_nlabels = labelsNormREDdataset(labels, labels_orig)

        if input1d:
            data_import1D = sio.loadmat(dest_folder + 'dataset_spectra_TEST.mat')
            dataset1D = data_import1D['dataset_spectra']
            # reshaping
            dataset1D = np.transpose(dataset1D, (0, 2, 1))
            dataset1D = inputConcat1D(dataset1D)

            return dataset1D, nlabels, w_nlabels, snr_v, shim_v
        else:
            if test_diff_conc_bounds == 0:
                data_import2D = sio.loadmat(dest_folder + 'dataset_spgram_TEST.mat')
            else:
                data_import2D = sio.loadmat(dest_folder + 'dataset_spgram_TEST_0mu.mat')
            dataset2D = data_import2D['output']

            if flat_input:
                dataset2D = inputConcat2D(dataset2D)

            return dataset2D, nlabels, w_nlabels, snr_v, shim_v

    datatestimport()
else:

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

    snr_v = sio.loadmat(folder + 'snr_v')
    readme_SHIM = sio.loadmat(folder + 'shim_v.mat')
    snr_v_tot = snr_v['snr_v']
    shim_v_tot = readme_SHIM['shim_v']
    snr_v = snr_v_tot[18000:20000, :]
    shim_v = shim_v_tot[18000:20000, :]

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33/labels/'
    filenames = ['labels_c_1.mat',
                 'labels_c_2.mat',
                 'labels_c_3.mat',
                 'labels_c_4.mat']
    keyname = 'labels_c'

    y_train, y_val, y_test = labelsimport_md(folder, filenames, keyname)

    nlabels, w_nlabels = labelsNorm(y_test)
    dataset2D = X_test

model = newModel(dim='2D', type='dualpath_net', subtype='ShallowELU')


# def mu_sigma(output):
#     p = 17  # y_train.shape[1]
#     mu = output[0:p]
#     T1 = output[p:2 * p]
#     T2 = output[2 * p:]
#     ones = tf.ones((p, p), dtype=tf.float32)
#     mask_a = tf.linalg.band_part(ones, 0, -1)
#     mask_b = tf.linalg.band_part(ones, 0, 0)
#     mask = tf.subtract(mask_a, mask_b)
#     zero = tf.constant(0, dtype=tf.float32)
#     non_zero = tf.not_equal(mask, zero)
#     indices = tf.where(non_zero)
#     T2 = tf.sparse.SparseTensor(indices, T2, dense_shape=tf.cast((p, p),
#                                                                  dtype=tf.int64))
#     T2 = tf.sparse.to_dense(T2)
#     T1 = tf.linalg.diag(T1)
#     sigma = T1 + T2
#     return mu, sigma

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"
subfolder = "" #"typology/"
net_name = "ShallowNet-2D2c-ESMRMB_dualPath_wmse_noiseless"
checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

model.load_weights(checkpoint_path)
loss = model.evaluate(dataset2D, nlabels, verbose=2)

output_dualPath = model.predict(dataset2D)  # normalized [0-1] absolute concentrations prediction
pred_abs = output_dualPath[:,0:17]

pred_un = np.empty(pred_abs.shape)  # un-normalized absolute concentrations
pred = np.empty(pred_abs.shape)  # relative un normalized concentrations (referred to water prediction)

pred_un = ilabelsNorm(pred_abs, w_nlabels)
y_test = ilabelsNorm(nlabels, w_nlabels)

for i in range(17):
    pred[:, i] = pred_un[:, i] / pred_un[:, 16] * 64.5
    y_test[:, i] = y_test[:, i] / y_test[:, 16] * 64.5


logsigma = output_dualPath[:,17:]
sigma = np.exp(logsigma) #NB: this refers to diagonal of covariance matrix = sigma^2 or capital sigma !!
std = np.sqrt(sigma)

sigma_un = ilabelsNorm(sigma, w_nlabels)
std_un = np.sqrt(sigma_un)

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']

order = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5] # to order metabolites plot from good to bad

doSNR = 0
doShim = 1

# single plots to check
# fig = plt.figure()
# jointregression(fig, y_test[:, 0], pred[:, 0], metnames[0], snr_v=snr_v)

if doSNR:
    plotREGR2x4fromindex(0, y_test, pred, order, metnames, snr_v)
    plotREGR2x4fromindex(8, y_test, pred, order, metnames, snr_v)
else:
    plotREGR2x4fromindex(0, y_test, pred, order, metnames, snr=[])
    plotREGR2x4fromindex(8, y_test, pred, order, metnames, snr=[])


if doShim:
    if doSNR:
        plotSHIM2x4fromindex(0, y_test, pred, order, metnames, shim_v, snr_v)
        plotSHIM2x4fromindex(8, y_test, pred, order, metnames, shim_v, snr_v)
    else:
        plotSHIM2x4fromindex(0, y_test, pred, order, metnames, shim_v, snr=[])
        plotSHIM2x4fromindex(8, y_test, pred, order, metnames, shim_v, snr=[])


# ----- savings of scores
from util import save_scores_tab
filename = net_name
filepath = outpath + folder + subfolder
save_scores_tab(filename, filepath, y_test, pred)

# ----- plot distribution of precision per metabolite
def plotSIGMA2x4fromindex(i):
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row,col])
            if (i==0) or (i==8):
                sigma_distro(fig, std[:, order[i]], metnames[order[i]], outer=spec[row,col], sharey=1)
            elif (i==4) or (i==12):
                sigma_distro(fig, std[:, order[i]], metnames[order[i]], outer=spec[row,col], sharex=1, sharey=1)
            elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
                sigma_distro(fig, std[:, order[i]], metnames[order[i]], outer=spec[row, col], sharex=1)
            else:
                sigma_distro(fig, std[:, order[i]], metnames[order[i]], outer=spec[row, col])

            i += 1

plotSIGMA2x4fromindex(0)
plotSIGMA2x4fromindex(8)

# ----- plot precision as function of ground truth value of concentration
y_test_sort = np.zeros(y_test.shape)
pred_sort = np.zeros(pred.shape)
std_un_sort = np.zeros(std_un.shape)
for i in range(17):
    idx = np.argsort(y_test[:,i])
    y_test_sort[:,i] = y_test[idx,i]
    pred_sort[:,i] = pred[idx,i]
    std_un_sort[:, i] = std_un[idx, i]

def plotGTvsSIGMA2x4fromindex(i):
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row,col])
            if (i==0) or (i==8):
                sigma_vs_gt(fig, y_test_sort[:, order[i]], std_un_sort[:, order[i]], metnames[order[i]], outer=spec[row,col], sharey=1)
            elif (i==4) or (i==12):
                sigma_vs_gt(fig, y_test_sort[:, order[i]], std_un_sort[:, order[i]], metnames[order[i]], outer=spec[row,col], sharex=1, sharey=1)
            elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
                sigma_vs_gt(fig, y_test_sort[:, order[i]], std_un_sort[:, order[i]], metnames[order[i]], outer=spec[row, col], sharex=1)
            else:
                sigma_vs_gt(fig, y_test_sort[:, order[i]], std_un_sort[:, order[i]], metnames[order[i]], outer=spec[row, col])

            i += 1

plotGTvsSIGMA2x4fromindex(0)
plotGTvsSIGMA2x4fromindex(8)
