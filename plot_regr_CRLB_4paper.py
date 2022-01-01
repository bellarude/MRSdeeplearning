from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import xlsxwriter

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from data_load_norm import dataNorm, labelsNorm, ilabelsNorm, inputConcat1D, inputConcat2D, dataimport2D_md, labelsimport_md
from models import newModel

import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 15

md_input = 0
flat_input = 0

if md_input == 0:
    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/test dataset/'

    def datatestimport():
        global dataset1D, dataset2D, nlabels, w_nlabels, snr_v, shim_v

        data_import2D   = sio.loadmat(dest_folder + 'dataset_spgram_TEST.mat')
        data_import1D = sio.loadmat(dest_folder + 'dataset_spectra_TEST.mat')
        labels_import = sio.loadmat(dest_folder + 'labels_c_TEST.mat')
        snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
        readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')


        dataset2D = data_import2D['output']
        dataset1D = data_import1D['dataset_spectra']
        labels  = labels_import['labels_c']*64.5
        snr_v = snr_v['snr_v']
        shim_v = readme_SHIM['shim_v']

        #reshaping
        dataset1D = np.transpose(dataset1D, (0, 2, 1))
        dataset1D = inputConcat1D(dataset1D)
        # labels = np.transpose(labels,(1,0))

        # nndataset_rs = dataNorm(ndataset_rs)
        # nndataset_rs = ndataset_rs

        nlabels, w_nlabels = labelsNorm(labels)

        return dataset1D, dataset2D, nlabels, w_nlabels, snr_v, shim_v


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

if flat_input:
    dataset1D_flat = inputConcat1D(dataset1D)
    dataset2D_flat = inputConcat2D(dataset2D)


outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"
subfolder = "typology/"
net_name = "ShallowELU_hp"
checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
# model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowInception_fact_v2')

model.load_weights(checkpoint_path)
loss = model.evaluate(dataset2D, nlabels, verbose=2)

pred_abs = model.predict(dataset2D)  # normalized [0-1] absolute concentrations prediction
pred_un = np.empty(pred_abs.shape)  # un-normalized absolute concentrations
pred = np.empty(pred_abs.shape)  # relative un normalized concentrations (referred to water prediction)

pred_un = ilabelsNorm(pred_abs, w_nlabels)
y_test = ilabelsNorm(nlabels, w_nlabels)

for i in range(17):
    pred[:, i] = pred_un[:, i] / pred_un[:, 16] * 64.5
    y_test[:, i] = y_test[:, i] / y_test[:, 16] * 64.5


regr = linear_model.LinearRegression()

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']

# -------------------------------------------------------------
# plot joint distribution of regression
# -------------------------------------------------------------
def jointregression(index, met, outer=None, sharey = 0, sharex = 0):
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
    x = y_test[:, index].reshape(-1, 1)
    y = pred[:, index]
    regr.fit(x, y)
    lin = regr.predict(np.arange(0, np.max(y_test[:, index]), 0.01).reshape(-1, 1))
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)

    snr1idx = np.nonzero(snr_v[:, 0] < 16.7)
    snr2idx = np.nonzero((snr_v[:, 0] >= 16.7) & (snr_v[:, 0] < 28.4))
    snr3idx = np.nonzero(snr_v[:, 0] >= 28.4)

    x1 = x[snr1idx[0]]
    x2 = x[snr2idx[0]]
    x3 = x[snr3idx[0]]
    y1 = y[snr1idx[0]]
    y2 = y[snr2idx[0]]
    y3 = y[snr3idx[0]]

    mse1 = mean_squared_error(x1, y1)
    mse2 = mean_squared_error(x2, y2)
    mse3 = mean_squared_error(x3, y3)
    # ----------------------------------------------

    ax2 = plt.subplot(gs[2])
    p1 = ax2.scatter(y_test[:, index], pred[:, index], c=snr_v, cmap='summer', label = 'observation')
    m = np.max(y_test[:, index])
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
    sns.distplot(y_test[:, index], ax=ax0, color='tab:olive')
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
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
             verticalalignment='top', bbox=props)

    # text
    textstr2 = '\n'.join((
        r'$\sigma 1=%.2f$' % (np.sqrt(mse1),),
        r'$\sigma 2=%.2f$' % (np.sqrt(mse2),),
        r'$\sigma 3=%.2f$' % (np.sqrt(mse3),)))
    # ax1 = plt.subplot(gs[1])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, verticalalignment='top', bbox=props)

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

order = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5] # to order metabolites plot from good to bad

doSNR = 0
doShim = 1


# single plots to check
# fig = plt.figure()
# jointregression(0, metnames[0])
# fig = plt.figure()
# blandAltmann_SNR(3, metnames[3], None, 'noisegt')

# fig = plt.figure()
# blandAltmann_SNR(3, metnames[3], None, 'noise')
# fig = plt.figure()
# blandAltmann_SNR(3, metnames[3], None, 'snr')
# fig = plt.figure()
# blandAltmann_Shim(3, metnames[3])


# -------------------------------------------------------------
# plot regression 2x4
# -------------------------------------------------------------
def plotREGR2x4fromindex(i):
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row,col])
            if (i==0) or (i==8):
                jointregression(order[i], metnames[order[i]], spec[row,col], sharey=1)
            elif (i==4) or (i==12):
                jointregression(order[i], metnames[order[i]], spec[row,col], sharex=1, sharey=1)
            elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
                jointregression(order[i], metnames[order[i]], spec[row, col], sharex=1)
            else:
                jointregression(order[i], metnames[order[i]], spec[row, col])

            i += 1
#
# def plotREGR_paper_fromindex(i):
#     fig = plt.figure(figsize = (10,40))
#
#     widths = 2*np.ones(3)
#     heights = 2*np.ones(4)
#     spec = fig.add_gridspec(ncols=3, nrows=4, width_ratios=widths,
#                               height_ratios=heights)
#     for row in range(4):
#         for col in range(3):
#             ax = fig.add_subplot(spec[row,col])
#             if (i==0) or (i==8):
#                 jointregression(order[i], metnames[order[i]], spec[row,col], sharey=1)
#             elif (i==4) or (i==12):
#                 jointregression(order[i], metnames[order[i]], spec[row,col], sharex=1, sharey=1)
#             elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
#                 jointregression(order[i], metnames[order[i]], spec[row, col], sharex=1)
#             else:
#                 jointregression(order[i], metnames[order[i]], spec[row, col])
#
#             i += 1

# plotREGR_paper_fromindex(0)
plotREGR2x4fromindex(0)
plotREGR2x4fromindex(8)

inputFolder = r'C:\Users\Rudy\Desktop\FitAidProject+RAWdata\rawfit_areaboundLikeSIM_lorfix_testset20'
fileName = r'\fitStatistics.xlsx'

fit = pd.read_excel(inputFolder + fileName, sheet_name= 'area', header=None)
crlb = pd.read_excel(inputFolder + fileName, sheet_name= 'c_area', header=None)
orderFit = [15,11,10,0,16,1,2,4,3,6,5,7,9,12,13,14,17]
ordernames = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5] #to plot them in the order we want

ofit = np.empty((fit.shape[0], fit.shape[1]-1))
ocrlb = np.empty((fit.shape[0], fit.shape[1]))
for idx in range(0,len(orderFit)):
    ofit[:, idx] = fit.values[:, orderFit[idx]]/64.5*fit.values[:, 17]
    ocrlb[:,idx] = crlb.values[:, orderFit[idx]]

dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/test dataset/'
def simulationimport():
    global  nlabels, w_nlabels, snr_v, shim_v

    labels_import = sio.loadmat(dest_folder + 'labels_c_TEST.mat')
    snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
    readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')

    labels = labels_import['labels_c'] * 64.5
    snr_v = snr_v['snr_v']
    shim_v = readme_SHIM['shim_v']

    # reshaping
    # labels = np.transpose(labels, (1, 0))
    nlabels, w_nlabels = labelsNorm(labels)

    return  nlabels, w_nlabels, snr_v, shim_v

simulationimport()
y_test = ilabelsNorm(nlabels, w_nlabels)
for i in range(17):
    y_test[:, i] = y_test[:, i] / y_test[:, 16] * 64.5


regr = linear_model.LinearRegression()

# -------------------------------------------------------------
# plot joint distribution of regression
# -------------------------------------------------------------
def jointregressionfit(index, gt, pred, met, outer=None, sharey = 0, sharex = 0):
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

    snr1idx = np.nonzero(snr_v[:, 0] < 16.7)
    snr2idx = np.nonzero((snr_v[:, 0] >= 16.7) & (snr_v[:, 0] < 28.4))
    snr3idx = np.nonzero(snr_v[:, 0] >= 28.4)

    x1 = x[snr1idx[0]]
    x2 = x[snr2idx[0]]
    x3 = x[snr3idx[0]]
    y1 = y[snr1idx[0]]
    y2 = y[snr2idx[0]]
    y3 = y[snr3idx[0]]

    mse1 = mean_squared_error(x1, y1)
    mse2 = mean_squared_error(x2, y2)
    mse3 = mean_squared_error(x3, y3)
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
            ax2.set_ylabel('Estimates[mM]')

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
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
             verticalalignment='top', bbox=props)

    # text
    textstr2 = '\n'.join((
        r'$\sigma 1=%.2f$' % (np.sqrt(mse1),),
        r'$\sigma 2=%.2f$' % (np.sqrt(mse2),),
        r'$\sigma 3=%.2f$' % (np.sqrt(mse3),)))
    # ax1 = plt.subplot(gs[1])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, verticalalignment='top', bbox=props)

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
                jointregressionfit(ordernames[i], labels, pred, metnames[ordernames[i]], spec[row,col], sharey=1)
            elif (i==4) or (i==12):
                jointregressionfit(ordernames[i], labels, pred, metnames[ordernames[i]], spec[row,col], sharex=1, sharey=1)
            elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
                jointregressionfit(ordernames[i], labels, pred, metnames[ordernames[i]], spec[row, col], sharex=1)
            else:
                jointregressionfit(ordernames[i], labels, pred, metnames[ordernames[i]], spec[row, col])

            i += 1

plotREGR2x4fromindex(0, y_test, ofit)
plotREGR2x4fromindex(8, y_test, ofit)


# -------------------------------------------------------------
# plot CRLB
# -------------------------------------------------------------
def crlbeval(index, met, outer=None, sharey = 0, sharex = 0):
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

    minX = np.min(ofit[:, index])
    maxX = np.max(ofit[:, index])
    minY = np.min(ocrlb[:, index])
    maxY = np.max(ocrlb[:, index])

    ax2 = plt.subplot(gs[2])
    ax2.set_xlim(minX - (0.05 * maxX), maxX + (0.05 * maxX))
    ax2.set_ylim(minY - (0.05 * maxY), maxY + (0.05 * maxY))
    p1 = ax2.scatter(ofit[:,index], ocrlb[:, index], c=snr_v, cmap='summer', label = 'observation')

    if outer == None:
        cbaxes = inset_axes(ax2, width="30%", height="3%", loc=2)
        plt.colorbar(p1 ,cax=cbaxes, orientation ='horizontal')

    if outer != None:
        if sharex :
            ax2.set_xlabel('Predicted Concentration [mM]')
        if sharey:
            ax2.set_ylabel('Absolute CRLB [mM]')

    ax0 = plt.subplot(gs[0])
    ax0.set_title(met, fontweight="bold")
    sns.distplot(ofit[:,index], ax=ax0, color='tab:olive')
    ax0.set_xlim(minX - (0.05 * maxX), maxX + (0.05 * maxX))
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    ax3 = plt.subplot(gs[3])
    sns.distplot(ocrlb[:, index], ax=ax3, vertical=True, color='tab:olive')
    ax3.set_ylim(minY - (0.05 * maxY), maxY + (0.05 * maxY))
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    # ax3.hist(y, bins=20, orientation =u'horizontal')

    # text
    # textstr = '\n'.join((
    #     r'$a=%.2f$' % (regr.coef_[0],),
    #     r'$q=%.2f$' % (regr.intercept_,),
    #     r'$R^{2}=%.2f$' % (r_sq,),
    #     r'$\sigma=%.2f$' % (np.sqrt(mse),)))
    # ax1 = plt.subplot(gs[1])
    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    #
    # patch_t1 = mpatches.Patch(facecolor='w', label=r'$a=%.3f$' % (regr.coef_[0],))
    # patch_t2 = mpatches.Patch(facecolor='w', label=r'$q=%.3f$' % (regr.intercept_,))
    # patch_t3 = mpatches.Patch(facecolor='w', label=r'$R^{2}=%.3f$' % (r_sq,))
    # patch_t4 = mpatches.Patch(facecolor='w', label=r'$std.=%.3f$ [mM]' % (np.sqrt(mse),))
    # patch2 = mpatches.Patch(facecolor='tab:red', label='$y=ax+q$', linestyle='-')
    # patch3 = mpatches.Patch(facecolor='k', label = '$y=x$', linestyle='--')
    # patch4 = mpatches.Patch(facecolor = 'tab:orange', label = '$y=\pm std. \dot x$', linestyle='-')

    # ax1.legend(handles = [p1, patch2, patch3, patch4, patch_t1, patch_t2, patch_t3, patch_t4],bbox_to_anchor=(0.5, 0.3, 0.5, 0.5))

    # ax1.axis('off')
    # gs.tight_layout()

#-------------------------------------------------------------
# plot CRLB w/ 4 histograms
# -------------------------------------------------------------
def crlbeval3hist(index, met, outer=None, sharey = 0, sharex = 0):
    from matplotlib import gridspec
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import scipy.stats as st
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter

    # fig = plt.figure()
    if outer == None:
        gs = fig.add_gridspec(2, 4, width_ratios=[3, 1, 1, 1], height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec = outer, width_ratios=[3, 1, 1, 1], height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)

    minX = np.min(ofit[:, index])
    maxX = np.max(ofit[:, index])
    minY = np.min(ocrlb[:, index])
    maxY = np.max(ocrlb[:, index])

    ax2 = plt.subplot(gs[4])
    ax2.set_xlim(minX - (0.05 * maxX), maxX + (0.05 * maxX))
    ax2.set_ylim(minY - (0.05 * maxY), maxY + (0.05 * maxY))
    p1 = ax2.scatter(ofit[:,index], ocrlb[:, index], c=snr_v, cmap='summer', label = 'observation')

    if outer == None:
        cbaxes = inset_axes(ax2, width="30%", height="3%", loc=2)
        plt.colorbar(p1 ,cax=cbaxes, orientation ='horizontal')

    if outer != None:
        if sharex :
            ax2.set_xlabel('Estimates [mM]')
        if sharey:
            ax2.set_ylabel('Absolute CRLB [mM]')

    ax0 = plt.subplot(gs[0])
    ax0.set_title(met, fontweight="bold")
    sns.distplot(ofit[:,index], ax=ax0, color='tab:olive')
    ax0.set_xlim(minX - (0.05 * maxX), maxX + (0.05 * maxX))
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    snr1idx = np.nonzero(snr_v[:, 0] < 16.7)
    snr2idx = np.nonzero((snr_v[:, 0] >= 16.7) & (snr_v[:, 0] < 28.4))
    snr3idx = np.nonzero(snr_v[:, 0] >= 28.4)

    ocrlb_1 = ocrlb[snr1idx[0], index]
    y, x = np.histogram(ocrlb_1, 50)
    mcrlb_1 = x[np.argmax(y)]
    mu_crlb_1 = np.mean(ocrlb_1)

    ocrlb_2 = ocrlb[snr2idx[0], index]
    y, x = np.histogram(ocrlb_2, 50)
    mcrlb_2 = x[np.argmax(y)]

    ocrlb_3 = ocrlb[snr3idx[0], index]
    y, x = np.histogram(ocrlb_3, 50)
    mcrlb_3 = x[np.argmax(y)]

    textstr = '\n'.join((
        r'$\mu:%.2f$' % (mu_crlb_1),
        r'$Mo:%.2f$' % (mcrlb_1,)))

    ax3 = plt.subplot(gs[5])
    sns.distplot(ocrlb_1, ax=ax3, vertical=True, color='darkgreen')
    ax3.axhline(mcrlb_1, 0, 1, color='k', alpha=0.5, ls='--')
    ax3.axhline(mu_crlb_1, 0, 1, color='gray', alpha=0.5, ls='--')
    ax3.set_ylim(minY - (0.05 * maxY), maxY + (0.05 * maxY))
    # ax3.xaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
    for i, label in enumerate(ax3.axes.get_xticklabels()):
        if i < len(ax3.axes.get_xticklabels()) - 1:
            label.set_visible(False)
    ax3.axes.set_xlabel(textstr, fontsize=11)
    ax3.xaxis.set_label_position('top')

    ax4 = plt.subplot(gs[6])
    sns.distplot(ocrlb_2, ax=ax4, vertical=True, color='limegreen')
    ax4.set_ylim(minY - (0.05 * maxY), maxY + (0.05 * maxY))
    ax4.axhline(mcrlb_2, 0, 1, color='k', alpha=0.5, ls='--')
    # ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
    for i, label in enumerate(ax4.axes.get_xticklabels()):
        if i < len(ax4.axes.get_xticklabels()) - 1:
            label.set_visible(False)
    ax4.axes.set_xlabel(r'$Mo:%.2f$' % mcrlb_2, fontsize=11 )
    ax4.xaxis.set_label_position('top')

    ax5 = plt.subplot(gs[7])
    sns.distplot(ocrlb_3, ax=ax5, vertical=True, color='lawngreen')
    ax5.set_ylim(minY - (0.05 * maxY), maxY + (0.05 * maxY))
    ax5.axhline(mcrlb_3, 0, 1, color='k', alpha=0.5, ls='--')
    # ax5.xaxis.set_visible(False)
    ax5.yaxis.set_visible(False)
    ax5.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))
    for i, label in enumerate(ax5.axes.get_xticklabels()):
        if i < len(ax5.axes.get_xticklabels()) - 1:
            label.set_visible(False)
    ax5.axes.set_xlabel(r'$Mo:%.2f$' % mcrlb_3, fontsize=11 )
    ax5.xaxis.set_label_position('top')


def plotCRLB3histfromindex(i):
    fig = plt.figure(figsize=(30, 10))

    widths = 2 * np.ones(3)
    heights = 2 * np.ones(2)
    spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                            height_ratios=heights)

    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(spec[row, col])

            if (i == 0) or (i == 6) or (i==12):
                crlbeval3hist(ordernames[i], metnames[ordernames[i]], outer=spec[row, col], sharey=1)
            elif (i == 3) or (i == 9) or (i == 15):
                crlbeval3hist(ordernames[i], metnames[ordernames[i]], outer=spec[row, col], sharex=1, sharey=1)
            elif (i == 4) or (i == 5) or (i == 10) or (i == 11) or (i == 16) or (i == 17):
                crlbeval3hist(ordernames[i], metnames[ordernames[i]], outer=spec[row, col], sharex=1)
            else:
                crlbeval3hist(ordernames[i], metnames[ordernames[i]], outer=spec[row, col])

            i += 1

plotCRLB3histfromindex(0)
plotCRLB3histfromindex(6)
# plotCRLB3histfromindex(12)


def plot_manuscript(i):
    fig = plt.figure(figsize = (20,5))
    # widths = 2 * np.ones(3)
    heights = 2 * np.ones(1)
    spec = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[2,2,3],
                            height_ratios=heights)
    for col in range(3):
        ax = fig.add_subplot(spec[0,col])

        if col == 0:
            jointregression(order[i], metnames[order[i]], spec[0, col], sharey=1, sharex=1)
        elif col == 1:
            jointregressionfit(ordernames[i], y_test, ofit, metnames[ordernames[i]], spec[0, col], sharey=1, sharex=1)
        else:
            crlbeval3hist(ordernames[i], metnames[ordernames[i]], outer=spec[0, col], sharey=1, sharex=1)

plot_manuscript(0)
plot_manuscript(4)
plot_manuscript(11)
plot_manuscript(14)
plot_manuscript(2)

# CRLB as function(GT) values, with different SNR counted in

from scipy.signal import savgol_filter

set_low_size = np.shape(np.nonzero(snr_v[:,0] < 16.7)[0])[0]
set_mid_size = np.shape(np.nonzero((snr_v[:,0] >= 16.7) & (snr_v[:,0] < 28.4))[0])[0]
set_high_size = np.shape(np.nonzero(snr_v[:,0] >= 28.4)[0])[0]

y_test_sort = np.zeros(y_test.shape)
snr_v_sort = np.zeros(y_test.shape)
snr_sort_idx_low = np.zeros((set_low_size, 17), dtype=int)
snr_sort_idx_mid = np.zeros((set_mid_size, 17), dtype=int)
snr_sort_idx_high = np.zeros((set_high_size, 17), dtype=int)
ofit_sort = np.zeros(y_test.shape)
ocrlb_sort = np.zeros(y_test.shape)
ocrlb_interp = np.zeros(y_test.shape)
bias2 = np.zeros(y_test.shape)
bias = np.zeros(y_test.shape)
for i in range(17):
    idx_gt = np.argsort(y_test[:, i])
    y_test_sort[:, i] = y_test[idx_gt, i]
    snr_v_sort[:,i] = snr_v[idx_gt,0]
    ofit_sort[:,i] = ofit[idx_gt,i]
    bias2[:,i] = np.sqrt(np.power(ofit_sort[:, i]-y_test_sort[:, i], 2))
    bias[:, i] = ofit_sort[:, i] - y_test_sort[:, i]
    ocrlb_sort[:,i] = ocrlb[idx_gt, i]
    ocrlb_interp[:, i] = savgol_filter(ocrlb_sort[:, i], 51, 3)
    # std_p_sort_y_test[:, i] = std_p[idx_gt, i]
    # std_intrp_y_test[:, i] = savgol_filter(std_p_sort_y_test[:, i], 51, 3)
    # pred_det_sort_aslabel[:,i] = pred_det[idx_gt, i]
    snr_sort_idx_low[:,i] = np.nonzero(snr_v_sort[:,i] < 16.7)[0]
    snr_sort_idx_mid[:,i] = np.nonzero((snr_v_sort[:,i] >= 16.7) & (snr_v_sort[:,i] < 28.4))[0]
    snr_sort_idx_high[:,i] = np.nonzero(snr_v_sort[:,i] >= 28.4)[0]


fig = plt.figure()
plt.plot(y_test_sort[:,0], ofit_sort[:,0])
fig = plt.figure()
plt.plot(y_test_sort[:,0], ocrlb_sort[:,0])
plt.plot(y_test_sort[:,0], ocrlb_interp[:,0])

fig = plt.figure()
plt.subplot(231)
plt.plot(y_test_sort[snr_sort_idx_low[:, 0], 0], bias2[snr_sort_idx_low[:, 0],0])
plt.subplot(232)
plt.plot(y_test_sort[snr_sort_idx_mid[:, 0], 0], bias2[snr_sort_idx_mid[:, 0],0])
plt.subplot(233)
plt.plot(y_test_sort[snr_sort_idx_high[:, 0], 0], bias2[snr_sort_idx_high[:, 0],0])
plt.subplot(234)
plt.plot(y_test_sort[snr_sort_idx_low[:, 0], 0], ocrlb_sort[snr_sort_idx_low[:, 0],0])
plt.subplot(235)
plt.plot(y_test_sort[snr_sort_idx_mid[:, 0], 0], ocrlb_sort[snr_sort_idx_mid[:, 0],0])
plt.subplot(236)
plt.plot(y_test_sort[snr_sort_idx_high[:, 0], 0], ocrlb_sort[snr_sort_idx_high[:, 0],0])

def bias_crlb_fit(index):
    from matplotlib import gridspec
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1],
                                               wspace=0.2, hspace=0.2)

    ax0 = plt.subplot(gs[0])
    ax0.plot(y_test_sort[snr_sort_idx_low[:, index], index], bias[snr_sort_idx_low[:, index], index])
    ax0.plot(y_test_sort[snr_sort_idx_low[:, index], index],
             savgol_filter(bias[snr_sort_idx_low[:, index], index], 51, 3), linewidth=3)

    # text
    textstr = '\n'.join((
        r'$\mu =%.2f$' % (np.mean(bias[snr_sort_idx_low[:, index], index]),),
        r'$\sigma =%.2f$' % (np.sqrt(mean_squared_error(y_test_sort[snr_sort_idx_low[:, index],index], ofit_sort[snr_sort_idx_low[:, index],index])),),))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax0.text(0.05, 0.95, textstr, transform=ax0.transAxes, verticalalignment='top', bbox=props)


    ax1 = plt.subplot(gs[1])
    ax1.plot(y_test_sort[snr_sort_idx_mid[:, index], index], bias[snr_sort_idx_mid[:, index], index])
    ax1.plot(y_test_sort[snr_sort_idx_mid[:, index], index],
             savgol_filter(bias[snr_sort_idx_mid[:, index], index], 51, 3), linewidth=3)

    # text
    textstr = '\n'.join((
        r'$\mu =%.2f$' % (np.mean(bias[snr_sort_idx_mid[:, index], index]),),
        r'$\sigma =%.2f$' % (np.sqrt(mean_squared_error(y_test_sort[snr_sort_idx_mid[:, index],index], ofit_sort[snr_sort_idx_mid[:, index],index])),),))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, verticalalignment='top', bbox=props)

    ax2 = plt.subplot(gs[2])
    ax2.plot(y_test_sort[snr_sort_idx_high[:, index], index], bias[snr_sort_idx_high[:, index], index])
    ax2.plot(y_test_sort[snr_sort_idx_high[:, index], index],
             savgol_filter(bias[snr_sort_idx_high[:, index], index], 51, 3), linewidth=3)

    # text
    textstr = '\n'.join((
        r'$\mu =%.2f$' % (np.mean(bias[snr_sort_idx_high[:, index], index]),),
        r'$\sigma =%.2f$' % (np.sqrt(mean_squared_error(y_test_sort[snr_sort_idx_high[:, index], index],
                                                      ofit_sort[snr_sort_idx_high[:, index], index])),),))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, verticalalignment='top', bbox=props)

    ax3 = plt.subplot(gs[3])
    ax3.plot(y_test_sort[snr_sort_idx_low[:, index], index], ocrlb_sort[snr_sort_idx_low[:, index], index])
    ax3.plot(y_test_sort[snr_sort_idx_low[:, index], index],
             savgol_filter(ocrlb_sort[snr_sort_idx_low[:, index], index], 51, 3), linewidth=3)

    # text
    textstr = '\n'.join((
        r'$\mu =%.2f$' % (np.mean(ocrlb_sort[snr_sort_idx_low[:, index], index]),),
        r'$\sigma =%.2f$' % (np.std(ocrlb_sort[snr_sort_idx_low[:, index], index]),)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, verticalalignment='top', bbox=props)

    ax4 = plt.subplot(gs[4])
    ax4.plot(y_test_sort[snr_sort_idx_mid[:, index], index], ocrlb_sort[snr_sort_idx_mid[:, index], index])
    ax4.plot(y_test_sort[snr_sort_idx_mid[:, index], index],
             savgol_filter(ocrlb_sort[snr_sort_idx_mid[:, index], index], 51, 3), linewidth=3)

    # text
    textstr = '\n'.join((
        r'$\mu =%.2f$' % (np.mean(ocrlb_sort[snr_sort_idx_mid[:, index], index]),),
        r'$\sigma =%.2f$' % (np.std(ocrlb_sort[snr_sort_idx_mid[:, index], index]),)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, verticalalignment='top', bbox=props)

    ax5 = plt.subplot(gs[5])
    ax5.plot(y_test_sort[snr_sort_idx_high[:, index], index], ocrlb_sort[snr_sort_idx_high[:, index], index])
    ax5.plot(y_test_sort[snr_sort_idx_high[:, index], index],
             savgol_filter(ocrlb_sort[snr_sort_idx_high[:, index], index], 51, 3), linewidth=3)

    # text
    textstr = '\n'.join((
        r'$\mu =%.2f$' % (np.mean(ocrlb_sort[snr_sort_idx_high[:, index], index]),),
        r'$\sigma =%.2f$' % (np.std(ocrlb_sort[snr_sort_idx_high[:, index], index]),)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, verticalalignment='top', bbox=props)

    ax3.set_xlabel('Ground Truth [mM]')
    ax4.set_xlabel('Ground Truth [mM]')
    ax5.set_xlabel('Ground Truth [mM]')
    ax0.set_ylabel('bias [mM]')
    ax3.set_ylabel('CRLB [mM]')
    ax0.set_title(metnames[index] + ' - low SNR', fontweight="bold")
    ax1.set_title(metnames[index] + ' - mid SNR', fontweight="bold")
    ax2.set_title(metnames[index] + ' - high SNR', fontweight="bold")

    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

bias_crlb_fit(ordernames[0])
bias_crlb_fit(ordernames[6])
bias_crlb_fit(ordernames[11])
bias_crlb_fit(ordernames[14])

def bias_crlb_fit_all(index):
    from matplotlib import gridspec
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter
    fig = plt.figure(figsize=(4.5, 5))
    gs = fig.add_gridspec(2, 1, width_ratios=[1], height_ratios=[1, 1],
                                               wspace=0.2, hspace=0.2)

    ax0 = plt.subplot(gs[0])
    ax0.plot(y_test_sort[:, index], bias[:, index])
    ax0.plot(y_test_sort[:, index],
             savgol_filter(bias[:, index], 51, 3), linewidth=3)

    # text
    textstr = '\n'.join((
        r'$\mu =%.2f$' % (np.mean(bias[:, index]),),
        r'$\sigma =%.2f$' % (np.sqrt(mean_squared_error(y_test_sort[:,index], ofit_sort[:,index])),),))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax0.text(0.05, 0.95, textstr, transform=ax0.transAxes, verticalalignment='top', bbox=props)

    ax1 = plt.subplot(gs[1])
    ax1.plot(y_test_sort[:, index], ocrlb_sort[:, index])
    ax1.plot(y_test_sort[:, index],
             savgol_filter(ocrlb_sort[:, index], 51, 3), linewidth=3)

    # text
    textstr = '\n'.join((
        r'$\mu =%.2f$' % (np.mean(ocrlb_sort[:, index]),),
        r'$\sigma =%.2f$' % (np.std(ocrlb_sort[:, index]),)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, verticalalignment='top', bbox=props)


    ax1.set_xlabel('Ground Truth [mM]')
    ax0.set_ylabel('bias [mM]')
    ax1.set_ylabel('CRLB [mM]')
    ax0.set_title(metnames[index], fontweight="bold")

    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

bias_crlb_fit_all(ordernames[0])
bias_crlb_fit_all(ordernames[6])
bias_crlb_fit_all(ordernames[8])
bias_crlb_fit_all(ordernames[14])
