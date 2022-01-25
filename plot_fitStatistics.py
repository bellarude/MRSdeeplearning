from __future__ import print_function
import os
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import xlsxwriter

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from data_load_norm import labelsNorm, ilabelsNorm

import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 15

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']

amsm_gpu = 1
if amsm_gpu:
    inputFolder = r'C:\Users\Rudy\Desktop\FitAidProject+RAWdata\rawfit_areaboundLikeSIM_lorfix_testset20'
else:
    inputFolder = r'C:\Users\Rudy\Documents\WMD\01_Project 2 - Deep Learning\FitAidProject+RAWdata\fitAid_average_project_boundAreLikeSIM_fixLor'
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
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
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

plotSNR2x4fromindex(0,  y_test, ofit)
plotSNR2x4fromindex(8,  y_test, ofit)


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


plotSHIM2x4fromindex(0,  y_test, ofit)
plotSHIM2x4fromindex(8,  y_test, ofit)


# # -------------------------------------------------------------
# # exel savings
# # -------------------------------------------------------------
#
# def scores4fit(index):
#     x = y_test[:, index].reshape(-1, 1)
#     y = ofit[:, index]
#     regr.fit(x, y)
#     # lin = regr.predict(np.arange(0, np.max(gt[:, index]), 0.01).reshape(-1, 1))
#     mse = mean_squared_error(x, y)
#     r_sq = regr.score(x, y)
#
#
#     return regr.coef_[0], regr.intercept_, r_sq, mse
#
# excelname = '/' 'fit_model_A_eval.xlsx'
# workbook = xlsxwriter.Workbook(inputFolder + excelname)
# worksheet = workbook.add_worksheet()
# for i in range(16):
#     a, q, r2, mse = scores4fit(i)
#     s = 'A' + str(i * 4 + 1)
#     worksheet.write(s, a)
#     s = 'A' + str(i * 4 + 2)
#     worksheet.write(s, q)
#     s = 'A' + str(i * 4 + 3)
#     worksheet.write(s, r2)
#     s = 'A' + str(i * 4 + 4)
#     worksheet.write(s, mse)
#
# workbook.close()
# print('xlsx SAVED')


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

def plotCRLBfromindex(i):
    fig = plt.figure(figsize=(40, 20))

    widths = 2 * np.ones(4)
    heights = 2 * np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                            height_ratios=heights)

    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row, col])

            if (i == 0) or (i == 8):
                crlbeval(ordernames[i], metnames[ordernames[i]], outer=spec[row, col], sharey=1)
            elif (i == 4) or (i == 12):
                crlbeval(ordernames[i], metnames[ordernames[i]], outer=spec[row, col], sharex=1, sharey=1)
            elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                crlbeval(ordernames[i], metnames[ordernames[i]], outer=spec[row, col], sharex=1)
            else:
                crlbeval(ordernames[i], metnames[ordernames[i]], outer=spec[row, col])

            i += 1

plotCRLBfromindex(0)

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
            ax2.set_xlabel('Estimated Concentration [mM]')
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
plotCRLB3histfromindex(12)

fig = plt.figure()
crlbeval3hist(0, metnames[0])

