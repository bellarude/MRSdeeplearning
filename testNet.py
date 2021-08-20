from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import xlsxwriter

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from data_load_norm import dataNorm, labelsNorm, ilabelsNorm, inputConcat1D, inputConcat2D, dataimport2D_md, labelsimport_md
from models import newModel

md_input = 1
flat_input = 0

if md_input == 0:
    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'

    def datatestimport():
        global dataset1D, dataset2D, nlabels, w_nlabels, snr_v, shim_v

        data_import2D   = sio.loadmat(dest_folder + 'dataset_spgram_TEST.mat')
        data_import1D = sio.loadmat(dest_folder + 'dataset_spectra_TEST.mat')
        labels_import = sio.loadmat(dest_folder + 'labels_c_TEST_abs.mat')
        snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
        readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')


        dataset2D = data_import2D['output']
        dataset1D = data_import1D['dataset_spectra']
        labels  = labels_import['labels_c']*64.5
        snr_v = snr_v['snr_v']
        shim_v = readme_SHIM['shim_v']

        #reshaping
        dataset1D = np.transpose(dataset1D, (0, 2, 1))
        labels = np.transpose(labels,(1,0))

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
net_name = "ShallowELU_hp_md_gt"
checkpoint_path = outpath + folder + net_name + ".best.hdf5"
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


def subplotconcentration(index):
    # ----------------------------------------------
    x = y_test[:, index].reshape(-1, 1)
    y = pred[:, index]
    regr.fit(x, y)
    lin = regr.predict(np.arange(0, np.max(y_test[:, index]), 0.01).reshape(-1, 1))
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)

    # ----------------------------------------------
    plt.scatter(y_test[:, index], pred[:, index], c=snr_v)
    m = np.max(y_test[:, index])
    plt.plot(np.arange(0, m, 0.01), lin, linewidth=3)
    plt.plot(np.arange(0, m, 0.01), lin - np.sqrt(mse), linewidth=2)
    plt.plot(np.arange(0, m, 0.01), lin + np.sqrt(mse), linewidth=2)
    ident = [0.0, m]
    plt.plot(ident, ident, '--', linewidth=3, color='k')

    plt.title(metnames[index] + r' - Coeff: ' + str(np.round(regr.coef_, 3)) + ' - $R^2$: ' + str(
        np.round(r_sq, 3)) + ' - mse: ' + str(np.round(mse, 3)) + ' - std: ' + str(
        np.round(np.sqrt(mse), 3))), plt.xlabel('GT'), plt.ylabel('estimates')

    return regr.coef_[0], r_sq, mse

# cc_array = []
# for i in range(17):
#     # plt.subplot(16,1,i+1)
#     fig = plt.figure(figsize=(18, 6))
#     plt.subplot(1, 3, 1)
#     cc, rr, mm = subplotconcentration(i)
#     cc_array += [cc]
#     plt.subplot(1, 3, 2)
#     plt.hist(pred[:, i], 20)
#     plt.title('PRED distribution')
#     plt.subplot(1, 3, 3)
#     plt.hist(y_test[:, i])
#     plt.title('GT distribution')

    #filename = '/content/drive/My Drive/RR/nets models/met_{0}.png'
    #filename = filename.format(i)
    #plt.savefig(filename)
    # plt.show()


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

def blandAltmann_Shim(index, met, outer=None, sharey = 0, sharex = 0):
    from matplotlib import gridspec
    from matplotlib.ticker import FormatStrFormatter
    x = y_test[:, index]
    diff = pred[:, index] - x
    shim = shim_v[:,0]

    idx_s = np.argsort(shim)
    sort = np.sort(shim)

    s_diff = diff[idx_s]
    std_diff = np.empty((y_test.shape[0], 1))
    bsize = 125

    nbins = np.int(y_test.shape[0] / bsize)

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

def blandAltmann_SNR(index, met, outer=None, xlabel='noise', sharey = 0, sharex = 0):
    from matplotlib import gridspec
    from matplotlib.ticker import FormatStrFormatter
    x = y_test[:, index]
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
    std_diff = np.empty((y_test.shape[0],1))
    bsize = 125
    nbins = np.int(y_test.shape[0]/bsize)

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

plotREGR2x4fromindex(0)
plotREGR2x4fromindex(8)


if doSNR:
    def plotSNR2x4fromindex(i):
        fig = plt.figure(figsize = (40,20))

        widths = 2*np.ones(4)
        heights = 2*np.ones(2)
        spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                  height_ratios=heights)
        for row in range(2):
            for col in range(4):
                ax = fig.add_subplot(spec[row,col])

                if (i==0) or (i==8):
                    blandAltmann_SNR(order[i], metnames[order[i]], outer=spec[row, col], xlabel='noise', sharey=1)
                elif (i == 4) or (i == 12):
                    blandAltmann_SNR(order[i], metnames[order[i]], outer=spec[row, col], xlabel='noise', sharex=1, sharey=1)
                elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                    blandAltmann_SNR(order[i], metnames[order[i]], outer=spec[row, col], xlabel='noise', sharex=1)
                else:
                    blandAltmann_SNR(order[i], metnames[order[i]], outer=spec[row, col], xlabel='noise')

                i += 1

    plotSNR2x4fromindex(0)
    plotSNR2x4fromindex(8)

if doShim:
    def plotSHIM2x4fromindex(i):
        fig = plt.figure(figsize = (40,20))

        widths = 2*np.ones(4)
        heights = 2*np.ones(2)
        spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                  height_ratios=heights)

        for row in range(2):
            for col in range(4):
                ax = fig.add_subplot(spec[row,col])

                if (i==0) or (i==8):
                    blandAltmann_Shim(order[i], metnames[order[i]], outer=spec[row, col], sharey=1)
                elif (i == 4) or (i == 12):
                    blandAltmann_Shim(order[i], metnames[order[i]], outer=spec[row, col], sharex=1, sharey=1)
                elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                    blandAltmann_Shim(order[i], metnames[order[i]], outer=spec[row, col], sharex=1)
                else:
                    blandAltmann_Shim(order[i], metnames[order[i]], outer=spec[row, col])

                i += 1

    plotSHIM2x4fromindex(0)
    plotSHIM2x4fromindex(8)


fig = plt.figure()
workbook = xlsxwriter.Workbook(outpath + folder + '/eval.xlsx')
worksheet = workbook.add_worksheet()
for i in range(16):
    c, r, m = subplotconcentration(i)
    s = 'A' + str(i * 3 + 1)
    worksheet.write(s, c)
    s = 'A' + str(i * 3 + 2)
    worksheet.write(s, r)
    s = 'A' + str(i * 3 + 3)
    worksheet.write(s, m)

workbook.close()
print('xlsx SAVED')
