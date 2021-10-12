from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
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


input1d = 0
md_input = 0
flat_input = 0

if md_input == 0:
    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/test dataset/'
    # dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'

    def datatestimport():
        global dataset1D, dataset2D, nlabels, w_nlabels, snr_v, shim_v

        snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
        readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')
        labels_import = sio.loadmat(dest_folder + 'labels_c_TEST.mat')

        labels  = labels_import['labels_c']*64.5
        snr_v = snr_v['snr_v']
        shim_v = readme_SHIM['shim_v']
        # labels = np.transpose(labels,(1,0))
        nlabels, w_nlabels = labelsNorm(labels)

        if input1d:
            data_import1D = sio.loadmat(dest_folder + 'dataset_spectra_TEST.mat')
            dataset1D = data_import1D['dataset_spectra']
            # reshaping
            dataset1D = np.transpose(dataset1D, (0, 2, 1))
            dataset1D = inputConcat1D(dataset1D)

            return dataset1D, nlabels, w_nlabels, snr_v, shim_v
        else:
            data_import2D = sio.loadmat(dest_folder + 'dataset_spgram_TEST.mat')
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

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"
subfolder = "typology/"
net_name = "ShallowELU_hp"
checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowInception_fact_v2')

model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp_MC')
model.load_weights(checkpoint_path)

def predict_prob(data, model, num_samples):
    pred_stack = np.zeros((nlabels.shape[0], nlabels.shape[1], num_samples))
    for j in range(num_samples):

        # loss.append(model.evaluate(data, nlabels, verbose=2))

        pred_abs = model.predict(data)  # normalized [0-1] absolute concentrations prediction
        # pred_un = np.empty(pred_abs.shape)  # un-normalized absolute concentrations
        pred = np.empty(pred_abs.shape)  # relative un normalized concentrations (referred to water prediction)
        pred_un = ilabelsNorm(pred_abs, w_nlabels)

        for i in range(17):
            pred[:, i] = pred_un[:, i] / pred_un[:, 16] * 64.5

        pred_stack[:,:,j] = pred
    return pred_stack

p = predict_prob(dataset2D,model, 100)

y_test = ilabelsNorm(nlabels, w_nlabels)
for i in range(17):
    y_test[:, i] = y_test[:, i] / y_test[:, 16] * 64.5

fig = plt.figure()
plt.hist(p[0,0,:])

fig = plt.figure()
sns.distplot(p[0,0,:], color='tab:olive')

std_p = np.std(p, axis=2)
mean_p = np.mean(p, axis=2)

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
mean_p_sort = np.zeros(mean_p.shape)
std_p_sort = np.zeros(mean_p.shape)
std_intrp = np.zeros(mean_p.shape)
for i in range(17):
    idx = np.argsort(mean_p[:,i])
    mean_p_sort[:,i] = mean_p[idx,i]
    std_p_sort[:, i] = std_p[idx, i]
    # f = interp1d(np.linspace(0, mean_p_sort[-1,i], 2500), std_p_sort[:, i], kind='cubic')
    std_intrp[:,i] = savgol_filter(std_p_sort[:, i], 51, 3)
    # std_intrp[:,i] = f.y

regr = linear_model.LinearRegression()

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']
order = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5]


def jointregression(index, met, labels, preds, preds_sort, std_intrp, snr,  outer=None, sharey = 0, sharex = 0):
    from matplotlib import gridspec
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter

    # fig = plt.figure()
    if outer == None:
        gs = fig.add_gridspec(2, 3, width_ratios=[3, 1, 1], height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec = outer, width_ratios=[3, 1, 1], height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)
    # ----------------------------------------------
    x = labels[:, index].reshape(-1, 1)
    y = preds[:,index]

    # x = y_test[:, index].reshape(-1, 1)
    # y = mean_p[:, index]
    regr.fit(x, y)
    lin = regr.predict(np.arange(0, np.max(labels[:, index]), 0.01).reshape(-1, 1))
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)

    # ----------------------------------------------

    ax3 = plt.subplot(gs[3])
    p1 = ax3.scatter(labels[:, index], preds[:, index], c=snr, cmap='summer', label = 'observation')
    m = np.max(labels[:, index])
    ax3.plot(np.arange(0, m, 0.01), lin, color='tab:olive', linewidth=3)
    ident = [0.0, m]
    ax3.plot(ident, ident, '--', linewidth=3, color='k')
    # ax1 = plt.subplot(gs[1])

    # ax2.plot(np.arange(0, m, 0.01), lin - np.sqrt(mse), color = 'tab:orange', linewidth=3)
    # ax2.plot(np.arange(0, m, 0.01), lin + np.sqrt(mse), color = 'tab:orange', linewidth=3)

    mP = np.min(y)
    MP = np.max(y)
    ax3.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax3.set_ylim(mP - (0.05*MP), MP + (0.05*MP))

    ax0 = plt.subplot(gs[0])
    ax0.set_title(met, fontweight="bold")
    sns.distplot(labels[:, index], ax=ax0, color='tab:olive')
    ax0.set_xlim(-0.250,m+0.250)
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax0.set_xlim(0 - (0.05 * m), m + (0.05 * m))

    ax4 = plt.subplot(gs[4])
    sns.distplot(y, ax=ax4, vertical=True, color='tab:olive')
    ax4.set_ylim(mP - (0.05 * MP), MP + (0.05 * MP))
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    ax5 = plt.subplot(gs[5])
    ax5.set_ylim(mP - (0.05*MP), MP + (0.05*MP))
    ax5.plot(std_intrp[:,index], preds_sort[:,index], 'green')
    ax5.yaxis.set_visible(False)

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
    if outer == None:
        cbaxes = inset_axes(ax3, width="30%", height="3%", loc=2)
        plt.colorbar(p1 ,cax=cbaxes, orientation ='horizontal')
        ax3.set_xlabel('Ground Truth [mM]')
        ax5.set_xlabel('$\sigma$ [mM]')
        ax3.set_ylabel('Avg. Pred. [mM]')

    if outer != None:
        if sharex:
            ax3.set_xlabel('Ground Truth [mM]')
            ax5.set_xlabel('$\sigma$ [mM]')
        if sharey:
            ax3.set_ylabel('Avg. Pred. [mM]')
    ax1.axis('off')
    # gs.tight_layout()

fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test, mean_p, mean_p_sort, std_intrp, snr_v, outer=None, sharey = 1, sharex = 1)


# -------------------------------------------------------------
# plot regression 2x3
# -------------------------------------------------------------
def plotREGR2x3fromindex(i):
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(3)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(spec[row,col])
            if (i==0) or (i==6) or (i==12):
                jointregression(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, snr_v, spec[row,col], sharey=1)
            elif (i==3) or (i==9) or (i==15):
                jointregression(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, snr_v, spec[row,col], sharex=1, sharey=1)
            elif (i==4) or (i==10) or (i==16) or (i==5) or (i==11):
                jointregression(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, snr_v, spec[row, col], sharex=1)
            else:
                jointregression(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, snr_v, spec[row, col])

            i += 1

plotREGR2x3fromindex(0)
plotREGR2x3fromindex(6)
# plotREGR2x3fromindex(12)

# SNR plots
snr1idx = np.nonzero(snr_v[:, 0] < 16.7)
snr2idx = np.nonzero((snr_v[:, 0] >= 16.7) & (snr_v[:, 0] < 28.4))
snr3idx = np.nonzero(snr_v[:, 0] >= 28.4)

snr_v1 = snr_v[snr1idx]
snr_v2 = snr_v[snr2idx]
snr_v3 = snr_v[snr3idx]

y_test1 = y_test[snr1idx[0], :]
y_test2 = y_test[snr2idx[0], :]
y_test3 = y_test[snr3idx[0], :]

mean_p1 = mean_p[snr1idx[0], :]
mean_p2 = mean_p[snr2idx[0], :]
mean_p3 = mean_p[snr3idx[0], :]

std_p1 = std_p[snr1idx[0], :]
std_p2 = std_p[snr2idx[0], :]
std_p3 = std_p[snr3idx[0], :]

mean_p1_sort = np.zeros(mean_p1.shape)
mean_p2_sort = np.zeros(mean_p2.shape)
mean_p3_sort = np.zeros(mean_p3.shape)
std_p1_sort = np.zeros(mean_p1.shape)
std_p2_sort = np.zeros(mean_p2.shape)
std_p3_sort = np.zeros(mean_p3.shape)
std_intrp1 = np.zeros(mean_p1.shape)
std_intrp2 = np.zeros(mean_p2.shape)
std_intrp3 = np.zeros(mean_p3.shape)

for i in range(17):
    idx = np.argsort(mean_p1[:,i])
    mean_p1_sort[:,i] = mean_p1[idx,i]
    std_p1_sort[:, i] = std_p1[idx, i]
    std_intrp1[:,i] = savgol_filter(std_p1_sort[:, i], 51, 3)

    idx2 = np.argsort(mean_p2[:, i])
    mean_p2_sort[:, i] = mean_p2[idx2, i]
    std_p2_sort[:, i] = std_p2[idx2, i]
    std_intrp2[:, i] = savgol_filter(std_p2_sort[:, i], 51, 3)

    idx3 = np.argsort(mean_p3[:, i])
    mean_p3_sort[:, i] = mean_p3[idx3, i]
    std_p3_sort[:, i] = std_p3[idx3, i]
    std_intrp3[:, i] = savgol_filter(std_p3_sort[:, i], 51, 3)

fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test1, mean_p1, mean_p1_sort, std_intrp1, snr_v1, outer=None, sharey = 1, sharex = 1)
fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test2, mean_p2, mean_p2_sort, std_intrp2, snr_v2, outer=None, sharey = 1, sharex = 1)
fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test3, mean_p3, mean_p3_sort, std_intrp3, snr_v3, outer=None, sharey = 1, sharex = 1)

# -------------------------------------------------------------
# plot regression 2x3
# -------------------------------------------------------------
def plotREGR2x3SNR(index1, index2):
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(3)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                              height_ratios=heights)
    ax = fig.add_subplot(spec[0, 0])
    jointregression(order[index1], metnames[order[index1]], y_test1, mean_p1, mean_p1_sort, std_intrp1, snr_v1, ax, sharey=1)
    ax = fig.add_subplot(spec[0, 1])
    jointregression(order[index1], metnames[order[index1]], y_test2, mean_p2, mean_p2_sort, std_intrp2, snr_v2, ax,
                    sharex=0, sharey=0)
    ax = fig.add_subplot(spec[0, 2])
    jointregression(order[index1], metnames[order[index1]], y_test3, mean_p3, mean_p3_sort, std_intrp3, snr_v3, ax,
                    sharex=0, sharey=0)

    ax = fig.add_subplot(spec[1, 0])
    jointregression(order[index2], metnames[order[index2]], y_test1, mean_p1, mean_p1_sort, std_intrp1, snr_v1, ax,
                    sharex=1, sharey=1)
    ax = fig.add_subplot(spec[1, 1])
    jointregression(order[index2], metnames[order[index2]], y_test2, mean_p2, mean_p2_sort, std_intrp2, snr_v2, ax,
                    sharex=1, sharey=0)
    ax = fig.add_subplot(spec[1, 2])
    jointregression(order[index2], metnames[order[index2]], y_test3, mean_p3, mean_p3_sort, std_intrp3, snr_v3, ax,
                    sharex=1, sharey=0)


plotREGR2x3SNR(0, 11)
plotREGR2x3SNR(4, 15)