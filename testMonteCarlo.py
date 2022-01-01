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
    # return dataset2D, nlabels, w_nlabels, snr_v, shim_v

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

model_det = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
model_det.load_weights(checkpoint_path)

#deterministic model
pred_abs_det = model_det.predict(dataset2D)  # normalized [0-1] absolute concentrations prediction
pred_det = np.empty(pred_abs_det.shape)  # relative un normalized concentrations (referred to water prediction)
pred_un_det = ilabelsNorm(pred_abs_det, w_nlabels)

for i in range(17):
    pred_det[:, i] = pred_un_det[:, i] / pred_un_det[:, 16] * 64.5



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
mean_p_sort_aslabel = np.zeros(mean_p.shape)
std_intrp = np.zeros(mean_p.shape)
y_test_sort = np.zeros(mean_p.shape)
std_p_sort_y_test = np.zeros(mean_p.shape)
std_intrp_y_test = np.zeros(mean_p.shape)
bindu = 20
y_test_bin_sort_std = np.zeros(y_test.shape)
y_test_bin_sort_sigma = np.zeros(y_test.shape)
pred_det_sort_aslabel = np.zeros(y_test.shape)
mean_p_bias = np.zeros(y_test.shape)
mean_p_bias_interp = np.zeros(y_test.shape)
accuracy_mc = np.zeros(y_test.shape)
accuracy_mc_interp = np.zeros(y_test.shape)
accuracy_data_bin = np.zeros(y_test.shape)

for i in range(17):
    idx = np.argsort(mean_p[:,i])
    mean_p_sort[:,i] = mean_p[idx,i]
    std_p_sort[:, i] = std_p[idx, i]
    std_intrp[:,i] = savgol_filter(std_p_sort[:, i], 51, 3)

    idx_gt = np.argsort(y_test[:, i])
    y_test_sort[:, i] = y_test[idx_gt, i]
    std_p_sort_y_test[:, i] = std_p[idx_gt, i]
    std_intrp_y_test[:, i] = savgol_filter(std_p_sort_y_test[:, i], 51, 3)
    pred_det_sort_aslabel[:,i] = pred_det[idx_gt, i]

    ## data uncertainty std
    mean_p_sort_aslabel[:,i] = mean_p[idx_gt, i]
    #ISMRM bias definition
    # mean_p_bias[:,i] = np.sqrt(np.power(mean_p_sort_aslabel[:,i] - y_test_sort[:,i],2))
    # after ISMRM22 bias definition (with sign information)
    mean_p_bias[:, i] = mean_p_sort_aslabel[:, i] - y_test_sort[:, i]
    mean_p_bias_interp[:,i] = savgol_filter(mean_p_bias[:, i], 51, 3)

    from sklearn.metrics import mean_squared_error
    for b in range(bindu):
        idx_start = np.int((y_test.shape[0] / bindu) * b)
        idx_stop = np.int(((y_test.shape[0] / bindu) * (b + 1)) - 1)
        # y_test_bin_sort_std[idx_start:idx_stop+1, i] = np.std(mean_p_sort_aslabel[idx_start:idx_stop+1, i]) #spread of point spread function
        y_test_bin_sort_std[idx_start:idx_stop + 1, i] = np.std(
            pred_det_sort_aslabel[idx_start:idx_stop + 1, i])  # spread of point spread function (deterministic prediction)
        # ISMRM22 bias definition
        #y_test_bin_sort_sigma[idx_start:idx_stop + 1, i] = np.sqrt(mean_squared_error(pred_det_sort_aslabel[idx_start:idx_stop + 1, i], y_test_sort[idx_start:idx_stop + 1, i]))
        # after ISMRM22 bias definition (with sign information)
        y_test_bin_sort_sigma[idx_start:idx_stop + 1, i] = np.mean(pred_det_sort_aslabel[idx_start:idx_stop + 1, i] - y_test_sort[idx_start:idx_stop + 1, i])


regr = linear_model.LinearRegression()

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']
order = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5]


def jointregression(index, met, labels, preds, preds_sort, std_intrp, preds_sort_label, std_intrp_label, snr,  outer=None, sharey = 0, sharex = 0):
    from matplotlib import gridspec
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter

    # fig = plt.figure()
    if outer == None:
        gs = fig.add_gridspec(3, 3, width_ratios=[3, 1.5, 1.5], height_ratios=[1.5, 1.5, 3],
                                               wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec = outer, width_ratios=[3, 1.5, 1.5], height_ratios=[1.5, 1.5, 3],
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

    ax3 = plt.subplot(gs[6])
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

    ax0 = plt.subplot(gs[3])
    sns.distplot(labels[:, index], ax=ax0, color='tab:olive')
    ax0.set_xlim(-0.250,m+0.250)
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax0.set_xlim(0 - (0.05 * m), m + (0.05 * m))

    ax4 = plt.subplot(gs[7])
    sns.distplot(y, ax=ax4, vertical=True, color='tab:olive')
    ax4.set_ylim(mP - (0.05 * MP), MP + (0.05 * MP))
    ax4.xaxis.set_visible(False)
    ax4.yaxis.set_visible(False)
    ax4.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    # ax5 = plt.subplot(gs[8])
    # ax5.set_ylim(mP - (0.05*MP), MP + (0.05*MP))
    # ax5.plot(std_intrp[:,index], preds_sort[:,index], 'green')
    # ax5.yaxis.set_visible(False)
    # ax5.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    ax6 = plt.subplot(gs[0])
    ax6.set_title(met, fontweight="bold")
    ax6.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax6.plot(preds_sort_label[:, index], std_intrp_label[:, index],  'green')
    ax6.plot(preds_sort_label[:, index], y_test_bin_sort_std[:, index], 'red')
    ax6.xaxis.set_visible(False)
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    regr.coef_[0], r_sq, mse
    # text
    textstr = '\n'.join((
        r'$a=%.2f$' % (regr.coef_[0],),
        r'$q=%.2f$' % (regr.intercept_,),
        r'$R^{2}=%.2f$' % (r_sq,),
        r'$\sqrt{MSE}=%.2f$' % (np.sqrt(mse),)))
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
        # ax5.set_xlabel('$\sigma$ [mM]')
        ax6.set_ylabel('$\sigma$ [mM]')
        ax3.set_ylabel('Avg. Pred. [mM]')

    if outer != None:
        if sharex:
            ax3.set_xlabel('Ground Truth [mM]')
            # ax5.set_xlabel('$\sigma$ [mM]')
        if sharey:
            ax6.set_ylabel('$\sigma$ [mM]')
            ax3.set_ylabel('Avg. Pred. [mM]')
    ax1.axis('off')
    # gs.tight_layout()

fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, outer=None, sharey = 1, sharex = 1)
fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test, mean_p, mean_p_sort, std_p_sort, y_test_sort, std_p_sort_y_test, snr_v, outer=None, sharey = 1, sharex = 1)

# -------------------------------------------------------------
# plot regression 2x4
# -------------------------------------------------------------
def plotREGR2x4fromindex(i):
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights,
                            top=0.925,
                            bottom=0.08,
                            left=0.075,
                            right=0.985,
                            hspace=0.17,
                            wspace=0.05)
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row,col])
            if (i==0) or (i==8):
                jointregression(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, spec[row,col], sharey=1)
            elif (i==4) or (i==12):
                jointregression(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, spec[row,col], sharex=1, sharey=1)
            elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
                jointregression(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, spec[row, col], sharex=1)
            else:
                jointregression(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, spec[row, col])

            i += 1

plotREGR2x4fromindex(0)
plotREGR2x4fromindex(8)



def jointregression_mcSTD(index, met, labels, preds, preds_sort, std_intrp, preds_sort_label, std_intrp_label, snr,  outer=None, sharey = 0, sharex = 0):
    from matplotlib import gridspec
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter

    # fig = plt.figure()
    if outer == None:
        gs = fig.add_gridspec(4, 1, width_ratios=[3], height_ratios=[1, 1, 1, 1],
                                               wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec = outer, width_ratios=[3], height_ratios=[1, 1, 1, 1],
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

    # ax3 = plt.subplot(gs[4])
    # p1 = ax3.scatter(labels[:, index], preds[:, index], c=snr, cmap='summer', label = 'observation')
    # # p1 = ax3.scatter(labels[:, index], preds[:, index], c='darkgreen', cmap='summer', label='observation')
    m = np.max(labels[:, index])
    # ax3.plot(np.arange(0, m, 0.01), lin, color='tab:olive', linewidth=3)
    # ident = [0.0, m]
    # ax3.plot(ident, ident, '--', linewidth=3, color='k')
    # ax1 = plt.subplot(gs[1])

    # ax2.plot(np.arange(0, m, 0.01), lin - np.sqrt(mse), color = 'tab:orange', linewidth=3)
    # ax2.plot(np.arange(0, m, 0.01), lin + np.sqrt(mse), color = 'tab:orange', linewidth=3)

    # mP = np.min(y)
    # MP = np.max(y)
    # ax3.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    # ax3.set_ylim(mP - (0.05*MP), MP + (0.05*MP))

    # ax0 = plt.subplot(gs[3])
    # sns.distplot(labels[:, index], ax=ax0, color='tab:olive')
    # ax0.set_xlim(-0.250,m+0.250)
    # ax0.xaxis.set_visible(False)
    # ax0.yaxis.set_visible(False)
    # ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    # ax0.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    #
    # ax4 = plt.subplot(gs[7])
    # sns.distplot(y, ax=ax4, vertical=True, color='tab:olive')
    # ax4.set_ylim(mP - (0.05 * MP), MP + (0.05 * MP))
    # ax4.xaxis.set_visible(False)
    # ax4.yaxis.set_visible(False)
    # ax4.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    # ax5 = plt.subplot(gs[8])
    # ax5.set_ylim(mP - (0.05*MP), MP + (0.05*MP))
    # ax5.plot(std_intrp[:,index], preds_sort[:,index], 'green')
    # ax5.yaxis.set_visible(False)
    # ax5.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    ax5 = plt.subplot(gs[0])
    ax5.set_title(met, fontweight="bold")
    ax5.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax5.plot(preds_sort_label[:, index],  mean_p_bias_interp[:, index], 'lightgreen')
    ax5.xaxis.set_visible(False)
    ax5.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    ax6 = plt.subplot(gs[1])
    ax6.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax6.plot(preds_sort_label[:, index], std_intrp_label[:, index],  'green')
    # ax6.plot(preds_sort_label[:, index], y_test_bin_sort_sigma[:, index], 'blue')
    ax6.xaxis.set_visible(False)
    ax6.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    ax8 = plt.subplot(gs[2])
    ax8.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax8.plot(preds_sort_label[:, index], y_test_bin_sort_sigma[:, index], 'dodgerblue')
    ax8.xaxis.set_visible(False)
    ax8.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))

    ax7 = plt.subplot(gs[3])
    ax7.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax7.plot(preds_sort_label[:, index], y_test_bin_sort_std[:, index], 'darkblue')
    # ax7.xaxis.set_visible(False)
    ax7.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))



    # regr.coef_[0], r_sq, mse
    # # text
    # textstr = '\n'.join((
    #     r'$a=%.2f$' % (regr.coef_[0],),
    #     r'$q=%.2f$' % (regr.intercept_,),
    #     r'$R^{2}=%.2f$' % (r_sq,),
    #     r'$\sqrt{MSE}=%.2f$' % (np.sqrt(mse),)))
    # ax1 = plt.subplot(gs[1])
    # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    # ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
    #         verticalalignment='top', bbox=props)

    # patch_t1 = mpatches.Patch(facecolor='w', label=r'$a=%.3f$' % (regr.coef_[0],))
    # patch_t2 = mpatches.Patch(facecolor='w', label=r'$q=%.3f$' % (regr.intercept_,))
    # patch_t3 = mpatches.Patch(facecolor='w', label=r'$R^{2}=%.3f$' % (r_sq,))
    # patch_t4 = mpatches.Patch(facecolor='w', label=r'$std.=%.3f$ [mM]' % (np.sqrt(mse),))
    # patch2 = mpatches.Patch(facecolor='tab:red', label='$y=ax+q$', linestyle='-')
    # patch3 = mpatches.Patch(facecolor='k', label = '$y=x$', linestyle='--')
    # patch4 = mpatches.Patch(facecolor = 'tab:orange', label = '$y=\pm std. \dot x$', linestyle='-')

    # ax1.legend(handles = [p1, patch2, patch3, patch4, patch_t1, patch_t2, patch_t3, patch_t4],bbox_to_anchor=(0.5, 0.3, 0.5, 0.5))
    if outer == None:
        # cbaxes = inset_axes(ax3, width="30%", height="3%", loc=2)
        # plt.colorbar(p1 ,cax=cbaxes, orientation ='horizontal')
        ax7.set_xlabel('Ground Truth [mM]')
        # ax5.set_xlabel('$\sigma$ [mM]')
        ax5.set_ylabel('bias|x [mM]', rotation=0, labelpad=50)
        ax6.set_ylabel('std|x [mM]', rotation=0, labelpad=50)
        ax8.set_ylabel('$\sqrt{MSE}$|x,w [mM]', rotation=0, labelpad=70)
        ax7.set_ylabel('spread [mM]|x,w', rotation=0, labelpad=70)
        # ax3.set_ylabel('Avg. Pred. [mM]')

    if outer != None:
        if sharex:
            ax7.set_xlabel('Ground Truth [mM]')
            # ax5.set_xlabel('$\sigma$ [mM]')
        if sharey:
            ax5.set_ylabel('bias|x [mM]', rotation=0, labelpad=50)
            ax6.set_ylabel('std|x [mM]', rotation=0, labelpad=50)
            ax8.set_ylabel('$\sqrt{MSE}$|x,w [mM]', rotation=0, labelpad=70)
            ax7.set_ylabel('spread [mM]|x,w', rotation=0, labelpad=70)
            # ax3.set_ylabel('Avg. Pred. [mM]')
    # ax1.axis('off')
    # gs.tight_layout()


fig = plt.figure()
jointregression_mcSTD(order[0], metnames[order[0]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, outer=None, sharey = 1, sharex = 1)

# -------------------------------------------------------------
# plot regression 2x4
# -------------------------------------------------------------
def plotREGR2x4fromindex_mcSTD(i):
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)
                            # top=0.925,
                            # bottom=0.08,
                            # left=0.075,
                            # right=0.985,
                            # hspace=0.17,
                            # wspace=0.05)
    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row,col])
            if (i==0) or (i==8):
                jointregression_mcSTD(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, spec[row,col], sharey=1)
            elif (i==4) or (i==12):
                jointregression_mcSTD(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, spec[row,col], sharex=1, sharey=1)
            elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
                jointregression_mcSTD(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, spec[row, col], sharex=1)
            else:
                jointregression_mcSTD(order[i], metnames[order[i]], y_test, mean_p, mean_p_sort, std_intrp, y_test_sort, std_intrp_y_test, snr_v, spec[row, col])

            i += 1

plotREGR2x4fromindex_mcSTD(0)
plotREGR2x4fromindex_mcSTD(8)

# trial for calibration plots
# accuracy definition (after Mauricio's talk)
accuracy_mc = np.zeros(std_p.shape)
confidence_mc = std_p

bincal = 50
accuracy_dt = np.zeros(np.shape(std_p))
confidence_dt = np.zeros(np.shape(std_p))
for m in range(17):
    for sig in range(2500):
        th = std_p[sig,m]
        accuracy_mc[sig, m] = np.nonzero(np.abs(p[sig, m, :] - y_test[sig, m]) < th)[0].size / p.shape[2]
        # accuracy_mc[sig, m] = np.abs(p[sig, m, :] - y_test[sig, m])


    for b in range(bincal):
        idx_start = np.int((y_test.shape[0] / bincal) * b)
        idx_stop = np.int(((y_test.shape[0] / bincal) * (b + 1)) - 1)
        th = np.std(pred_det_sort_aslabel[idx_start:idx_stop + 1, m])
        confidence_dt[idx_start:idx_stop + 1, m] = th  # spread of point spread function (deterministic prediction)
        accuracy_dt[idx_start:idx_stop + 1, m] = np.nonzero(np.abs(pred_det_sort_aslabel[idx_start:idx_stop + 1, m] - y_test_sort[idx_start:idx_stop + 1, m]) < th)[0].size / (np.shape(y_test)[0]/bincal)


def calibration_plots(index, met, outer=None, sharey = 0, sharex = 0):
    from matplotlib import gridspec
    import seaborn as sns
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FormatStrFormatter

    # fig = plt.figure()
    if outer == None:
        gs = fig.add_gridspec(2, 1, width_ratios=[1], height_ratios=[1, 1],
                                               wspace=0.2, hspace=0.5)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer, width_ratios=[1], height_ratios=[1, 1],
                                               wspace=0.2, hspace=0.5)

    idd = np.argsort(confidence_mc[:, index])
    conf_mc = confidence_mc[idd, index]
    acc_mc = accuracy_mc[idd, index]
    ident_mc = [np.min(conf_mc), np.max(conf_mc)]
    acc_mc_interp = savgol_filter(acc_mc, 101, 3)


    idd = np.argsort(confidence_dt[:, index])
    conf_dt = confidence_dt[idd, index]
    acc_dt = accuracy_dt[idd, index]
    ident_dt = [np.min(conf_dt), np.max(conf_dt)]

    ax0 = plt.subplot(gs[0])
    ax0.set_title(met, fontweight="bold")
    # ax5.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax0.plot(conf_mc, acc_mc, 'lightgreen')
    ax0.plot(conf_mc, acc_mc_interp, 'red')
    # ax0.plot(ident_mc, ident_mc, '--', linewidth=3, color='k')
    # ax5.xaxis.set_visible(False)
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax0.axhline(0.68, 0, 1, color='gray', alpha=0.5)

    ax1 = plt.subplot(gs[1])
    # ax6.set_xlim(0 - (0.05 * m), m + (0.05 * m))
    ax1.plot(conf_dt, acc_dt, 'dodgerblue')
    # ax1.plot(ident_dt, ident_dt, '--', linewidth=3, color='k')
    # ax6.plot(preds_sort_label[:, index], y_test_bin_sort_sigma[:, index], 'blue')
    # ax6.xaxis.set_visible(False)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax1.axhline(0.68, 0, 1, color='gray', alpha=0.5)


    # ax1.legend(handles = [p1, patch2, patch3, patch4, patch_t1, patch_t2, patch_t3, patch_t4],bbox_to_anchor=(0.5, 0.3, 0.5, 0.5))
    if outer == None:
        # cbaxes = inset_axes(ax3, width="30%", height="3%", loc=2)
        # plt.colorbar(p1 ,cax=cbaxes, orientation ='horizontal')
        ax0.set_ylabel('Model Conf [%]')
        ax1.set_ylabel('Data Conf [%]')
        # ax5.set_xlabel('$\sigma$ [mM]')
        ax0.set_xlabel('$\sigma$ - MC')
        ax1.set_xlabel('$\sigma$ - data bin')
        # ax3.set_ylabel('Avg. Pred. [mM]')

    # if outer != None:
    #     if sharex:
    #         ax7.set_xlabel('Ground Truth [mM]')
    #         # ax5.set_xlabel('$\sigma$ [mM]')
    #     if sharey:
    #         ax5.set_ylabel('bias|x [mM]', rotation=0, labelpad=50)
    #         ax6.set_ylabel('std|x [mM]', rotation=0, labelpad=50)
    #         ax8.set_ylabel('$\sqrt{MSE}$|x,w [mM]', rotation=0, labelpad=70)
    #         ax7.set_ylabel('spread [mM]|x,w', rotation=0, labelpad=70)
    #         # ax3.set_ylabel('Avg. Pred. [mM]')

fig = plt.figure()
i=0
calibration_plots(order[i], metnames[order[i]], outer=None, sharey = 0, sharex = 0)




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

y_test_p1_sort = np.zeros(mean_p1.shape)
y_test_p2_sort = np.zeros(mean_p2.shape)
y_test_p3_sort = np.zeros(mean_p3.shape)
std_p1_sort_ytest = np.zeros(mean_p1.shape)
std_p2_sort_ytest = np.zeros(mean_p2.shape)
std_p3_sort_ytest = np.zeros(mean_p3.shape)
std_intrp1_ytest = np.zeros(mean_p1.shape)
std_intrp2_ytest = np.zeros(mean_p2.shape)
std_intrp3_ytest = np.zeros(mean_p3.shape)
p_sort1 = np.zeros((mean_p1.shape[0], p.shape[1], p.shape[2]))
p_sort2 = np.zeros((mean_p2.shape[0], p.shape[1], p.shape[2]))
p_sort3 = np.zeros((mean_p3.shape[0], p.shape[1], p.shape[2]))

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

    idx_1gt = np.argsort(y_test1[:, i])
    y_test_p1_sort[:, i] = y_test1[idx_1gt, i]
    std_p1_sort_ytest[:, i] = std_p1[idx_1gt, i]
    std_intrp1_ytest[:, i] = savgol_filter(std_p1_sort_ytest[:, i], 51, 3)


    idx_2gt = np.argsort(y_test2[:, i])
    y_test_p2_sort[:, i] = y_test2[idx_2gt, i]
    std_p2_sort_ytest[:, i] = std_p2[idx_2gt, i]
    std_intrp2_ytest[:, i] = savgol_filter(std_p2_sort_ytest[:, i], 51, 3)

    idx_3gt = np.argsort(y_test3[:, i])
    y_test_p3_sort[:, i] = y_test3[idx_3gt, i]
    std_p3_sort_ytest[:, i] = std_p3[idx_3gt, i]
    std_intrp3_ytest[:, i] = savgol_filter(std_p3_sort_ytest[:, i], 51, 3)


fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test1, mean_p1, mean_p1_sort, std_intrp1, y_test_p1_sort, std_intrp1_ytest, snr_v1, outer=None, sharey = 1, sharex = 1)
fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test2, mean_p2, mean_p2_sort, std_intrp2, y_test_p2_sort, std_intrp2_ytest, snr_v2, outer=None, sharey = 1, sharex = 1)
fig = plt.figure()
jointregression(order[0], metnames[order[0]], y_test3, mean_p3, mean_p3_sort, std_intrp3, y_test_p3_sort, std_intrp3_ytest, snr_v3, outer=None, sharey = 1, sharex = 1)

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
    jointregression(order[index1], metnames[order[index1]], y_test1, mean_p1, mean_p1_sort, std_intrp1, y_test_p1_sort, std_intrp1_ytest, snr_v1, ax, sharey=1)
    ax = fig.add_subplot(spec[0, 1])
    jointregression(order[index1], metnames[order[index1]], y_test2, mean_p2, mean_p2_sort, std_intrp2, y_test_p2_sort, std_intrp2_ytest, snr_v2, ax,
                    sharex=0, sharey=0)
    ax = fig.add_subplot(spec[0, 2])
    jointregression(order[index1], metnames[order[index1]], y_test3, mean_p3, mean_p3_sort, std_intrp3, y_test_p3_sort, std_intrp3_ytest, snr_v3, ax,
                    sharex=0, sharey=0)

    ax = fig.add_subplot(spec[1, 0])
    jointregression(order[index2], metnames[order[index2]], y_test1, mean_p1, mean_p1_sort, std_intrp1, y_test_p1_sort, std_intrp1_ytest, snr_v1, ax,
                    sharex=1, sharey=1)
    ax = fig.add_subplot(spec[1, 1])
    jointregression(order[index2], metnames[order[index2]], y_test2, mean_p2, mean_p2_sort, std_intrp2, y_test_p2_sort, std_intrp2_ytest, snr_v2, ax,
                    sharex=1, sharey=0)
    ax = fig.add_subplot(spec[1, 2])
    jointregression(order[index2], metnames[order[index2]], y_test3, mean_p3, mean_p3_sort, std_intrp3, y_test_p3_sort, std_intrp3_ytest, snr_v3, ax,
                    sharex=1, sharey=0)


plotREGR2x3SNR(0, 4)
plotREGR2x3SNR(8, 14)


## density sort of plot

y_test_sort = np.zeros(mean_p.shape)
p_sort = np.zeros(p.shape)

snr_v_oidx = np.zeros((snr_v.shape[0], 17))
for i in range(17):
    idx_gt = np.argsort(y_test[:, i])
    snr_v_oidx[:,i] = snr_v[idx_gt,0]
    y_test_sort[:, i] = y_test[idx_gt, i]
    p_sort[:,i,:] = p[idx_gt, i, :]

bin=5
boxplot_matrix = np.zeros((bin, np.int(nlabels.shape[0]/bin)*p.shape[-1], 17))

y_test_sort_box = np.zeros((bin, np.int(nlabels.shape[0]/bin), 17))
y_ticks = np.zeros((17, bin+1))
boxplot_matrix_snr1 = []
boxplot_matrix_snr2 = []
boxplot_matrix_snr3 = []
std_check = np.zeros((bin, 17))
std_check_boxplotm = np.zeros((bin, 17))
for met in range(17):
    # boxplot_matrix_snr3_new[met] = None
    for i in range(bin):
        idx_start = np.int((nlabels.shape[0]/bin) * i)
        idx_stop =  np.int(( (nlabels.shape[0]/bin) * (i+1) ) - 1)
        boxplot_matrix[i, :, met] = np.reshape(p_sort[idx_start:idx_stop+1, met,:],
                                               (np.int(nlabels.shape[0]/bin)*p.shape[-1]))

        std_check[i, met] = np.mean(np.std(p_sort[idx_start:idx_stop+1, met,:], axis=1))
        std_check_boxplotm[i, met] = np.std(boxplot_matrix[i, :, met])
        idx_snr1 = np.nonzero(snr_v_oidx[idx_start:idx_stop + 1, met] < 16.7) #16.7
        ss = p_sort[idx_snr1[0] + idx_start, met, :]
        boxplot_matrix_snr1.append(ss.reshape((ss.shape[0] * ss.shape[1], 1)))

        idx_snr2 = np.nonzero((snr_v_oidx[idx_start:idx_stop + 1, met] >= 16.7) & (snr_v_oidx[idx_start:idx_stop + 1, met] < 28.4))
        ss = p_sort[idx_snr2[0] + idx_start, met, :]
        boxplot_matrix_snr2.append(ss.reshape((ss.shape[0] * ss.shape[1], 1)))

        idx_snr3 = np.nonzero(snr_v_oidx[idx_start:idx_stop+1, met] >= 28.4)
        ss = p_sort[idx_snr3[0]+idx_start, met, :]
        boxplot_matrix_snr3.append(ss.reshape((ss.shape[0]*ss.shape[1], 1)))

        y_test_sort_box[i, :, met] = y_test_sort[idx_start:idx_stop + 1, met]
        y_ticks[met, i] = y_test_sort[idx_stop, met]


def boxplotpsf(met, bins, candels, ax):
    import matplotlib.patches as patches
    meanpointprops = dict(marker='D', markeredgecolor="black", markerfacecolor='tab:orange', markersize=7)
    medianp = dict(linestyle='-', linewidth=1.5, color='black')
    flierprops = dict(marker='o', markerfacecolor='k', markersize=5,
                      linestyle='none')
    # fig, ax = plt.subplots(figsize=(10,10))

    posboxplt = np.arange(1, bins*(candels), candels)
    bp = ax.boxplot(boxplot_matrix[:, :, met].T, showmeans=True, positions=posboxplt,
                         patch_artist=True, meanprops=meanpointprops, medianprops=medianp, flierprops=flierprops,
                         widths=0.5)

    posboxplt_snr1 = np.arange(2, bins*(candels), candels)
    posboxplt_snr2 = np.arange(3, bins * (candels), candels)
    posboxplt_snr3 = np.arange(4, bins * (candels), candels)

    patchespos = np.arange(0, bins*(candels), candels)
    for i in range(bin):
        rect = patches.Rectangle((patchespos[i]-0.25, np.min(y_test_sort_box[i, :, met])), 0.5, np.max(y_test_sort_box[i, :, met] - np.min(y_test_sort_box[i, :, met])), linewidth=1, edgecolor='k', facecolor='r')
        ax.add_patch(rect)

        bp1 = ax.boxplot(boxplot_matrix_snr1[i+(met*bin)], showmeans=True, positions=[posboxplt_snr1[i]],
                     patch_artist=True, meanprops=meanpointprops, medianprops=medianp, flierprops=flierprops,
                     widths=0.5)
        bp2 = ax.boxplot(boxplot_matrix_snr2[i + (met * bin)], showmeans=True, positions=[posboxplt_snr2[i]],
                        patch_artist=True, meanprops=meanpointprops, medianprops=medianp, flierprops=flierprops,
                        widths=0.5)
        bp3 = ax.boxplot(boxplot_matrix_snr3[i + (met * bin)], showmeans=True, positions=[posboxplt_snr3[i]],
                        patch_artist=True, meanprops=meanpointprops, medianprops=medianp, flierprops=flierprops,
                        widths=0.5)

        [patch.set(facecolor='green', alpha=1.0) for patch in bp1['boxes']]
        [patch.set(facecolor='limegreen', alpha=1.0) for patch in bp2['boxes']]
        [patch.set(facecolor='yellow', alpha=1.0) for patch in bp3['boxes']]

    ax.set_yticks(y_ticks[met, :])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax.set_title(metnames[met], fontweight='bold')
    [patch.set(facecolor='tab:blue', alpha=1.0) for patch in bp['boxes']]

    vlinepos = np.arange(0, candels*bins, candels) - 0.5
    for j in range(bin-1):
        ax.axvline(vlinepos[j+1] , color = 'gray', alpha = 0.5)
    ax.xaxis.set_visible(False)

# boxplotpsf(order[7], bin, 5)

def boxplot3from(met):
    fig = plt.figure(figsize = (30,10))
    gs = fig.add_gridspec(1, 3, wspace=0.15, hspace=0.15, left = 0.040, right = 0.955)
    for col in range(3):
        ax = fig.add_subplot(gs[0, col])
        boxplotpsf(order[met], bin, 5, ax)
        met += 1

# boxplot3from(0)
# boxplot3from(3)
# boxplot3from(6)
# boxplot3from(9)
# boxplot3from(12)
# boxplot3from(15)
#
def psf(met):
    import matplotlib.gridspec as grid_spec
    import matplotlib.patches as patches

    labelname = metnames[met]
    spec = (grid_spec.GridSpec(ncols=1, nrows=bin, hspace=-0.5))
    ax_obj = []

    for p in range(bin):
        ax_obj.append(fig.add_subplot(spec[p]))
        sns.distplot(boxplot_matrix[bin-1-p, :, met], color='tab:olive')

        mu = np.mean(boxplot_matrix[bin-1-p, :, met])
        sigma = np.std(boxplot_matrix[bin-1-p, :, met])
        # text
        textstr = '\n'.join((
            r'$\mu=%.2f$' % (mu,),
            r'$\sigma=%.2f$' % (sigma,)))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_obj[p].text(0.85, 0.05, textstr, transform=ax_obj[p].transAxes,
                 verticalalignment='bottom', bbox=props)

        ax_obj[p].set_xlim(np.min(boxplot_matrix[0,:,met]), np.max(boxplot_matrix[bin-1,:,met]))

        # plt.axvline(np.round(pos[met,bin-1-p],1), 0, 1, color='red', alpha=0.5)
        # plt.axvline(np.min(y_test[:,met]), 0, 0.5, color='gray', alpha=0.5)
        # plt.axvline(np.max(y_test[:,met]), 0, 0.5, color='gray', alpha=0.5)
        # plt.fill_between(np.arange(np.min(y_test_sort_box[bin-1-p,:,met]),np.max(y_test_sort_box[bin-1-p,:,met]),0.01),0, 0.1, color='k', alpha=0.5)


        rect = patches.Rectangle((np.min(y_test_sort_box[bin-1-p,:,met]), 0),  np.max(y_test_sort_box[bin-1-p, :, met] - np.min(y_test_sort_box[bin-1-p, :, met])),
                                0.05*ax_obj[p].get_ylim()[1], linewidth=1, edgecolor='k', facecolor='r')
        ax_obj[p].add_patch(rect)

        rect = ax_obj[p].patch
        rect.set_alpha(0)

        ax_obj[p].set_yticklabels([])
        ax_obj[p].set_yticks([])
        ax_obj[p].set_ylabel(np.round(np.mean(y_test_sort_box[bin-1-p,:,met]), 2), loc='bottom')
        # ax_obj[p].set_ylabel(np.round(pos[met,bin-1-p],1), loc='bottom', fontsize = 10)
        spines = ["right", "left", "top"]

        for s in spines:
            ax_obj[p].spines[s].set_visible(False)

    for p in range(bin-1):
        ax_obj[p].set_xticklabels([])
        ax_obj[p].set_xticks([])

    ax_obj[0].set_title(labelname)

# for i in range(16):
#     fig = plt.figure(figsize=(5.5, 6))
#     psf(order[i])
fig = plt.figure(figsize=(5.5, 6))
psf(order[10])

b = [boxplot_matrix_snr1, boxplot_matrix_snr2, boxplot_matrix_snr3]
def psfSNR(met,snr_level):
    import matplotlib.gridspec as grid_spec
    import matplotlib.patches as patches

    labelname = metnames[met]
    spec = (grid_spec.GridSpec(ncols=1, nrows=bin, hspace=-0.5))
    ax_obj = []

    for p in range(bin):
        ax_obj.append(fig.add_subplot(spec[p]))
        sns.distplot(b[snr_level][bin-1-p + (met*bin)], color='tab:olive')

        mu = np.mean(b[snr_level][bin-1-p + (met*bin)])
        sigma = np.std(b[snr_level][bin-1-p + (met*bin)])
        # text
        textstr = '\n'.join((
            r'$\mu=%.2f$' % (mu,),
            r'$\sigma=%.2f$' % (sigma,)))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_obj[p].text(0.85, 0.05, textstr, transform=ax_obj[p].transAxes,
                 verticalalignment='bottom', bbox=props)

        ax_obj[p].set_xlim(np.min(b[snr_level][0 + (met*bin)]), np.max(boxplot_matrix_snr1[bin-1 + (met*bin)]))

        # plt.axvline(np.round(pos[met,bin-1-p],1), 0, 1, color='red', alpha=0.5)
        # plt.axvline(np.min(y_test[:,met]), 0, 0.5, color='gray', alpha=0.5)
        # plt.axvline(np.max(y_test[:,met]), 0, 0.5, color='gray', alpha=0.5)
        # plt.fill_between(np.arange(np.min(y_test_sort_box[bin-1-p,:,met]),np.max(y_test_sort_box[bin-1-p,:,met]),0.01),0, 0.1, color='k', alpha=0.5)

        rect = patches.Rectangle((np.min(y_test_sort_box[bin - 1 - p, :, met]), 0), np.max(
            y_test_sort_box[bin - 1 - p, :, met] - np.min(y_test_sort_box[bin - 1 - p, :, met])),
                                 0.05 * ax_obj[p].get_ylim()[1], linewidth=1, edgecolor='k', facecolor='r')
        ax_obj[p].add_patch(rect)

        rect = ax_obj[p].patch
        rect.set_alpha(0)

        ax_obj[p].set_yticklabels([])
        ax_obj[p].set_yticks([])
        ax_obj[p].set_ylabel(np.round(np.mean(y_test_sort_box[bin - 1 - p, :, met]), 2), loc='bottom')
        # ax_obj[p].set_ylabel(np.round(pos[met,bin-1-p],1), loc='bottom', fontsize = 10)
        spines = ["right", "left", "top"]

        for s in spines:
            ax_obj[p].spines[s].set_visible(False)

    for p in range(bin - 1):
        ax_obj[p].set_xticklabels([])
        ax_obj[p].set_xticks([])

    ax_obj[0].set_title(labelname)

fig = plt.figure(figsize=(5.5, 6))
psfSNR(order[0], 0)
fig = plt.figure(figsize=(5.5, 6))
psfSNR(order[0], 1)
fig = plt.figure(figsize=(5.5, 6))
psfSNR(order[0], 2)

std_check[:,order[0]]


# def jointPSF(met):
#     plt.plot(mean_p_sort[:,met], mean_p_sort[:,met])
#     # plt.scatter(y_test[:,met], mean_p[:,met])
#     up_lim = mean_p_sort[:, met] + std_intrp[:,met]
#     # down_lim = mean_p_sort[:, met] - std_intrp[:,met]
#     # up_lim = mean_p_sort[:,met] - std_p_sort[:,met]
#     # down_lim = mean_p[:, met] + std_p[:, met]
#     plt.plot(mean_p_sort[:, met], up_lim)
#     # plt.plot(down_lim)
#     # plt.fill_between(mean_p_sort[:,met], up_lim, down_lim, alpha=0.2)
#     # plt.plot(mean_p_sort[:, met])
#     # plt.plot(mean_p_sort[:,met], std_intrp[:, met], 'green')
#     # plt.plot(y_test_sort[:, met], std_intrp_y_test[:, met], 'red')
#
# fig = plt.figure()
# jointPSF(order[7])
# for i in range(17):
#     idx = np.argsort(mean_p[:,i])
#     mean_p_sort[:,i] = mean_p[idx,i]
#     std_p_sort[:, i] = std_p[idx, i]
#     std_intrp[:,i] = savgol_filter(std_p_sort[:, i], 51, 3)
#
#     idx_gt = np.argsort(y_test[:, i])
#     y_test_sort[:, i] = y_test[idx_gt, i]
#     std_p_sort_y_test[:, i] = std_p[idx_gt, i]
#     std_intrp_y_test[:, i] = savgol_filter(std_p_sort_y_test[:, i], 51, 3)

