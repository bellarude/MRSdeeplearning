from matplotlib import gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import savgol_filter

# -------------------------------------------------------------
# plot joint distribution of regression
# -------------------------------------------------------------

# instead of passing the whole matrix a give as input already the vectors
# y_test[:, index] == gt
# pred[:, index] == pred
def jointregression(fig, gt, pred, metname, snr_v=[], outer=None, sharey = 0, sharex = 0, yscale = 0, pred_ref = 0):
    """
    :param fig: figure where to plot
    :param gt: ground truth vector Nx1
    :param pred: prediction vector Nx1
    :param metname: name of the metabolite
    :param snr_v: vector with SNR values Nx1
    :param outer: 1 if the plot is a subplot
    :param sharey: 1 if y label must not be printed
    :param sharex: 1 if the x label must not be printed
    :param yscale: 1 if y-axis-lim == x-axis-lim
    :param pred_ref: plots line reference for predicted concentrations at GT limit
    :return: regression plot GT vs. prediction with marginal distributions
    """

    regr = linear_model.LinearRegression()

    # fig = plt.figure()
    if outer == None:
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec = outer, width_ratios=[3, 1], height_ratios=[1, 3],
                                               wspace=0.05, hspace=0.05)
    # ----------------------------------------------
    # SCOREs CALCULATION
    x = gt.reshape(-1, 1)
    y = pred
    regr.fit(x, y)
    lin = regr.predict(np.arange(np.min(gt), np.max(gt), 0.01).reshape(-1, 1))
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)
    # ----------------------------------------------

    ax2 = plt.subplot(gs[2])
    if len(snr_v) > 0:
        p1 = ax2.scatter(gt, pred, c=snr_v, cmap='summer', label = 'observation')
    else:
        p1 = ax2.scatter(gt, pred, c='darkgreen', cmap='summer', label='observation')

    ax2.plot(np.arange(np.min(gt), np.max(gt), 0.01), lin, color='tab:olive', linewidth=3)
    ident = [np.min(gt), np.max(gt)]
    ax2.plot(ident, ident, '--', linewidth=3, color='k')
    # ax1 = plt.subplot(gs[1])

    if outer == None:
        cbaxes = inset_axes(ax2, width="30%", height="3%", loc=2)
        plt.colorbar(p1, cax=cbaxes, orientation ='horizontal')

    if outer != None:
        if sharex :
            ax2.set_xlabel('Ground Truth [mM]')
        if sharey:
            ax2.set_ylabel('Predictions [mM]')

    # ax2.plot(np.arange(0, m, 0.01), lin - np.sqrt(mse), color = 'tab:orange', linewidth=3)
    # ax2.plot(np.arange(0, m, 0.01), lin + np.sqrt(mse), color = 'tab:orange', linewidth=3)

    mP = np.min(y)
    MP = np.max(y)


    ax2.set_xlim(np.min(gt) - (0.05 * np.max(gt)), np.max(gt) + (0.05 * np.max(gt)))
    if yscale:
        ax2.set_ylim(np.min(gt) - (0.05 * np.max(gt)), np.max(gt) + (0.05 * np.max(gt)))
    else:
        ax2.set_ylim(mP - (0.05 * MP), MP + (0.05 * MP))

    if pred_ref:
        plt.axhline(np.min(gt), 0, 1, color = 'gray', alpha = 0.5)
        plt.axhline(np.max(gt), 0, 1, color='gray', alpha=0.5)

    ax0 = plt.subplot(gs[0])
    ax0.set_title(metname, fontweight="bold")
    sns.distplot(gt, ax=ax0, color='tab:olive')
    ax0.set_xlim(-0.250,np.max(gt)+0.250)
    ax0.xaxis.set_visible(False)
    ax0.yaxis.set_visible(False)
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax0.set_xlim(np.min(gt) - (0.05 * np.max(gt)), np.max(gt) + (0.05 * np.max(gt)))

    ax3 = plt.subplot(gs[3])
    sns.distplot(y, ax=ax3, vertical=True, color='tab:olive')
    if yscale:
        ax3.set_ylim(np.min(gt) - (0.05 * np.max(gt)), np.max(gt) + (0.05 * np.max(gt)))
    else:
        ax3.set_ylim(mP - (0.05 * MP), MP + (0.05 * MP))

    if pred_ref:
        plt.axhline(np.min(gt), 0, 1, color = 'gray', alpha = 0.5)
        plt.axhline(np.max(gt), 0, 1, color='gray', alpha=0.5)

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

    # patch_t1 = mpatches.Patch(facecolor='w', label=r'$a=%.3f$' % (regr.coef_[0],))
    # patch_t2 = mpatches.Patch(facecolor='w', label=r'$q=%.3f$' % (regr.intercept_,))
    # patch_t3 = mpatches.Patch(facecolor='w', label=r'$R^{2}=%.3f$' % (r_sq,))
    # patch_t4 = mpatches.Patch(facecolor='w', label=r'$std.=%.3f$ [mM]' % (np.sqrt(mse),))
    # patch2 = mpatches.Patch(facecolor='tab:red', label='$y=ax+q$', linestyle='-')
    # patch3 = mpatches.Patch(facecolor='k', label = '$y=x$', linestyle='--')
    # patch4 = mpatches.Patch(facecolor = 'tab:orange', label = '$y=\pm std. \dot x$', linestyle='-')

    # ax1.legend(handles = [p1, patch2, patch3, patch4, patch_t1, patch_t2, patch_t3, patch_t4],bbox_to_anchor=(0.5, 0.3, 0.5, 0.5))

    ax1.axis('off')
    # gs.tight_layout()

def plotREGR2x4fromindex(i, gt, pred, order, metnames, snr):
    """
    extends jointregression plot in its basis configuration (missing optional parameters) to a 2x4 fashion via subplot
    :param i: index from where to start plotting. it plots from i to i+8
    :param gt: Mxm matrix of ground truth labels, M: # of samples, m: # of metabolites
    :param pred: Mxm matrix of predictions
    :param order: vector of ordering how to print
    :param metnames: vecotr with label names for each metabolite
    :param snr: eventual snr vector Mx1
    :return: plotting 2x4 via subplot of 8 regression plots for 8 metabolites
    """
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)

    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row, col])
            if (i == 0) or (i == 8):
                jointregression(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], snr_v=snr,
                                outer=spec[row, col], sharey=1)
            elif (i == 4) or (i == 12):
                jointregression(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], snr_v=snr,
                                outer=spec[row, col], sharex=1, sharey=1)
            elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                jointregression(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], snr_v=snr,
                                outer=spec[row, col], sharex=1)
            else:
                jointregression(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], snr_v=snr,
                                outer=spec[row, col])

            i += 1

def blandAltmann_Shim(fig, gt, pred, metname, shim_v, snr_v=[], outer=None, sharey = 0, sharex = 0):
    """
    :param fig: figure where to plot
    :param gt: ground truth vector Nx1
    :param pred: prediction vector Nx1
    :param shim_v: shim values in simulation, vector Nx1
    :param metname: name of the metabolite
    :param snr_v: vector with SNR values Nx1
    :param outer: 1 if the plot is a subplot
    :param sharey: 1 if y label must not be printed
    :param sharex: 1 if the x label must not be printed
    :param yscale: 1 if y-axis-lim == x-axis-lim
    :param pred_ref: plots line reference for predicted concentrations at GT limit
    :return: Bland-Altmann plot of diff = gt-pred vs. Shim values
    """

    diff = pred - gt
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
    ax0.set_title(metname, fontweight="bold")

    mm = np.mean(std_diff[:,0])
    ax0.set_ylim(mm - 0.8*mm, mm + 0.8*mm)

    ax1 = plt.subplot(gs[1])
    if len(snr_v)>0:
        ax1.scatter(sort, s_diff, c=snr_v[idx_s], cmap='summer')
    else:
        ax1.scatter(sort, s_diff, c='darkgreen', cmap='summer')
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

def plotSHIM2x4fromindex(i, gt, pred, order, metnames, shim, snr):
    """
    extends blandAltmann_SNR plot in its basis configuration (missing optional parameters) to a 2x4 fashion via subplot
    :param i: index from where to start plotting. it plots from i to i+8
    :param gt: Mxm matrix of ground truth labels, M: # of samples, m: # of metabolites
    :param pred: Mxm matrix of predictions
    :param order: vector of ordering how to print
    :param metnames: vecotr with label names for each metabolite
    :param shim: shim values Mx1 
    :param snr: eventual snr vector Mx1
    :return: plotting 2x4 via subplot of 8 regression plots for 8 metabolites
    """
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)

    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row, col])
            if (i == 0) or (i == 8):
                blandAltmann_Shim(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], shim_v = shim, snr_v=snr, outer=spec[row, col], sharey=1)
            elif (i == 4) or (i == 12):
                blandAltmann_Shim(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], shim_v = shim, snr_v=snr,
                                outer=spec[row, col], sharex=1, sharey=1)
            elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                blandAltmann_Shim(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], shim_v = shim, snr_v=snr,
                                outer=spec[row, col], sharex=1)
            else:
                blandAltmann_Shim(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], shim_v = shim, snr_v=snr,
                                outer=spec[row, col])

            i += 1

def blandAltmann_SNR(fig, gt, pred, metname, snr_v, outer=None, xlabel='noise', sharey=0, sharex=0):
    """
        :param fig: figure where to plot
        :param gt: ground truth vector Nx1
        :param pred: prediction vector Nx1
        :param metname: name of the metabolite
        :param snr_v: vector with SNR values Nx1
        :param outer: 1 if the plot is a subplot
        :param sharey: 1 if y label must not be printed
        :param sharex: 1 if the x label must not be printed
        :param yscale: 1 if y-axis-lim == x-axis-lim
        :param pred_ref: plots line reference for predicted concentrations at GT limit
        :return: Bland-Altmann plot of diff = gt-pred vs. SNR values
        """
    diff = pred - gt

    snr = snr_v[:,0]
    noise = 1/snr_v[:,0]
    noise_over_gt = noise/gt

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
    ax0.set_title(metname, fontweight="bold")


    ax1 = plt.subplot(gs[1])
    ax1.scatter(sort, s_diff, c=snr_v[idx_s], cmap='summer')
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

def plotSNR2x4fromindex(i, gt, pred, order, metnames, snr):
    """
    extends blandAltmann_SNR plot in its basis configuration (missing optional parameters) to a 2x4 fashion via subplot
    :param i: index from where to start plotting. it plots from i to i+8
    :param gt: Mxm matrix of ground truth labels, M: # of samples, m: # of metabolites
    :param pred: Mxm matrix of predictions
    :param order: vector of ordering how to print
    :param metnames: vecotr with label names for each metabolite
    :param snr: eventual snr vector Mx1
    :return: plotting 2x4 via subplot of 8 regression plots for 8 metabolites
    """
    fig = plt.figure(figsize = (40,10))

    widths = 2*np.ones(4)
    heights = 2*np.ones(2)
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                              height_ratios=heights)

    for row in range(2):
        for col in range(4):
            ax = fig.add_subplot(spec[row, col])
            if (i == 0) or (i == 8):
                blandAltmann_SNR(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], snr_v=snr, outer=spec[row, col], sharey=1)
            elif (i == 4) or (i == 12):
                blandAltmann_SNR(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], snr_v=snr,
                                outer=spec[row, col], sharex=1, sharey=1)
            elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                blandAltmann_SNR(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], snr_v=snr,
                                outer=spec[row, col], sharex=1)
            else:
                blandAltmann_SNR(fig, gt[:, order[i]], pred[:, order[i]], metnames[order[i]], snr_v=snr,
                                outer=spec[row, col])

            i += 1

def sigma_distro(fig, sigma, metname, outer=None, sharey = 0, sharex = 0):
    """
    :param fig: figure where to plot
    :param sigma: vector Nx1 of absolute std -> absolute given that they refer to [0-1] prediction
    :param metname: name of the metabolite
    :param outer: 1 if the plot is a subplot
    :param sharey: 1 if y label must not be printed
    :param sharex: 1 if the x label must not be printed
    :return: distribution plots of std referenced to absolute DL prediction in interval [0-1]
    """

    if outer == None:
        gs = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1],
                                               wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer, width_ratios=[1], height_ratios=[1],
                                               wspace=0.05, hspace=0.05)

    ax0 = plt.subplot(gs[0])
    sns.distplot(sigma, ax=ax0, color='tab:olive')

    if outer != None:
        if sharex :
            ax0.set_xlabel('Absolute STD [0-1]')
        # if sharey:
        ax0.set_ylabel('')

    ax0.set_title(metname, fontweight="bold")

def sigma_vs_gt(fig, gt, sigma, metname, outer=None, sharey = 0, sharex = 0):
    """
    :param fig: figure where to plot
    :param gt: vector Nx1 with Ground Truth values
    :param sigma: vector Nx1 of std converted to mM
    :param metname: name of the metabolite
    :param outer: 1 if the plot is a subplot
    :param sharey: 1 if y label must not be printed
    :param sharex: 1 if the x label must not be printed
    :return: plots of variability of std as function of GT space. A smoothed line is overlapped to the plot
    """

    if outer == None:
        gs = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1],
                                               wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer, width_ratios=[1], height_ratios=[1],
                                               wspace=0.05, hspace=0.05)

    sigma_smooth = savgol_filter(sigma, 51, 3)
    ax0 = plt.subplot(gs[0])
    ax0.plot(gt, sigma)
    ax0.plot(gt, sigma_smooth)

    if outer != None:
        if sharex :
            ax0.set_xlabel('Ground Truth [mM]')
        if sharey:
            ax0.set_ylabel('std [mM]')

    ax0.set_title(metname, fontweight="bold")

def reliability_plot(fig, pred01, nbins, metname, outer=None, sharey = 0, sharex = 0):
    """
    :param fig: figure where to plot
    :param sigma: vector Nx1 of absolute std -> absolute given that they refer to [0-1] prediction
    :param metname: name of the metabolite
    :param outer: 1 if the plot is a subplot
    :param sharey: 1 if y label must not be printed
    :param sharex: 1 if the x label must not be printed
    :return: distribution plots of std referenced to absolute DL prediction in interval [0-1]
    """

    if outer == None:
        gs = fig.add_gridspec(1, 1, width_ratios=[1], height_ratios=[1],
                              wspace=0.05, hspace=0.05)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer, width_ratios=[1], height_ratios=[1],
                                              wspace=0.05, hspace=0.05)

    idx = np.argsort(pred01)
    pred01_sort = pred01[idx]
    th = np.arange(0, 1 + 1 / nbins, 1 / nbins)
    accuracy = np.ones((nbins + 1,))

    ece = 0
    for i in range(nbins):
        # accuracy[i] = np.min(np.argwhere(pred01_sort > th[i])) / pred01.shape[0]
        accuracy[i] = len(np.argwhere(pred01 <= th[i])) / pred01.shape[0]
        # ece = ece + ((len(np.argwhere(pred01_sort > th[i])) / pred01.shape[0]) * np.abs(accuracy[i] - th[i]))
        ece = ece + accuracy[i] * np.abs(accuracy[i] - th[i])

    ax1 = plt.subplot(gs[0])
    ax1.plot(th, accuracy, drawstyle="steps")
    ax1.fill_between(th, accuracy, step="pre", alpha=0.4)
    ax1.plot(th, accuracy, 'o')
    ident = [0, 1]
    ax1.plot(ident, ident, '--')

    textstr = 'ECE=%.2f' % ece
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
             verticalalignment='top', bbox=props)

    if outer != None:
        if sharex :
            ax1.set_xlabel('Confidence')
        if sharey:
            ax1.set_ylabel('Accuracy')

    ax1.set_title(metname, fontweight="bold")