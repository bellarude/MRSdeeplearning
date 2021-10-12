from os import scandir
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['legend.fontsize'] = 20
matplotlib.rcParams['font.size'] = 15

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']
ordernames = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5]

d24e = pd.read_excel(r"C:\Users\Rudy\Desktop\DL_models\active_learning\active_learning_eval.xlsx", sheet_name = 'd24', header=None)
d25e = pd.read_excel(r"C:\Users\Rudy\Desktop\DL_models\active_learning\active_learning_eval.xlsx", sheet_name = 'd25', header=None)
d26e = pd.read_excel(r"C:\Users\Rudy\Desktop\DL_models\active_learning\active_learning_eval.xlsx", sheet_name = 'd26', header=None)
d27e = pd.read_excel(r"C:\Users\Rudy\Desktop\DL_models\active_learning\active_learning_eval.xlsx", sheet_name = 'd27', header=None)
d29e = pd.read_excel(r"C:\Users\Rudy\Desktop\DL_models\active_learning\active_learning_eval.xlsx", sheet_name = 'd29', header=None)
d30e = pd.read_excel(r"C:\Users\Rudy\Desktop\DL_models\active_learning\active_learning_eval.xlsx", sheet_name = 'd30', header=None)
d32e = pd.read_excel(r"C:\Users\Rudy\Desktop\DL_models\active_learning\active_learning_eval.xlsx", sheet_name = 'd32', header=None)
#NB: this dataset has saved std as mse so i need to evaluate std here.

d24 = np.empty((d24e.shape))
d25 = np.empty((d25e.shape))
d26 = np.empty((d26e.shape))
d27 = np.empty((d27e.shape))
d29 = np.empty((d29e.shape))
d30 = np.empty((d30e.shape))
d32 = np.empty((d32e.shape))

for j in range(d24e.shape[0]):
    if d24e.values[j,0] == 'mse':
        d24[j, 1:10] = np.sqrt(d24e.values[j, 1:10].astype(float))
        d25[j, 1:10] = np.sqrt(d25e.values[j, 1:10].astype(float))
        d26[j, 1:10] = np.sqrt(d26e.values[j, 1:10].astype(float))
        d27[j, 1:10] = np.sqrt(d27e.values[j, 1:10].astype(float))
        d29[j, 1:10] = np.sqrt(d29e.values[j, 1:10].astype(float))
        d30[j, 1:10] = np.sqrt(d30e.values[j, 1:10].astype(float))
        d32[j, 1:10] = np.sqrt(d32e.values[j, 1:10].astype(float))
    else:
        d24[j, 1:10] = d24e.values[j, 1:10].astype(float)
        d25[j, 1:10] = d25e.values[j, 1:10].astype(float)
        d26[j, 1:10] = d26e.values[j, 1:10].astype(float)
        d27[j, 1:10] = d27e.values[j, 1:10].astype(float)
        d29[j, 1:10] = d29e.values[j, 1:10].astype(float)
        d30[j, 1:10] = d30e.values[j, 1:10].astype(float)
        d32[j, 1:10] = d32e.values[j, 1:10].astype(float)

# data_d24 = np.array(d24.values[:, 2:7], dtype=np.float64)
# data_d25 = np.array(d25.values[:, 2:7], dtype=np.float64)

# avg_water = np.mean(data_water , axis=1)
# avg_nowater = np.mean(data_nowater , axis=1)
# std_water = np.std(data_water , axis=1)
# std_nowater = np.std(data_nowater , axis=1)




fig = plt.figure(constrained_layout=True, figsize=(10,20))
widths = 1*np.ones(4)
heights = 3*np.ones(8)
spec = fig.add_gridspec(ncols=4, nrows=8, width_ratios=widths,
                          height_ratios=heights)

meanpointprops = dict(marker='D', markeredgecolor="black", markerfacecolor='tab:orange', markersize=6)
medianprops = dict(linewidth=2.0, color='black')
boxprops = dict(linewidth=1.5)
outprops = dict(linewidth=1.5)

param = ['$a$', '$q$', '$R^2$', '$\sigma$ [mM]']
met = 0
plt_idx = 0
for row in range(8):
    for col in range(4):
        ax = fig.add_subplot(spec[row,col])

        bp1 = ax.boxplot([d24[ordernames[met]*4 + plt_idx, 1:10]],showmeans=True, positions=[1],
                        patch_artist=True, meanprops=meanpointprops)
        bp2 = ax.boxplot([d25[ordernames[met]*4 + plt_idx, 1:10]],showmeans=True, positions=[1.5],
                        patch_artist=True, meanprops=meanpointprops)
        bp3 = ax.boxplot([d26[ordernames[met] * 4 + plt_idx, 1:10]], showmeans=True, positions=[2],
                         patch_artist=True, meanprops=meanpointprops)
        bp4 = ax.boxplot([d27[ordernames[met] * 4 + plt_idx, 1:10]], showmeans=True, positions=[2.5],
                         patch_artist=True, meanprops=meanpointprops)
        bp5 = ax.boxplot([d29[ordernames[met] * 4 + plt_idx, 1:10]], showmeans=True, positions=[3],
                         patch_artist=True, meanprops=meanpointprops)
        bp6 = ax.boxplot([d30[ordernames[met] * 4 + plt_idx, 1:10]], showmeans=True, positions=[3.5],
                         patch_artist=True, meanprops=meanpointprops)
        bp7 = ax.boxplot([d32[ordernames[met] * 4 + plt_idx, 1:10]], showmeans=True, positions=[4],
                         patch_artist=True, meanprops=meanpointprops)
        ax.set_xticks([])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        [patch.set(facecolor='tab:blue', alpha=1.0) for patch in bp1['boxes']]
        [patch.set(facecolor='tab:orange', alpha=1.0) for patch in bp2['boxes']]
        [patch.set(facecolor='tab:green', alpha=1.0) for patch in bp3['boxes']]
        [patch.set(facecolor='tab:red', alpha=1.0) for patch in bp4['boxes']]
        [patch.set(facecolor='tab:purple', alpha=1.0) for patch in bp4['boxes']]
        [patch.set(facecolor='tab:olive', alpha=1.0) for patch in bp4['boxes']]
        [patch.set(facecolor='tab:cyan', alpha=1.0) for patch in bp4['boxes']]

        if row == 0:
            ax.title.set_text(param[col])
        #[patch.set(alpha=None, facecolor=(1, 0, 0, .5)) for patch in bp['boxes']]
        if col == 0:
            ax.set_ylabel(metnames[ordernames[met]], fontweight="bold")

        plt.xlim([0.5,4.5])
        plt_idx +=1

    met += 1
    plt_idx = 0
    # for col in range(3):
    #     col += 3
    #     ax = fig.add_subplot(spec[row, col])
    #     bp1 = ax.boxplot([dwater[ordernames[met]*3 + plt_idx, 2:7]],showmeans=True, positions=[1],
    #                      patch_artist=True, meanprops=meanpointprops)
    #     bp2 = ax.boxplot([dnowater[ordernames[met]*3 + plt_idx, 2:7]],showmeans=True, positions=[1.5],
    #                      patch_artist=True, meanprops=meanpointprops)
    #     ax.set_xticks([])
    #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    #     [patch.set(facecolor='green', alpha=1.0) for patch in bp1['boxes']]
    #     [patch.set(facecolor='blue', alpha=1.0) for patch in bp2['boxes']]
    #     ax.set_xticks([])
    #     ax.title.set_text(param[col-3])
    #
    #     if col == 3:
    #         ax.set_ylabel(metnames[ordernames[met]], fontweight="bold")
    #
    #     plt.xlim([0.5, 2])
    #     plt_idx +=1

    # met += 1
    # plt_idx = 0


# spec_name = 'waterNOwater_res'
# filename = 'C:/Users/Rudy/Documents/WMD/Project 2 - Deep Learning/waterNOwater/' + spec_name + '.png'
# plt.savefig(filename)
# plt.show()