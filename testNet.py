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
        global dataset1D, dataset2D, nlabels, w_nlabels

        data_import2D   = sio.loadmat(dest_folder + 'dataset_spgram_TEST.mat')
        data_import1D = sio.loadmat(dest_folder + 'dataset_spectra_TEST.mat')
        labels_import = sio.loadmat(dest_folder + 'labels_c_TEST_abs.mat')

        dataset2D = data_import2D['output']
        dataset1D = data_import1D['dataset_spectra']
        labels  = labels_import['labels_c']*64.5

        #reshaping
        dataset1D = np.transpose(dataset1D, (0, 2, 1))
        labels = np.transpose(labels,(1,0))

        # nndataset_rs = dataNorm(ndataset_rs)
        # nndataset_rs = ndataset_rs

        nlabels, w_nlabels = labelsNorm(labels)

        return dataset1D, dataset2D, nlabels, w_nlabels


    datatestimport()
else:

    # pred --> output
    # datasetX --> output_noisy
    # labelsY --> output_gt

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33_gauss/'
    filenames = ['zoomedSpgram_pred_1.mat',
                 'zoomedSpgram_pred_2.mat',
                 'zoomedSpgram_pred_3.mat',
                 'zoomedSpgram_pred_4.mat']
    keyname = 'output'

    X_train, X_val, X_test = dataimport2D_md(folder, filenames, keyname)

    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33_gauss/labels/'
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
net_name = "ShallowInception_fact_v2_md_pred"
checkpoint_path = outpath + folder + net_name + ".best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
# model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowInception_fact_v2')

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

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'Myo', 'PE', 'Scy',
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
    plt.plot(y_test[:, index], pred[:, index], 'o')
    m = np.max(y_test[:, index])
    plt.plot(np.arange(0, m, 0.01), lin, linewidth=4)
    ident = [0.0, m]
    plt.plot(ident, ident, '--', linewidth=3, color='k')

    plt.plot(np.arange(0, m, 0.01), lin - np.sqrt(mse))
    plt.plot(np.arange(0, m, 0.01), lin + np.sqrt(mse))
    plt.title(metnames[index] + r' - Coeff: ' + str(np.round(regr.coef_, 3)) + ' - $R^2$: ' + str(
        np.round(r_sq, 3)) + ' - mse: ' + str(np.round(mse, 3)) + ' - std: ' + str(
        np.round(np.sqrt(mse), 3))), plt.xlabel('GT'), plt.ylabel('estimates')

    return regr.coef_[0], r_sq, mse

cc_array = []
for i in range(17):
    # plt.subplot(16,1,i+1)
    fig = plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    cc, rr, mm = subplotconcentration(i)
    cc_array += [cc]
    plt.subplot(1, 3, 2)
    plt.hist(pred[:, i], 20)
    plt.title('PRED distribution')
    plt.subplot(1, 3, 3)
    plt.hist(y_test[:, i])
    plt.title('GT distribution')

    #filename = '/content/drive/My Drive/RR/nets models/met_{0}.png'
    #filename = filename.format(i)
    #plt.savefig(filename)
    # plt.show()

fig = plt.figure(figsize=(12, 6))
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
