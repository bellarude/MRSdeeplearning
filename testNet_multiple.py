from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import xlsxwriter

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from data_load_norm import dataNorm, labelsNorm, ilabelsNorm, inputConcat1D, inputConcat2D, dataimport2D_md, \
    labelsimport_md
from models import newModel

regr = linear_model.LinearRegression()

input1d = 1
md_input = 0
flat_input = 0

if md_input == 0:
    # dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/test dataset/'


    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'

    def datatestimport():
        global dataset1D, dataset2D, nlabels, w_nlabels, snr_v, shim_v

        snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
        readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')
        labels_import = sio.loadmat(dest_folder + 'labels_c_TEST_nw.mat')

        labels = labels_import['labels_c'] * 64.5
        snr_v = snr_v['snr_v']
        shim_v = readme_SHIM['shim_v']
        # labels = np.transpose(labels,(1,0))
        nlabels, w_nlabels = labelsNorm(labels)

        if input1d:
            data_import1D = sio.loadmat(dest_folder + 'dataset_spectra_nw_TEST.mat')
            dataset1D = data_import1D['dataset_spectra_nw']
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

if flat_input:
    dataset2D_flat = inputConcat2D(dataset2D)

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']
order = [2, 0, 4, 12, 7, 6, 1, 8, 9, 14, 10, 3, 13, 15, 11, 5]  # to order metabolites plot from good to bad

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "water_reference/"  # "net_type/"
subfolder = "ResNet_no_wat/"  # "typology/"
directory = outpath + folder + subfolder

excelname = '/water_reference_no_wat_eval.xlsx'
workbook = xlsxwriter.Workbook(directory + excelname)


def scores(index):
    # ----------------------------------------------
    x = y_test[:, index].reshape(-1, 1)
    y = pred[:, index]
    regr.fit(x, y)
    lin = regr.predict(np.arange(0, np.max(y_test[:, index]), 0.01).reshape(-1, 1))
    mse = mean_squared_error(x, y)
    r_sq = regr.score(x, y)

    return regr.coef_[0], regr.intercept_, r_sq, mse


num_train = 10
j = 0
for filename in os.listdir(directory):
    if filename.endswith(".hdf5"):
        j += 1
        if (j == num_train + 1) | (j == 1):
            worksheet = workbook.add_worksheet(filename[19:22])
            # worksheet = workbook.add_worksheet()
            j = 1

        checkpoint_path = outpath + folder + subfolder + filename
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model = newModel(dim='1D', type='ResNet', subtype='ResNet_fed_hp_nw')
        model.load_weights(checkpoint_path)

        loss = model.evaluate(dataset1D, nlabels, verbose=2)

        pred_abs = model.predict(dataset1D)  # normalized [0-1] absolute concentrations prediction
        pred_un = np.empty(pred_abs.shape)  # un-normalized absolute concentrations
        pred = np.empty(pred_abs.shape)  # relative un normalized concentrations (referred to water prediction)

        pred_un = ilabelsNorm(pred_abs, w_nlabels)
        y_test = ilabelsNorm(nlabels, w_nlabels)

        # for i in range(17):
        #     pred[:, i] = pred_un[:, i] / pred_un[:, 16] * 64.5
        #     y_test[:, i] = y_test[:, i] / y_test[:, 16] * 64.5

        # nowat quantification
        for i in range(16):
            pred[:, i] = pred_un[:, i]
            y_test[:, i] = y_test[:, i]

        for i in range(16):
            a, q, r2, mse = scores(i)
            # s = 'A' + str(i * 4 + 1)
            worksheet.write(i * 4, 0, 'a')
            worksheet.write_number(i * 4, j, a)
            # s = 'A' + str(i * 4 + 2)
            worksheet.write(i * 4+1, 0, 'q')
            worksheet.write_number(i * 4 + 1, j, q)
            # s = 'A' + str(i * 4 + 3)
            worksheet.write(i * 4+2, 0, 'R^2')
            worksheet.write_number(i * 4 + 2, j, r2)
            # s = 'A' + str(i * 4 + 4)
            worksheet.write(i * 4+3, 0, 'mse')
            worksheet.write_number(i * 4 + 3, j, mse)

workbook.close()
print('xlsx SAVED')
