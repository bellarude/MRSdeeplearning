from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import xlsxwriter

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from data_load_norm import dataNorm, labelsNorm, ilabelsNorm, inputConcat1D, inputConcat2D, dataimport2D_md, labelsNormREDdataset, labelsimport_md
from models import newModel

import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 18
matplotlib.rcParams['legend.fontsize'] = 12
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 15


input1d = 1
md_input = 1
flat_input = 0
test_diff_conc_bounds = 0
reliability = 0

doSave = 1 #saves scores
doSavePred = 1 #saves predictions and labels
doSNR = 0
doShim = 0

if doSave:
    saving_spec_scores = '_scores'

if doSavePred:
    saving_spec = "_pred"

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
            if doSNR:
                snr_v = snr_v['snr_v']
            if doShim:
                shim_v = readme_SHIM['shim_v']

            nlabels, w_nlabels = labelsNorm(labels)
        else:
            snr_v = sio.loadmat(dest_folder + 'snr_v_TEST_0406')
            readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST_0406.mat')
            labels_import = sio.loadmat(dest_folder + 'labels_c_TEST_0406.mat')

            labels = labels_import['labels_c'] * 64.5
            if doSNR:
                snr_v = snr_v['snr_v']
            if doShim:
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
                data_import2D = sio.loadmat(dest_folder + 'dataset_spgram_TEST_0406.mat')
            dataset2D = data_import2D['output']

            if flat_input:
                dataset2D = inputConcat2D(dataset2D)

            return dataset2D, nlabels, w_nlabels, snr_v, shim_v

    datatestimport()
else:

    if input1d:
        dest_folder = 'C:/Users/Rudy/Desktop/toMartyna/toRUDY/'

        # data = np.load(dest_folder + 'X_noisy_data.npy')
        data = np.load(dest_folder + 'GT_data.npy')
        # data = np.load(dest_folder + 'pred_denoised_DL.npy')
        scaling = np.max(data)
        data = data / scaling
        X_train = data[0:17000, :, :]
        X_val = data[17000:19000, :, :]
        X_test = data[19000:20000, :, :]  # unused

        folder = 'C:/Users/Rudy/Desktop/toMartyna/toRUDY/labels/'
        filenames = ['labels_c_1.mat',
                     'labels_c_2.mat',
                     'labels_c_3.mat',
                     'labels_c_4.mat']
        keyname = 'labels_c'
        y_train, y_val, y_test = labelsimport_md(folder, filenames, keyname)
        nlabels, w_nlabels = labelsNorm(y_test)
        dataset1D = X_test
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

        if doSNR:
            snr_v = sio.loadmat(folder + 'snr_v')
            snr_v_tot = snr_v['snr_v']
            snr_v = snr_v_tot[19000:20000, :]
        if doShim:
            readme_SHIM = sio.loadmat(folder + 'shim_v.mat')
            shim_v_tot = readme_SHIM['shim_v']
            shim_v = shim_v_tot[19000:20000, :]

        folder = 'C:/Users/Rudy/Desktop/datasets/dataset_33/labels/'
        filenames = ['labels_c_1.mat',
                     'labels_c_2.mat',
                     'labels_c_3.mat',
                     'labels_c_4.mat']
        keyname = 'labels_c'

        y_train, y_val, y_test = labelsimport_md(folder, filenames, keyname)
        nlabels, w_nlabels = labelsNorm(y_test)
        dataset2D = X_test

outpath = "C:/Users/Rudy/Desktop/DL_models/"
folder = "net_type/"
subfolder = "decoder_1D_denoising_quant//"
net_name = "Inception-Net-1D2c-v0_trained_on_GT_iter_0"
checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)
model = newModel(dim='1D', type='InceptionNet_1D2c', subtype='v0')
# model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowInception_fact_v2')

model.load_weights(checkpoint_path)

if input1d:
    loss = model.evaluate(dataset1D, nlabels, verbose=2)
    pred_abs = model.predict(dataset1D)  # normalized [0-1] absolute concentrations prediction
else:
    loss = model.evaluate(dataset2D, nlabels, verbose=2)
    pred_abs = model.predict(dataset2D)  # normalized [0-1] absolute concentrations prediction

pred_un = np.empty(pred_abs.shape)  # un-normalized absolute concentrations
pred = np.empty(pred_abs.shape)  # relative un normalized concentrations (referred to water prediction)

pred_un = ilabelsNorm(pred_abs, w_nlabels)
y_test = ilabelsNorm(nlabels, w_nlabels)

for i in range(17):
    pred[:, i] = pred_un[:, i] / pred_un[:, 16] * 64.5
    y_test[:, i] = y_test[:, i] / y_test[:, 16] * 64.5

# for no_water referenced case
# for i in range(16):
#     pred[:, i] = pred_un[:, i]
#     y_test[:, i] = y_test[:, i]

regr = linear_model.LinearRegression()

metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'mI', 'PE', 'sI',
            'Tau', 'Water']

order = [2,0,4,12,7,6,1,8,9,14,10,3,13,15,11,5] # to order metabolites plot from good to bad
max_range = [25.8, 5, 18.5, 14.7, 20, 2, 2.8, 5.8, 2, 0.6, 2, 3.5, 3.3, 2, 1, 1.8] # max concentration per metabolite in the simulation



# ----- savings predictions
if doSavePred:
    #from util import save_scores_tab
    import pandas as pd
    filename = net_name + saving_spec
    filepath = outpath + folder + subfolder

    excelname = filename + ".xlsx"
    #workbook = xlsxwriter.Workbook(filepath + excelname)
    #worksheet = workbook.add_worksheet()

    df1 = pd.DataFrame(pred_un)
    df2 = pd.DataFrame(y_test)
    # create an Excel writer object
    writer = pd.ExcelWriter(filepath +excelname)

    # write each dataframe to a separate sheet in the Excel file
    df1.to_excel(writer, sheet_name='predictions')
    df2.to_excel(writer, sheet_name='ground_truth')
    writer.save()
    #for col_num in range(pred_un.shape[1]):
     #   for row_num in range(pred_un.shape[0]):
      #      worksheet.write(row_num,colum_num,pred_un[row_num,col_num])

    #workbook.close()
    print('xlsx w/ pred SAVED')

from util_plot import plotREGR2x4fromindex, plotSNR2x4fromindex, plotSHIM2x4fromindex

if doSNR:
    if test_diff_conc_bounds:
        plotREGR2x4fromindex(0, y_test, pred, order, metnames, snr=snr_v, yscale = 2, pred_ref = 1, ref_max_v = max_range)
        plotREGR2x4fromindex(8, y_test, pred, order, metnames, snr=snr_v, yscale = 2, pred_ref = 1, ref_max_v = max_range)
    else:
        plotREGR2x4fromindex(0, y_test, pred, order, metnames, snr=snr_v, yscale = 2, pred_ref = 1, ref_max_v = max_range)
        plotREGR2x4fromindex(8, y_test, pred, order, metnames, snr=snr_v, yscale = 2, pred_ref = 1, ref_max_v = max_range)
else:
    if test_diff_conc_bounds:
        plotREGR2x4fromindex(0, y_test, pred, order, metnames, snr=[], yscale = 2, pred_ref = 1, ref_max_v = max_range)
        plotREGR2x4fromindex(8, y_test, pred, order, metnames, snr=[], yscale = 2, pred_ref = 1, ref_max_v = max_range)
    else:
        plotREGR2x4fromindex(0, y_test, pred, order, metnames, snr=[], yscale = 2, pred_ref = 1, ref_max_v = max_range)
        plotREGR2x4fromindex(8, y_test, pred, order, metnames, snr=[], yscale = 2, pred_ref = 1, ref_max_v = max_range)

if doSNR:
    plotSNR2x4fromindex(0, y_test, pred, order, metnames, snr=snr_v)
    plotSNR2x4fromindex(8, y_test, pred, order, metnames, snr=snr_v)

if doShim:
    if doSNR:
        plotSHIM2x4fromindex(0, y_test, pred, order, metnames, shim_v, snr_v)
        plotSHIM2x4fromindex(8, y_test, pred, order, metnames, shim_v, snr_v)
    else:
        plotSHIM2x4fromindex(0, y_test, pred, order, metnames, shim_v, snr=[])
        plotSHIM2x4fromindex(8, y_test, pred, order, metnames, shim_v, snr=[])



# def scores(index):
#     # ----------------------------------------------
#     x = y_test[:, index].reshape(-1, 1)
#     y = pred[:, index]
#     regr.fit(x, y)
#     lin = regr.predict(np.arange(0, np.max(y_test[:, index]), 0.01).reshape(-1, 1))
#     mse = mean_squared_error(x, y)
#     r_sq = regr.score(x, y)
#
#     return regr.coef_[0], regr.intercept_, r_sq, mse

# ----- savings of scores
if doSave:
    from util import save_scores_tab
    filename = net_name + saving_spec_scores
    filepath = outpath + folder + subfolder
    save_scores_tab(filename, filepath, y_test, pred)


if reliability:
    from util_plot import reliability_plot

    def plotRELIABILITY2x4fromindex(i, pred):
        fig = plt.figure(figsize=(40, 20))

        widths = 2 * np.ones(4)
        heights = 2 * np.ones(2)
        spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                height_ratios=heights)

        for row in range(2):
            for col in range(4):
                ax = fig.add_subplot(spec[row, col])

                if (i == 0) or (i == 8):
                    reliability_plot(fig, pred[:,order[i]], 10, metnames[order[i]], outer=spec[row, col], sharey=1)
                elif (i == 4) or (i == 12):
                    reliability_plot(fig, pred[:,order[i]], 10, metnames[order[i]], outer=spec[row, col], sharex=1, sharey=1)
                elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                    reliability_plot(fig, pred[:,order[i]], 10, metnames[order[i]], outer=spec[row, col], sharex=1)
                else:
                    reliability_plot(fig, pred[:,order[i]], 10, metnames[order[i]], outer=spec[row, col])

                i += 1

    plotRELIABILITY2x4fromindex(0, pred_abs)
    plotRELIABILITY2x4fromindex(8, pred_abs)

    # --- calibration
    from data_load_norm import dataimport2D, labelsimport, labelsNorm

    #NB: I assume the training done with dataset_20, nevertheless I start with validation set to calibrate.
    folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'
    dataname = 'dataset_spgram.mat'
    labelsname = 'labels_c.mat'
    X_train, X_val = dataimport2D(folder, dataname, 'dataset')
    y_train, y_val = labelsimport(folder, labelsname, 'labels_c')

    ny_train, w_y_train = labelsNorm(y_train)
    ny_val, w_y_val = labelsNorm(y_val)

    pred_val = model.predict(X_val)  # normalized [0-1] absolute concentrations prediction

    def recalibration_models(pred_val, ny_val, plot=0):
        """
        isotonic regression algorithm that recalibrate predictions from a neutral network model recalibrating the system
        inspired by:
        1. Kuleshow V. et al, Accurate uncertainties for Deep LEarning using Calibrated Regression, 35th ICML 2018
        2. Niculescu-Mizil A. et al, Predicting good probabilities with supervised learning, 22nd ICML 2005

        :param pred_val: Nxm  matrix with absolute [0,1] prediction from a given model on validation dataset; N:number of sample, n: number of metabolites
        :param ny_val: Nxm matrix of ground truth labels on validation set
        :param plot: boolean, =1 if the user wants to monitor the calibration plot pre and post isotonic regression fit.
        :return: list with m isotonic regression models that corrects the prediction of the given model
        """
        iso_reg = []
        for met in range(pred_val.shape[1]):
            pred01 = pred_val[:, met]
            label01 = ny_val[:, met]

            idx = np.argsort(pred01)
            pred01_sort = pred01[idx]
            idx = np.argsort(label01)
            th = label01[idx]
            p_hat = np.zeros(pred01_sort.shape)

            for i in range(pred01_sort.shape[0]):
                p_hat[i] = len(np.argwhere(pred01_sort <= th[i])) / pred01.shape[0]



            from sklearn.isotonic import IsotonicRegression
            iso_reg.append(IsotonicRegression(out_of_bounds='clip').fit(th, p_hat))

            if plot:
                fig = plt.figure()
                plt.title('Reliability diagram - idx: ' + str(met))
                plt.plot(th, p_hat, label='DL calibration')
                cal_pred_plot= iso_reg[-1].predict(pred01_sort)
                plt.plot(th, cal_pred_plot, label='recalibrated')
                plt.legend(loc="upper left")
                plt.xlabel('predicted value')
                plt.ylabel('fraction of positive')

        return iso_reg

    iso_regr_model = recalibration_models(pred_val, ny_val, plot=0)

    # test set recalibration
    pred_abs_cal = np.zeros(pred_abs.shape)

    # pred_abs[pred_abs<0]=0
    # pred_abs[pred_abs>1]=1

    for met in range(pred_abs.shape[1]):
        pred_abs_cal[:, met] = iso_regr_model[met].predict(pred_abs[:, met])


    pred_un_cal = np.empty(pred_abs.shape)  # un-normalized absolute concentrations
    pred_cal = np.empty(pred_abs.shape)  # relative un normalized concentrations (referred to water prediction)

    pred_un_cal = ilabelsNorm(pred_abs_cal, w_nlabels)

    for i in range(17):
        pred_cal[:, i] = pred_un_cal[:, i] / pred_un_cal[:, 16] * 64.5



    # -------------------------------------------------------------
    # plot CALIBRATED regression 2x4
    # -------------------------------------------------------------
    # from util_plot import jointregression
    # def plotREGR_CAL_2x4fromindex(i):
    #     fig = plt.figure(figsize = (40,10))
    #
    #     widths = 2*np.ones(4)
    #     heights = 2*np.ones(2)
    #     spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
    #                               height_ratios=heights)
    #     for row in range(2):
    #         for col in range(4):
    #             ax = fig.add_subplot(spec[row,col])
    #             if (i == 0) or (i == 8):
    #                 jointregression(fig, y_test[:, order[i]], pred_cal[:, order[i]], metnames[order[i]], snr_v=snr_v,
    #                                 outer=spec[row, col], sharey=1)
    #             elif (i == 4) or (i == 12):
    #                 jointregression(fig, y_test[:, order[i]], pred_cal[:, order[i]], metnames[order[i]], snr_v=snr_v,
    #                                 outer=spec[row, col], sharex=1, sharey=1)
    #             elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
    #                 jointregression(fig, y_test[:, order[i]], pred_cal[:, order[i]], metnames[order[i]], snr_v=snr_v,
    #                                 outer=spec[row, col], sharex=1)
    #             else:
    #                 jointregression(fig, y_test[:, order[i]], pred_cal[:, order[i]], metnames[order[i]], snr_v=snr_v,
    #                                 outer=spec[row, col])
    #
    #             i += 1
    #
    # plotREGR_CAL_2x4fromindex(0)
    # plotREGR_CAL_2x4fromindex(8)

    plotREGR2x4fromindex(0, y_test, pred_cal, order, metnames, snr_v)
    plotREGR2x4fromindex(8, y_test, pred_cal, order, metnames, snr_v)

    plotRELIABILITY2x4fromindex(0, pred_abs_cal)
    plotRELIABILITY2x4fromindex(8, pred_abs_cal)


    if doSNR:
        plotSNR2x4fromindex(0, y_test, pred_cal, order, metnames, snr_v)
        plotSNR2x4fromindex(8, y_test, pred_cal, order, metnames, snr_v)

    if doShim:
        if doSNR:
            plotSHIM2x4fromindex(0, y_test, pred_cal, order, metnames, shim_v, snr_v)
            plotSHIM2x4fromindex(8, y_test, pred_cal, order, metnames, shim_v, snr_v)
        else:
            plotSHIM2x4fromindex(0, y_test, pred_cal, order, metnames, shim_v, snr=[])
            plotSHIM2x4fromindex(8, y_test, pred_cal, order, metnames, shim_v, snr=[])


    #--------------

    delta = pred_cal - pred

    fig = plt.figure()
    plt.scatter(snr_v,delta[:,13])

    fig = plt.figure()
    plt.scatter(pred[:,13], pred_cal[:,13])