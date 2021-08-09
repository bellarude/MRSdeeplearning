from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
#from keras import backend as K
import tensorflow.keras.backend as K

from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperModel, HyperParameters, RandomSearch
from kerastuner.tuners import BayesianOptimization

import time
import xlsxwriter

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from data_load_norm import dataNorm, labelsNorm, ilabelsNorm, inputConcat2D, inputConcat1D, dataimport2D, labelsimport, dataimport1D
from models import newModel

dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'
def testimport():
    global dataset2D, dataset1D, nlabels, w_nlabels

    data2D_import = sio.loadmat(dest_folder + 'dataset_spgram_TEST.mat')
    data1D_import = sio.loadmat(dest_folder + 'dataset_spectra_TEST.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_c_TEST_abs.mat')

    dataset2D = data2D_import['output']
    dataset1D = data1D_import['dataset_spectra']
    labels = labels_import['labels_c']*64.5

    #reshaping
    # dataset2D_rs = np.transpose(dataset2D, (0, 3, 1, 2))
    dataset1D = np.transpose(dataset1D, (0, 2, 1))
    labels = np.transpose(labels,(1,0))

    nlabels, w_nlabels = labelsNorm(labels)

    return dataset2D, dataset1D, nlabels, w_nlabels


testimport()
dataset1D_flat = inputConcat1D(dataset1D)

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"

net_names = ['ShallowELU_hp',
             'ShallowELU_hp2',
             'ShallowELU_hp3',
             'ShallowELU_hp4',
             'ShallowELU_hp5',
             'ResNet_fed_hp',
             'ResNet_fed_hp2',
             'ResNet_fed_hp3',
             'ResNet_fed_hp4',
             'ResNet_fed_hp5']

model_type = {0: ['2D', 'ShallowCNN', 'ShallowELU_hp'],
              1: ['2D', 'ShallowCNN', 'ShallowELU_hp'],
              2: ['2D', 'ShallowCNN', 'ShallowELU_hp'],
              3: ['2D', 'ShallowCNN', 'ShallowELU_hp'],
              4: ['2D', 'ShallowCNN', 'ShallowELU_hp'],
              5: ['1D', 'ResNet', 'ResNet_fed_hp'],
              6: ['1D', 'ResNet', 'ResNet_fed_hp'],
              7: ['1D', 'ResNet', 'ResNet_fed_hp'],
              8: ['1D', 'ResNet', 'ResNet_fed_hp'],
              9: ['1D', 'ResNet', 'ResNet_fed_hp']}

# model = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
members = list()
for i in range(len(net_names)):
    checkpoint_path = outpath + folder + net_names[i] + ".best.hdf5"
    model = newModel(dim=model_type[i][0], type=model_type[i][1], subtype=model_type[i][2])
    model.load_weights(checkpoint_path)
    members.append(model)

# net_name1 = "ShallowELU_hp"
# checkpoint_path1 = outpath + folder + net_name1 + ".best.hdf5"
# model1 = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
# model1.load_weights(checkpoint_path1)
#
# net_name2 = "ShallowELU_hp2"
# checkpoint_path2 = outpath + folder + net_name2 + ".best.hdf5"
# model2 = newModel(dim='2D', type='ShallowCNN', subtype='ShallowELU_hp')
# model2.load_weights(checkpoint_path2)

#source: https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
# -----------------------------------------------------------------------------------------------------
# train, val, test sets defined entirely over an unseen testset about 2500k by the ensebled networks!

# train2D = dataset2D[0:1500, :,:,:]
# val2D = dataset2D[1500:2000,:,:,:]
# test2D = dataset2D[2000:2500, :,:,:]
# labels_train = nlabels[0:1500, :]
# labels_val = nlabels[1500:2000, :]
# labels_test = nlabels[2000:2500, :]
# -----------------------------------------------------------------------------------------------------

folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'
dataname = 'dataset_spgram.mat'
X_train, X_val = dataimport2D(folder, dataname, 'dataset')

labelsname = 'labels_c.mat'
y_train, y_val = labelsimport(folder, labelsname, 'labels_c')

ny_train, w_y_train = labelsNorm(y_train)
ny_val, w_y_val = labelsNorm(y_val)

train2D = X_train
val2D = X_val
test2D = dataset2D
labels_train = ny_train
labels_val = ny_val
labels_test = nlabels

dataname = 'dataset_spectra.mat'
X_train, X_val = dataimport1D(folder, dataname, 'dataset_spectra')
X_train_flat = inputConcat1D(X_train)
X_val_flat = inputConcat1D(X_val)

train1D = X_train_flat
val1D = X_val_flat
test1D = dataset1D_flat
# -----------------------------------------------------------------------------------------------------

# members = [model1, model2]

# define stacked model from multiple member input models
from keras import layers
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate

def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name

    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    concat = lambda a, b: layers.Concatenate(axis=-1)([a, b])

    for i in range(len(members)-1):
        if i == 0:
            merge = concat(ensemble_outputs[0], ensemble_outputs[1])
        else:
            merge = concat(merge, ensemble_outputs[i+1])

    hidden = Dense(1000, activation='relu')(merge)
    hidden = Dense(500, activation='relu')(hidden)
    output = Dense(17, activation=None)(hidden)
    stacked_model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    # plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    stacked_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),  # 2e-4 as starter
            loss=tf.keras.losses.MeanSquaredError(),
            experimental_run_tf_function=False)
    return stacked_model

# define ensemble model
stacked_model = define_stacked_model(members)

stacked_model.summary()

v = 0
# fit a stacked model
def fit_stacked_model(model, train2D, val2D, labels_train, labels_val):
    # prepare input data
    X_train = [train2D for _ in range(len(model.input))]
    X_val = [val2D for _ in range(len(model.input))]
    # encode output data
    # inputy_enc = to_categorical(inputy)
    # fit model
    # model.fit(X, labels_1, epochs=300, verbose=1)
    outpath = 'C:/Users/Rudy/Desktop/DL_models/'
    folder = "net_type/"
    net_name = "ensemble"

    checkpoint_path = outpath + folder + net_name + ".best.hdf5"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                         save_weights_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Reduce learning rate when a metric has stopped improving
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # selected channel 0 to keep only Re(spectrogram)
    history = model.fit(X_train, labels_train,
                        epochs=300,
                        batch_size=50,
                        shuffle=True,
                        validation_data=(X_val, labels_val),
                        validation_freq=1,
                        callbacks=[es, mc],
                        verbose=1)

    fig = plt.figure(figsize=(10, 10))
    # summarize history for loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model losses')
    plt.xlabel('epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def fit_stacked_hybrid_model(model, train2D, val2D, train1D, val1D, labels_train, labels_val):
    # prepare input data
    X_train = []
    X_val = []
    for i in range(len(model.input)):
        if model_type[i][0] == '2D':
            X_train.append(train2D)
            X_val.append(val2D)
        else:
            X_train.append(train1D)
            X_val.append(val1D)

    # encode output data
    # inputy_enc = to_categorical(inputy)
    # fit model
    # model.fit(X, labels_1, epochs=300, verbose=1)
    outpath = 'C:/Users/Rudy/Desktop/DL_models/'
    folder = "net_type/"
    net_name = "ensemble_hybrid_n2"

    checkpoint_path = outpath + folder + net_name + ".best.hdf5"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                         save_weights_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Reduce learning rate when a metric has stopped improving
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # selected channel 0 to keep only Re(spectrogram)
    history = model.fit(X_train, labels_train,
                        epochs=300,
                        batch_size=50,
                        shuffle=True,
                        validation_data=(X_val, labels_val),
                        validation_freq=1,
                        callbacks=[es, mc],
                        verbose=1)

    fig = plt.figure(figsize=(10, 10))
    # summarize history for loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model losses')
    plt.xlabel('epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# fit stacked model on test dataset
# fit_stacked_model(stacked_model, train2D, val2D, labels_train, labels_val)
fit_stacked_hybrid_model(stacked_model, train2D, val2D, train1D, val1D, labels_train, labels_val)

# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)

def evaluate_stacked_model(model, inputX, labelX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.evaluate(X, labelX, verbose=2)

# make a prediction with a stacked model
def predict_hybrid_stacked_model(model, input2D, input1D):
    # prepare input data
    X = []
    for i in range(len(model.input)):
        if model_type[i][0] == '2D':
            X.append(input2D)
        else:
            X.append(input1D)

    # make prediction
    return model.predict(X, verbose=0)

def evaluate_hybrid_stacked_model(model, input2D, input1D, labelX):
    # prepare input data
    X = []
    for i in range(len(model.input)):
        if model_type[i][0] == '2D':
            X.append(input2D)
        else:
            X.append(input1D)
    # make prediction
    return model.evaluate(X, labelX, verbose=2)

# make predictions and evaluate
def rel2absPred(p_abs, w_nlabels):
    # p_abs = model.predict(testset)
    p_un = ilabelsNorm(p_abs, w_nlabels)

    p = np.empty(p_abs.shape)
    for i in range(17):
        p[:, i] = p_un[:, i] / p_un[:, 16] * 64.5

    return p

pred = list()
loss = list()
for m in range(len(members)):
    if model_type[m][0] == '2D':
        p_abs = members[m].predict(test2D)
        loss.append(members[m].evaluate(test2D, labels_test, verbose=2))
    else:
        p_abs = members[m].predict(test1D)
        loss.append(members[m].evaluate(test1D, labels_test, verbose=2))

    p = rel2absPred(p_abs, w_nlabels)
    pred.append(p)


# p_abs_e = predict_stacked_model(stacked_model, test2D)
p_abs_e = predict_hybrid_stacked_model(stacked_model, test2D, test1D)
p_e = rel2absPred(p_abs_e, w_nlabels)

pred.append(p_e)
# loss.append(evaluate_stacked_model(stacked_model, test2D, labels_test))
loss.append(evaluate_hybrid_stacked_model(stacked_model, test2D, test1D, labels_test))

y_test = ilabelsNorm(labels_test, w_nlabels)
for i in range(17):
    y_test[:, i] = y_test[:, i] / y_test[:, 16] * 64.5

# pred_abs_ensemble = predict_stacked_model(stacked_model, test2D)
# pred_abs_1 = model1.predict(test2D)
# pred_abs_2 = model2.predict(test2D)

# loss1 = model1.evaluate(test2D, labels_test, verbose=2)
# loss2 = model2.evaluate(test2D, labels_test, verbose=2)
# lossE = evaluate_stacked_model(stacked_model, test2D, labels_test)

# pred_un = np.empty(pred_abs_1.shape)  # un-normalized absolute concentrations
# pred1 = np.empty(pred_abs_1.shape)  # relative un normalized concentrations (referred to water prediction)
# pred2 = np.empty(pred_abs_1.shape)
# predE = np.empty(pred_abs_1.shape)
#
# pred_un_1 = ilabelsNorm(pred_abs_1, w_nlabels)
# pred_un_2 = ilabelsNorm(pred_abs_2, w_nlabels)
# pred_un_ensemble = ilabelsNorm(pred_abs_ensemble, w_nlabels)
# y_test = ilabelsNorm(labels_test, w_nlabels)
#
# for i in range(17):
#     pred1[:, i] = pred_un_1[:, i] / pred_un_1[:, 16] * 64.5
#     pred2[:, i] = pred_un_2[:, i] / pred_un_2[:, 16] * 64.5
#     predE[:, i] = pred_un_ensemble[:, i] / pred_un_ensemble[:, 16] * 64.5
#     y_test[:, i] = y_test[:, i] / y_test[:, 16] * 64.5


regr = linear_model.LinearRegression()
metnames = ['tCho', 'NAAG', 'NAA', 'Asp', 'tCr', 'GABA', 'Glc', 'Glu', 'Gln', 'GSH', 'Gly', 'Lac', 'Myo', 'PE', 'Scy',
            'Tau', 'Water']

def subplotconcentration(index, pred):
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

# cc= np.empty((3,1))
# rr=cc
# mm=cc

# idx = 11
# fig = plt.figure(figsize=(18, 6))
# plt.subplot(3, 3, 1)
# cc[0], rr[0], mm[0] = subplotconcentration(idx, pred1)
# plt.subplot(3, 3, 4)
# cc[1], rr[1], mm[1] = subplotconcentration(idx, pred2)
# plt.subplot(3, 3, 7)
# cc[2], rr[2], mm[2] = subplotconcentration(idx, predE)
#
# plt.subplot(3, 3, 2)
# plt.hist(pred1[:, idx], 20)
# plt.title('PRED distribution')
# plt.subplot(3, 3, 5)
# plt.hist(pred2[:, idx], 20)
# plt.title('PRED distribution')
# plt.subplot(3, 3, 8)
# plt.hist(predE[:, idx], 20)
# plt.title('PRED distribution')
#
# plt.subplot(3, 3, 3)
# plt.hist(y_test[:, idx])
# plt.title('GT distribution')
# plt.subplot(3, 3, 6)
# plt.hist(y_test[:, idx])
# plt.title('GT distribution')
# plt.subplot(3, 3, 9)
# plt.hist(y_test[:, idx])
# plt.title('GT distribution')

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"
workbook = xlsxwriter.Workbook(outpath + folder + 'ensemble_eval.xlsx')
worksheet = workbook.add_worksheet()

for j in range(len(pred)):
    for i in range(16):
        c, r, m = subplotconcentration(i, pred[j])
        # s = 'A' + str(i * 3 + 1)
        row = i * 3 + 1
        col = j
        # worksheet.write(s, c)
        worksheet.write(row, col, c)
        # s = 'A' + str(i * 3 + 2)
        row = i * 3 + 2
        worksheet.write(row, col, r)
        # s = 'A' + str(i * 3 + 3)
        row = i * 3 + 3
        worksheet.write(row, col, m)

    # c, r, m = subplotconcentration(i, pred2)
    # s = 'B' + str(i * 3 + 1)
    # worksheet.write(s, c)
    # s = 'B' + str(i * 3 + 2)
    # worksheet.write(s, r)
    # s = 'B' + str(i * 3 + 3)
    # worksheet.write(s, m)
    #
    # c, r, m = subplotconcentration(i, predE)
    # s = 'C' + str(i * 3 + 1)
    # worksheet.write(s, c)
    # s = 'C' + str(i * 3 + 2)
    # worksheet.write(s, r)
    # s = 'C' + str(i * 3 + 3)
    # worksheet.write(s, m)

workbook.close()
print('xlsx SAVED')