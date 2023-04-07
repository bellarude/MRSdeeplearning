from __future__ import print_function
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, \
    Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
# from keras import backend as K
import tensorflow.keras.backend as K

from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import kerastuner as kt
from kerastuner import HyperModel, HyperParameters, RandomSearch
from kerastuner.tuners import BayesianOptimization

import time
from util import textMe
from numba import cuda


# X_test_full  (1000, 1024, 2)
def calculate_DQ_DQw(pred, X_test_full, Y_test_full):

    DQ_num = np.zeros((pred.shape[0], 2))
    DQ_denom = np.zeros((pred.shape[0], 2))
    DQw_num = np.zeros((pred.shape[0], 2))
    DQw_denom = np.zeros((pred.shape[0], 2))
    norm = np.zeros((pred.shape[0], 2))

    for i in range(pred.shape[0]):
        I_DN_GT = np.power((pred[i, :, :] - Y_test_full[i, :, :]), 2)
        I_N = np.power((X_test_full[i, :, :] - Y_test_full[i, :, :]), 2)
        abs_I_GT = np.sqrt(np.power(Y_test_full[i, :, :], 2))

        #------
        #RR
        wI_DN_GT = np.multiply(np.abs(Y_test_full[i, :, :]), np.power((pred[i, :, :] - Y_test_full[i, :, :]), 2))
        wI_N = np.multiply(np.abs(Y_test_full[i, :, :]), np.power((X_test_full[i, :, :] - Y_test_full[i, :, :]), 2))
        norm[i,:] = np.sum(np.abs(Y_test_full[i, :, :]), axis=0)
        # ------
        # print('I_DN_GT: ', I_DN_GT.shape)
        # print('I_N: ', I_N.shape)
        # print('abs_I_GT: ', abs_I_GT.shape)

        DQ_num[i, :] = np.sum(I_DN_GT, axis=0)
        DQ_denom[i, :] = np.sum(I_N, axis=0)

        DQw_num[i, :] = np.sum((abs_I_GT * I_DN_GT), axis=0)
        DQw_denom[i, :] = np.sum((abs_I_GT * I_N), axis=0)
        # DQw_num[i,:] = np.sum((Y_test_full[i, :, :], ), axis=0)
        # DQw_denom[i,:] = np.sum(, axis=0)

        # ------
        #RR
        DQw_num[i, :] = np.sum(wI_DN_GT, axis=0)*pred.shape[1]
        #DQw_denom[i, :] = np.sum(wI_N, axis=0)
        DQw_denom[i, :] = np.sum(I_N, axis=0)*norm[i]
        # ------

    DQ_complex = np.sqrt(DQ_num / DQ_denom)
    DQw_complex = np.sqrt(DQw_num / DQw_denom)


    DQ = np.sum(((DQ_complex[:, 0]), (DQ_complex[:, 1])), axis=0)
    DQw = np.sum(((DQw_complex[:, 0]), (DQw_complex[:, 1])), axis=0)

    #DQw_v2 = np.sum(((DQw_complex_v2[:, 0]), (DQw_complex_v2[:, 1])), axis=0)

    return DQ, DQw, DQ_complex, DQw_complex

def calculateDQ_old(pred, X_test_full, Y_test_full):

    DQ_num = np.zeros((pred.shape[0], 2))
    DQ_denom = np.zeros((pred.shape[0], 2))
    DQw_num = np.zeros((pred.shape[0], 2))
    DQw_denom = np.zeros((pred.shape[0], 2))
    norm = np.zeros((pred.shape[0], 2))

    for i in range(pred.shape[0]):

        I_DN_GT = np.power((Y_test_full[i, :, :] - pred[i, :, :]), 2)
        I_N = np.power((X_test_full[i, :, :] - Y_test_full[i, :, :]), 2)
        abs_I_GT = np.sqrt(np.power(Y_test_full[i, :, :], 2))


        DQ_num[i, :] = np.sum(I_DN_GT, axis=0)
        DQ_denom[i, :] = np.sum(I_N, axis=0)

        DQw_num[i, :] = np.sum((np.power(Y_test_full[i, :, :], 2) * I_DN_GT), axis=0)
        DQw_denom[i, :] = np.sum((np.power(Y_test_full[i, :, :], 2) * I_N), axis=0)
        # DQw_num[i,:] = np.sum((Y_test_full[i, :, :], ), axis=0)
        # DQw_denom[i,:] = np.sum(, axis=0)


    DQ_complex = np.sqrt(DQ_num / DQ_denom)
    DQw_complex = np.sqrt(DQw_num / DQw_denom)


    DQ = np.sum(((DQ_complex[:, 0]), (DQ_complex[:, 1])), axis=0)
    DQw = np.sum(((DQw_complex[:, 0]), (DQw_complex[:, 1])), axis=0)

    #DQw_v2 = np.sum(((DQw_complex_v2[:, 0]), (DQw_complex_v2[:, 1])), axis=0)

    return DQ, DQw, DQ_complex, DQw_complex


# pred  (1000, 1024, 2)  (pred_spgram_concat_, X_spgram_concat_, Y_spgram_concat_
def calculateDQ_DQw_NEW(pred, X_test_full, Y_test_full):

    DQ_num = np.zeros((pred.shape[0], pred.shape[2]))
    DQ_denom = np.zeros((pred.shape[0], pred.shape[2]))

    DQw_num = np.zeros((pred.shape[0], pred.shape[2]))
    DQw_denom = np.zeros((pred.shape[0], pred.shape[2]))
    norm = np.zeros((pred.shape[0], pred.shape[2]))
    w_k_sum = np.zeros((pred.shape[0], pred.shape[2]))

    for k in range(pred.shape[0]):

        # DQ --------------------------------------------
        I_DN_GT = np.power((Y_test_full[k, :, :] - pred[k, :, :]), 2)
        I_N = np.power((X_test_full[k, :, :] - Y_test_full[k, :, :]), 2)

        DQ_num[k, :] = np.sum(I_DN_GT, axis=0)
        DQ_denom[k, :] = np.sum(I_N, axis=0)

        # DQw --------------------------------------------l
        w_k = np.abs(Y_test_full[k, :, :])

        DQw_num[k, :] = np.sum(np.multiply(w_k, I_DN_GT), axis=0)
        DQw_denom[k, :] = np.sum(I_N, axis=0)
        w_k_sum[k, :] =  np.sum(w_k, axis=0)

    DQ_2 = DQ_num/DQ_denom
    x1 = np.divide(DQw_num, DQw_denom)
    x2 = np.divide(pred.shape[1],np.sum(w_k_sum))
    DQw_2 = np.multiply(x1, x2)
    #DQw_2 =  np.divide(DQw_num, DQw_denom) * np.divide(pred.shape[1],w_k_sum)

    return DQ_2, DQw_2


def calculateDQ_DQw_v3(pred, X_test_full, Y_test_full):

    DQ_num = np.zeros((pred.shape[0], pred.shape[2]))
    DQ_denom = np.zeros((pred.shape[0], pred.shape[2]))

    DQw_num = np.zeros((pred.shape[0], pred.shape[2]))
    DQw_denom = np.zeros((pred.shape[0], pred.shape[2]))
    norm = np.zeros((pred.shape[0], pred.shape[2]))
    w_k_sum = np.zeros((pred.shape[0], pred.shape[2]))

    for k in range(pred.shape[0]):

        # DQ --------------------------------------------
        I_DN_GT = np.power((pred[k, :, :] - Y_test_full[k, :, :]), 2)
        I_N = np.power((X_test_full[k, :, :] - Y_test_full[k, :, :]), 2)

        DQ_num[k, :] = np.sum(I_DN_GT, axis=0)
        DQ_denom[k, :] = np.sum(I_N, axis=0)

        # DQw --------------------------------------------l
        #w_k = np.abs(Y_test_full[k, :, :])
        w_k = np.power(Y_test_full[k, :, :],2)

        DQw_num[k, :] = np.sum(np.multiply(w_k, I_DN_GT), axis=0)
        DQw_denom[k, :] = np.sum(np.multiply(w_k, I_N), axis=0)

        w_k_sum[k, :] = np.sum(w_k, axis=0)
        #DQw_denom[k, :] = np.divide(np.sum(I_N, axis=0), w_k_sum[k])


    DQ_2 = DQ_num/DQ_denom
    DQw_2 = DQw_num/DQw_denom
    #x1 = np.divide(DQw_num, DQw_denom) * pred.shape[1]
    #x2 = np.divide((pred.shape[1]-1),np.sum(w_k_sum))
    #DQw_2 = np.multiply(x1, x2)
    #DQw_2 =  np.divide(DQw_num, DQw_denom) * np.divide(pred.shape[1],w_k_sum)

    return DQ_2, DQw_2



# X_spgram, Y_spgram, pred_spgram  [1000, 1024, 2]

# ----------------------------------------------------------------------------------------------------
#                               load SNR [ 18000:19000]
# ----------------------------------------------------------------------------------------------------

SNR_folder = 'C:/Users/Rudy/Desktop/datasets/spectra_for_1D_Unet/labels/'
readme = sio.loadmat(SNR_folder + 'readme_4.mat')
SNR = readme['readme']['SNR'][0][0]  # 5000, 1
# ----------------------------------

spgram4096_dir = 'C:/Users/Rudy/Desktop/toMartyna/DeepLearning_MRS/forFig5/zoomedData/spectra/'
X_spgram_4 = sio.loadmat(spgram4096_dir + 'X_noisy_specta_matRecon_4.mat')
Y_spgram_4 = sio.loadmat(spgram4096_dir + 'Y_GT_specta_matRecon_4.mat')
pred_spgram_4 = sio.loadmat(spgram4096_dir + 'pred_spectra_32f_6dr_5_nr1_4.mat')
maxVal = sio.loadmat('C:/Users/Rudy/Desktop/toMartyna/DeepLearning_MRS/forFig5/zoomedData/labels/max_norm_Global_16met_wat_spgram_normGlobal_-1_1.mat')

X_4 = X_spgram_4['X_test_matlabRecon'][:,3000:4000].T      # [4096, 5000] -> [1000, 4096]
Y_4 = Y_spgram_4['Y_test_matlabRecon'][:,3000:4000].T
pred_4 = pred_spgram_4['pred_matlab_GTreco'][:,3000:4000].T      #['pred_matlab_Unet_denois']

X_spgram4 = np.zeros((1000, 4096, 2))
X_spgram4[:,:,0] = np.real(np.fft.fft(X_4))
X_spgram4[:,:,1] = np.imag(np.fft.fft(X_4))

Y_spgram4 = np.zeros((1000, 4096, 2))
Y_spgram4[:,:,0] = np.real(np.fft.fft(Y_4))
Y_spgram4[:,:,1] = np.imag(np.fft.fft(Y_4))

pred_spgram4 = np.zeros((1000, 4096,2))
pred_spgram4[:,:,0] = np.real(np.fft.fft(pred_4))/maxVal['mydata'][0][0]
pred_spgram4[:,:,1] = np.imag(np.fft.fft(pred_4))/maxVal['mydata'][0][0]


# -------------------------------------------------------------------------------------------------------
#                               ----- data to be processd -----
# -------------------------------------------------------------------------------------------------------
X_spgram = X_spgram4[:,-1024:,:]
Y_spgram = Y_spgram4[:,-1024:,:]
pred_spgram = pred_spgram4[:,-1024:,:]

#X_spgram_concat = np.zeros([X_spgram.shape[0], X_spgram.shape[1]*2, 1])
#X_spgram_concat[:,0:1024,1] = X_spgram[:,:,0]

X_spgram_concat = np.concatenate((X_spgram[:,:,0], X_spgram[:,:,1]), axis=1)
X_spgram_concat_ = np.expand_dims(X_spgram_concat, axis=2)

Y_spgram_concat = np.concatenate((Y_spgram[:,:,0], Y_spgram[:,:,1]), axis=1)
Y_spgram_concat_ = np.expand_dims(Y_spgram_concat, axis=2)

pred_spgram_concat = np.concatenate((pred_spgram[:,:,0], pred_spgram[:,:,1]), axis=1)
pred_spgram_concat_ = np.expand_dims(pred_spgram_concat, axis=2)

# -------------------------------------------------------------------------------------------------------
#                               ----- 1D U-net data to be processd -----
# -------------------------------------------------------------------------------------------------------
X_1D_spectra = np.concatenate((X_test_full[:,:,0], X_test_full[:,:,1]), axis=1)
X_1D_spectra = np.expand_dims(X_1D_spectra, axis=2)

Y_1D_spectra = np.concatenate((Y_test_full[:,:,0], Y_test_full[:,:,1]), axis=1)
Y_1D_spectra = np.expand_dims(Y_1D_spectra, axis=2)

pred_1D_spectra = np.concatenate((pred[:,:,0], pred[:,:,1]), axis=1)
pred_1D_spectra = np.expand_dims(pred_1D_spectra, axis=2)
# -------------------------------------------------------------------------------------------------------
#   calculate DQ, DQw for 1D Unet
# ---------------
DQ_2_1D_spectra, DQw_2_1D_spectra = calculateDQ_DQw_v3(pred_1D_spectra, X_1D_spectra, Y_1D_spectra)

'''
saveDir = 'C:/Users/Rudy/Desktop/toMartyna/DQ_DQw_1D_Unet/'
np.save(saveDir + 'DQ_2_1D_spectra.mat', DQ_2_1D_spectra)

from scipy.io import savemat
import numpy as np
import glob
import os

savemat(saveDir + 'DQ_2_1D_spectra.mat', {"data":DQ_2_1D_spectra})
savemat(saveDir + 'DQw_2_1D_spectra.mat', {"data":DQw_2_1D_spectra})

savemat(saveDir + 'pred_1D_spectra_testSet.mat', {"data":pred_1D_spectra})
savemat(saveDir + 'X_1D_spectra_testSet.mat', {"data":X_1D_spectra})
savemat(saveDir + 'Y_1D_spectra_testSet.mat', {"data":Y_1D_spectra})

savemat(saveDir + 'pred2D_Unet_testSet.mat', {"data":pred_spgram_concat_})
savemat(saveDir + 'X_2D_Unet_testSet.mat', {"data":X_spgram_concat_})
savemat(saveDir + 'Y_2D_Unet_testSet.mat', {"data":Y_spgram_concat_})

savemat(saveDir + 'SNR_testSet_nr4_3000to4000.mat', {"data":SNR[3000:4000]})

'''
# -------------------------------------------------------------------------------------------------------
#                calculate DQ, DQw for 2D Unet
#  ------------------------------------------------------------------------------------------------------
DQ_2_2DUnet, DQw_2_2DUnet = calculateDQ_DQw_v3(pred_spgram_concat_, X_spgram_concat_, Y_spgram_concat_)

DQ_2_noisy, DQw_2_noisy = calculateDQ_DQw_NEW(X_spgram_concat_, X_spgram_concat_, Y_spgram_concat_)
DQ_2_GT, DQw_2_GT = calculateDQ_DQw_NEW(Y_spgram_concat_, X_spgram_concat_, Y_spgram_concat_)


# -------------------------------------------------------------------------------------------------------
#                plot DQ, DQw for 1D and 2D UNet
#  ------------------------------------------------------------------------------------------------------
ymin = 0
ymax = 1.4

plt.figure()
plt.scatter(SNR[3000:4000,0], np.sqrt(DQ_2_2DUnet), s=40, facecolors='none', edgecolors='r')
#plt.ylim([0,1])
plt.xlabel('global SNR')
plt.ylabel('DQ')
plt.title('DQ - 2D Unet')
plt.ylim([ymin, ymax])
plt.savefig('C:/Users/Rudy/Desktop/toMartyna/DQ_DQw_1D_Unet/DQ_2DUnet_scaled.png')
plt.show()

plt.figure()
plt.scatter(SNR[3000:4000,0], np.sqrt(DQw_2_2DUnet), s=40, facecolors='none', edgecolors='r')
#plt.ylim([0,1])
plt.xlabel('global SNR')
plt.ylabel('DQw')
plt.title('DQw - 2D Unet')
plt.ylim([ymin, ymax])
plt.savefig('C:/Users/Rudy/Desktop/toMartyna/DQ_DQw_1D_Unet/DQw_2DUnet_scaled.png')
plt.show()

# -------------------------------------------------------------------------------------------------------
#                   PLOT for Fig2
# pred_1D_spectra  Y_1D_spectra  X_1D_spectra
# X_spgram_concat_ Y_spgram_concat_ pred_spgram_concat_  (1000, 2048, 1)
# -------------------------------------------------------------------------------------------------------
pred_1D_plot = pred_1D_spectra
X_1D_plot = X_1D_spectra
Y_1D_plot = Y_1D_spectra
pred_2D_plot = pred_spgram_concat_
X_2D_plot = X_spgram_concat_
Y_2D_plot = Y_spgram_concat_


spgramNr = 5
fig3 = plt.figure(constrained_layout=True, figsize=(20,7))
gs = fig3.add_gridspec(2, 5)

f3_ax1 = fig3.add_subplot(gs[0, 0])
f3_ax1.plot(X_2D_plot[spgramNr,:1023,0], color='#3776ab')
f3_ax1.set_title('noisy/input spectrum')

f3_ax2 = fig3.add_subplot(gs[1, 0])
f3_ax2.plot(Y_2D_plot[spgramNr,:1023,0], color='black')
f3_ax2.set_title('ground truth (GT)')

f3_ax3 = fig3.add_subplot(gs[0:2, 1:3])
f3_ax3.plot(Y_2D_plot[spgramNr,:1023,0], label='GT', color='black')
f3_ax3.plot(pred_2D_plot[spgramNr,:1023,0], label='pred 2D', color='darkorange')
f3_ax3.plot(pred_1D_plot[spgramNr,:1023,0], label='pred 1D', color='forestgreen')
f3_ax3.legend()
f3_ax3.set_title('GT and 2D Unet prediction')

f3_ax4 = fig3.add_subplot(gs[0, 3])
f3_ax4.plot(pred_1D_plot[spgramNr,:1023,0] - Y_1D_plot[spgramNr,:1023,0], color='forestgreen')
f3_ax4.set_ylim([-0.25,0.25])
f3_ax4.set_title('1D Unet residues')

f3_ax5 = fig3.add_subplot(gs[1, 3])
f3_ax5.plot(pred_2D_plot[spgramNr,:1023,0] - Y_2D_plot[spgramNr,:1023,0], color='darkorange')
f3_ax5.set_ylim([-0.25,0.25])
f3_ax5.set_title('2D Unet residues')

f3_ax6 = fig3.add_subplot(gs[0, 4])
f3_ax6.plot(X_2D_plot[spgramNr,:1023,0] - Y_2D_plot[spgramNr,:1023,0], color='#3776ab')
f3_ax6.set_ylim([-0.5,0.5])
f3_ax6.set_title('noise')
fig3.savefig('C:/Users/Rudy/Desktop/toMartyna/DQ_DQw_1D_Unet/comparison_1D_2D_5_DQw_0_04_1DUnet.png')

# ----------------------------
pred_plot = pred_1D_spectra
X_plot = X_1D_spectra
Y_plot = Y_1D_spectra


fig, axs = plt.subplots(5, figsize=(10,15))      #, figsize=(10,15)
#fig.suptitle('2D Unet')
axs[0].plot(pred_plot[spgramNr,:1023,0], color='orange')
axs[0].set_title('denoised')
axs[0].invert_xaxis()
#axs[0].autoscale()
#axs[0].annotate('DQ =', np.sqrt(DQ_2[0]))
axs[1].plot(X_plot[spgramNr,:1023,0])
axs[1].set_title('X_noisy')
axs[1].invert_xaxis()
#axs[1].autoscale()
axs[2].plot(Y_plot[spgramNr,:1023,0])
axs[2].set_title('Y_GT')
axs[2].invert_xaxis()
#axs[2].autoscale()
axs[3].plot(Y_plot[spgramNr,:1023,0]-pred_plot[spgramNr,:1023,0])
axs[3].set_title('residues')
axs[3].set_ylim([-0.25,0.25])
axs[3].invert_xaxis()
#axs[3].autoscale()
axs[4].plot(X_plot[spgramNr,:1023,0]-Y_plot[spgramNr,:1023,0])
axs[4].set_title('noise')
axs[4].set_ylim([-0.5,0.5])
axs[4].invert_xaxis()
#axs[4].autoscale()
fig.subplots_adjust(hspace=0.35)


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
'''
DQw > 1  ~1.2  [5]
DQw = 1        [9]
DQw < 1  ~0.04 [20]
2D Unet: pred_spgram_concat_, X_spgram_concat_, Y_spgram_concat_
1D Unet: pred_1D_spectra, X_1D_spectra, Y_1D_spectra)
'''

pred_plot = pred_1D_spectra
X_plot = X_1D_spectra
Y_plot = Y_1D_spectra

spgramNr = 20

fig, axs = plt.subplots(5, figsize=(10,15))      #, figsize=(10,15)
#fig.suptitle('2D Unet')
axs[0].plot(pred_plot[spgramNr,:1023,0], color='orange')
axs[0].set_title('denoised')
axs[0].invert_xaxis()
#axs[0].autoscale()
#axs[0].annotate('DQ =', np.sqrt(DQ_2[0]))
axs[1].plot(X_plot[spgramNr,:1023,0])
axs[1].set_title('X_noisy')
axs[1].invert_xaxis()
#axs[1].autoscale()
axs[2].plot(Y_plot[spgramNr,:1023,0])
axs[2].set_title('Y_GT')
axs[2].invert_xaxis()
#axs[2].autoscale()
axs[3].plot(Y_plot[spgramNr,:1023,0]-pred_plot[spgramNr,:1023,0])
axs[3].set_title('residues')
axs[3].set_ylim([-0.25,0.25])
axs[3].invert_xaxis()
#axs[3].autoscale()
axs[4].plot(X_plot[spgramNr,:1023,0]-Y_plot[spgramNr,:1023,0])
axs[4].set_title('noise')
axs[4].set_ylim([-0.5,0.5])
axs[4].invert_xaxis()
#axs[4].autoscale()
fig.subplots_adjust(hspace=0.35)
fig.savefig('C:/Users/Rudy/Desktop/toMartyna/DQ_DQw_1D_Unet/spectra_20_DQw_1_02_1DUnet.png')

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------






DQ_2Dspgram, DQw_2Dspgram, DQ_complex, DQw_complex = calculate_DQ_DQw(pred_spgram4, X_spgram4, Y_spgram4)
#DQ_2Dspgram, DQw_2Dspgram, DQ_complex, DQw_complex = calculate_DQ_DQw_v2(pred_spgram4, X_spgram4, Y_spgram4)
DQ_old, DQw_old, DQ_complex_old, DQw_complex_old = calculateDQ_old(pred_spgram4, X_spgram4, Y_spgram4)


plt.figure()
plt.scatter(SNR[3000:4000,0], DQ_2_GT, s=40, facecolors='none', edgecolors='r')
#plt.ylim([0,1])
plt.xlabel('global SNR')
plt.ylabel('DQ value')
plt.title('DQ on GT')
#plt.ylim([0,3.5])
plt.show()

plt.figure()
plt.scatter(SNR[3000:4000,0], DQw_2_GT, s=40, facecolors='none', edgecolors='r')
#plt.ylim([0,1])
plt.xlabel('global SNR')
plt.ylabel('DQw onGT')
plt.title('DQw old')
#plt.ylim([0,3.5])
plt.show()

# ----------------------
plt.figure()
plt.scatter(SNR[3000:4000,0], DQ_2Dspgram, s=40, facecolors='none', edgecolors='r')
#plt.ylim([0,1])
plt.xlabel('global SNR')
plt.ylabel('DQ value')
plt.title('DQ new')
plt.ylim([0,3.5])
plt.show()

plt.figure()
plt.scatter(SNR[3000:4000,0], DQw_2Dspgram, s=40, facecolors='none', edgecolors='r')
#plt.ylim([0,1])
plt.xlabel('global SNR')
plt.ylabel('DQw value')
plt.title('DQw new')
#plt.ylim([0,3.5])
plt.show()

# --------------
plt.figure()
plt.scatter(SNR[3000:4000,0], DQ_complex[:,1], s=40, facecolors='none', edgecolors='r')
#plt.ylim([0,1])
plt.xlabel('global SNR')
plt.ylabel('DQ_spgram (2D) value')
plt.ylim([0,3.5])
plt.show()

plt.figure()
plt.scatter(SNR[3000:4000,0], DQw_complex[:,1], s=40, facecolors='none', edgecolors='r')
#plt.ylim([0,1])
plt.xlabel('global SNR')
plt.ylabel('DQw_spgram (2D) value')
plt.ylim([0,3.5])
plt.show()

# ----------------------

DQ_1D, DQw_1D, DQ_1Dcomplex, DQw_1Dcomplex = calculate_DQ_DQw(pred, X_test_full, Y_test_full)


plt.figure()
plt.scatter(SNR[3000:4000,0], DQ_1D[:], s=40, facecolors='none', edgecolors='r')
plt.ylim([0,3.5])
plt.xlabel('global SNR')
plt.ylabel('DQ 1D-spectrum value')
plt.show()

plt.figure()
plt.scatter(SNR[3000:4000,0], DQw_1D[:], s=40, facecolors='none', edgecolors='r')
plt.ylim([0,3.5])
plt.xlabel('global SNR')
plt.ylabel('DQw 1D-spectrum value')
plt.show()





'''
DQ_num = np.zeros((pred.shape[0],2))
DQ_denom =  np.zeros((pred.shape[0],2))
DQw_num =  np.zeros((pred.shape[0],2))
DQw_denom =  np.zeros((pred.shape[0],2))

for i in range(pred.shape[0]):

    I_DN_GT = np.power((pred[i, :, :] - Y_test_full[i, :, :]), 2)
    I_N = np.power((X_test_full[i, :,:] - Y_test_full[i, :,:]),2)
    abs_I_GT = np.sqrt(np.power(Y_test_full[i, :, :],2))

    # print('I_DN_GT: ', I_DN_GT.shape)
    #print('I_N: ', I_N.shape)
    #print('abs_I_GT: ', abs_I_GT.shape)

    DQ_num[i,:] = np.sum( I_DN_GT,  axis=0)
    DQ_denom[i,:] = np.sum(I_N, axis=0)

    DQw_num[i,:] = np.sum( (abs_I_GT*I_DN_GT) ,axis=0)
    DQw_denom[i,:] = np.sum( (abs_I_GT*I_N ),axis=0)
    #DQw_num[i,:] = np.sum((Y_test_full[i, :, :], ), axis=0)
    #DQw_denom[i,:] = np.sum(, axis=0)



DQ_complex = np.sqrt(DQ_num/DQ_denom)
DQw_complex = np.sqrt(DQw_num/DQw_denom)

DQ = np.sum(((DQ_complex[:,0]), (DQ_complex[:,1])), axis=0)
DQw = np.sum(((DQw_complex[:,0]), (DQw_complex[:,1])), axis=0)
'''