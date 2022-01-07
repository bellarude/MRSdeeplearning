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
from util import textMe

def newModel(met=[], type=0):
    """
    :param met: integer, =[] when the model is fixed for different metabolite, equal = 1 to 17 when it is specific
                TODO: specific metabolite dependent optimized network for variational case
    :param type: integer, 0: traditional model, 1: variational model
    :return: model function aiming to predict metabolite basis set from real-1-channel-spectra passed as input
    """

    K.set_image_data_format('channels_last')
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 2

    datapoints = 1406
    channels = 1  # number of channels
    input_shape = (datapoints, channels)
    inputs = Input(shape=input_shape)

    # --- Define kwargs dictionary
    kwargs = {
        'strides': (1),
        'padding': 'same'}

    # --- Define poolargs dictionary
    poolargs = {
        'pool_size': (2),
        'strides': (2)}

    # -----------------------------------------------------------------------------
    # Define lambda functions
    # -----------------------------------------------------------------------------

    conv = lambda x, kernel_size, filters: layers.Conv1D(filters=filters, kernel_size=kernel_size, **kwargs)(x)
    conv_s = lambda x, strides, filters: layers.Conv1D(filters=filters, kernel_size=3, strides=strides, padding='same')(
        x)
    # --- Define stride-1, stride-2 blocks
    conv1 = lambda filters, x: relu(norm(conv_s(x, filters=filters, strides=1)))
    conv1_lin = lambda filters, x: norm(conv_s(x, filters=filters, strides=1))
    conv2 = lambda filters, x: relu(norm(conv_s(x, filters=filters, strides=2)))
    # --- Define single transpose
    tran = lambda x, filters, strides: layers.Conv1DTranspose(filters=filters, strides=strides, kernel_size=3,
                                                              padding='same')(x)
    # --- Define transpose block
    tran1 = lambda filters, x: relu(norm(tran(x, filters, strides=1)))
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=2)))

    norm = lambda x: layers.BatchNormalization(axis=channel_axis)(x)
    relu = lambda x: layers.ReLU()(x)
    maxP = lambda x, pool_size, strides: layers.MaxPooling1D(pool_size=pool_size, strides=strides)(x)

    flatten = lambda x: layers.Flatten()(x)
    dense = lambda units, x: layers.Dense(units=units)(x)

    convBlock = lambda x, kernel_size, filters: relu(norm(conv(x, kernel_size, filters)))
    convBlock2 = lambda x, kernel_size, filters: convBlock(convBlock(x, kernel_size, filters), kernel_size, filters)

    convBlock_lin = lambda x, kernel_size, filters: norm(conv(x, kernel_size, filters))
    convBlock2_lin = lambda x, kernel_size, filters: convBlock(convBlock(x, kernel_size, filters), kernel_size, filters)

    concat = lambda a, b: layers.Concatenate(axis=channel_axis)([a, b])

    def concatntimes(x, n):
        output = concat(x, x)
        for i in range(n - 1):
            output = concat(output, output)
        return output

    add = lambda x, y: layers.Add()([x, y])
    ResidualBlock = lambda x, y: relu(add(x, y))

    dropout = lambda x, percentage, size: layers.Dropout(percentage, size)(x)

    pad = 9


    if met == []:

        if type == 0:
            # -----------------------------------------------------------------------------
            # RR-Unet 2xconv1
            # -----------------------------------------------------------------------------
            # --- Define contracting layers
            #
            l1 = conv1(64, conv1(32, tf.keras.layers.ZeroPadding1D(padding=(pad))(inputs)))
            l2 = conv1(128, conv1(64, conv2(32, l1)))
            l3 = conv1(256, conv1(128, conv2(48, l2)))
            l4 = conv1(512, conv1(256, conv2(64, l3)))
            l5 = conv1(256, conv1(512, conv2(80, l4)))

            # --- Define expanding layers
            l6 = tran2(256, l5)

            # --- Define expanding layers
            l7 = tran2(128, tran1(64, tran1(64, concat(l4, l6))))
            l8 = tran2(64, tran1(48, tran1(48, concat(l3, l7))))
            l9 = tran2(32, tran1(32, tran1(32, concat(l2, l8))))
            l10 = conv1_lin(32, conv1(32, l9)) # RR:03.012021 recall that old model with same==1 was tuned without linear activation here: conv1(32, conv1(32, l9))

            # --- Create logits
            outputs = tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=1))
            loss_f = tf.keras.losses.MeanSquaredError()
            lrate = 1e-3

        else:
            # -----------------------------------------------------------------------------
            # RR-Unet 2xconv1 -> 2x nets with mu and sigma estimation
            # -----------------------------------------------------------------------------

            # ----- mu network
            # -----------------------------------------------------------------------------

            # --- Define contracting layers
            #
            l1 = conv1(64, conv1(32, tf.keras.layers.ZeroPadding1D(padding=(pad))(inputs)))
            l2 = conv1(128, conv1(64, conv2(32, l1)))
            l3 = conv1(256, conv1(128, conv2(48, l2)))
            l4 = conv1(512, conv1(256, conv2(64, l3)))
            l5 = conv1(256, conv1(512, conv2(80, l4)))

            # --- Define expanding layers
            l6 = tran2(256, l5)

            # --- Define expanding layers
            l7 = tran2(128, tran1(64, tran1(64, concat(l4, l6))))
            l8 = tran2(64, tran1(48, tran1(48, concat(l3, l7))))
            l9 = tran2(32, tran1(32, tran1(32, concat(l2, l8))))
            l10 = conv1_lin(32, conv1(32, l9))

            # --- Create logits
            mu = tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=1))

            # ----- sigma network
            # -----------------------------------------------------------------------------

            # --- Define contracting layers
            #
            l11 = conv1(64, conv1(32, tf.keras.layers.ZeroPadding1D(padding=(pad))(inputs)))
            l21 = conv1(128, conv1(64, conv2(32, l11)))
            l31 = conv1(256, conv1(128, conv2(48, l21)))
            l41 = conv1(512, conv1(256, conv2(64, l31)))
            l51 = conv1(256, conv1(512, conv2(80, l41)))

            # --- Define expanding layers
            l61 = tran2(256, l51)

            # --- Define expanding layers
            l71 = tran2(128, tran1(64, tran1(64, concat(l41, l61))))
            l81 = tran2(64, tran1(48, tran1(48, concat(l31, l71))))
            l91 = tran2(32, tran1(32, tran1(32, concat(l21, l81))))
            l101 = conv1_lin(32, conv1(32, l91))

            # --- Create logits
            sigma = tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l101, kernel_size=3, filters=1))

            assert sigma.shape[1] == 1406, sigma.shape[1]
            assert sigma.shape[2] == 1, sigma.shape[2]

            # --- Create model
            model_mu = Model(inputs=inputs, outputs=mu)
            model_cov = Model(inputs=inputs, outputs=sigma)
            outputs = concat(model_mu.output, model_cov.output)

            lrate = 1e-3

            def wmse_loss_unet():
                def loss(ytrue, ypreds):
                    # n_dims = int(int(ypreds.shape[1]) / 2)
                    mu = ypreds[:, :, 0]  # 1st channel is mu
                    logsigma = ypreds[:, :, 1]  # 2nd channel is sigma

                    loss1 = tf.reduce_mean(tf.exp(-logsigma) * tf.square((mu - ytrue)))
                    loss2 = tf.reduce_mean(logsigma)
                    loss = .5 * (loss1 + loss2)

                    return loss

                return loss

            loss_f = wmse_loss_unet()

    else:
        tCho_k = [30, 30, 60, 60, 140, 70, 370, 250, 240]
        naag_k = [10, 20, 40, 70, 140, 80, 190, 250, 560]
        naa_k = [40, 50, 70, 100, 160, 80, 210, 320, 600]
        asp_k = [20, 20, 90, 80, 100, 170, 120, 320, 650]
        tCr_k = [30, 30, 50, 30, 130, 160, 380, 130, 590]
        gaba_k = [20, 40, 50, 70, 150, 110, 210, 210, 580]
        glc_k = [30, 30, 40, 70, 90, 160, 190, 260, 330]
        glu_k = [20, 30, 40, 50, 80, 130, 290, 250, 190]
        gln_k = [20, 40, 60, 70, 180, 170, 220, 100, 570]
        gsh_k = [30, 40, 90, 60, 40, 130, 240, 380, 250]
        gly_k = [20, 10, 30, 90, 100, 130, 130, 90, 260]
        lac_k = [10, 10, 30, 70, 100, 130, 360, 240, 170]
        mi_k = [50, 30, 20, 100, 50, 70, 90, 200, 800]
        pe_k = [20, 40, 80, 40, 150, 170, 80, 390, 600]
        si_k = [20, 40, 80, 40, 150, 170, 80, 390, 600]
        # si_k = [40, 20, 70, 80, 50, 90, 280, 280, 310] #correct one
        tau_k = [30, 30, 80, 50, 60, 100, 370, 160, 590]
        wat_k = [50, 10, 20, 100, 200, 40, 400, 80, 160]

        tCho_lr = 0.000889448
        naag_lr = 0.001127238
        naa_lr = 0.000651682
        asp_lr = 0.006484923
        tCr_lr = 0.001581492
        gaba_lr = 0.001278078
        glc_lr = 0.002948043
        glu_lr = 0.000924686
        gln_lr = 0.00050156
        gsh_lr = 0.008352496
        gly_lr = 0.00777366
        lac_lr = 0.001482918
        mi_lr = 0.002843735
        pe_lr = 0.002223201
        si_lr = 0.0039873
        tau_lr = 0.009666607
        wat_lr = 0.000439969

        ks = {'tCho': tCho_k, 'NAAG': naag_k, 'NAA': naa_k, 'Asp': asp_k, 'tCr': tCr_k, 'GABA': gaba_k,
              'Glc': glc_k, 'Glu': glu_k, 'Gln': gln_k, 'GSH': gsh_k, 'Gly': gly_k, 'Lac': lac_k, 'mI': mi_k,
              'PE': pe_k,
              'sI': si_k, 'Tau': tau_k, 'Water': wat_k}  # kernel size

        lr = {'tCho': tCho_lr, 'NAAG': naag_lr, 'NAA': naa_lr, 'Asp': asp_lr, 'tCr': tCr_lr, 'GABA': gaba_lr,
              'Glc': glc_lr, 'Glu': glu_lr, 'Gln': gln_lr, 'GSH': gsh_lr, 'Gly': gly_lr, 'Lac': lac_lr, 'mI': mi_lr,
              'PE': pe_lr,
              'sI': si_lr, 'Tau': tau_lr, 'Water': wat_lr}  # learning rate,

        l1 = conv1(ks[met][0] * 2, conv1(ks[met][0], tf.keras.layers.ZeroPadding1D(padding=(pad))(inputs)))
        l2 = conv1(ks[met][2] * 2, conv1(ks[met][2], conv2(ks[met][1], l1)))
        l3 = conv1(ks[met][4] * 2, conv1(ks[met][4], conv2(ks[met][3], l2)))
        l4 = conv1(ks[met][6] * 2, conv1(ks[met][6], conv2(ks[met][5], l3)))
        l5 = conv1(ks[met][8] * 2, conv1(ks[met][8], conv2(ks[met][7], l4)))

        # --- Define expanding layers
        l6 = tran2(ks[met][7], l5)

        # --- Define expanding layers
        l7 = tran2(ks[met][5], tran1(ks[met][6], tran1(ks[met][6] * 2, concat(l4, l6))))
        l8 = tran2(ks[met][3], tran1(ks[met][4], tran1(ks[met][4] * 2, concat(l3, l7))))
        l9 = tran2(ks[met][1], tran1(ks[met][2], tran1(ks[met][2] * 2, concat(l2, l8))))
        l10 = conv1_lin(ks[met][0], conv1(ks[met][0] * 2, l9))

        # --- Create logits
        outputs = tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=1))

        loss_f = tf.keras.losses.MeanSquaredError()
        lrate = lr[met]


    # --- Create model
    model = Model(inputs=inputs, outputs=outputs)

    # --- Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
        loss=loss_f,
        experimental_run_tf_function=False
    )


    print(model.summary())
    return model






