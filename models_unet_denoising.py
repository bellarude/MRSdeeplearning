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

def newModel(type=0):
    """
    :param type: integer, 0: traditional model, 1: variational model
    :return: model function aiming to predict metabolite basis set from real-1-channel-spectra passed as input
    """

    K.set_image_data_format('channels_last')
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 2

    datapoints = 1024
    channels = 2  # number of channels
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

    conv1_elu = lambda filters, x: elu(norm(conv_s(x, filters=filters, strides=1)))
    conv2_elu = lambda filters, x: elu(norm(conv_s(x, filters=filters, strides=2)))
    # --- Define single transpose
    tran = lambda x, filters, strides: layers.Conv1DTranspose(filters=filters, strides=strides, kernel_size=3,
                                                              padding='same')(x)
    # --- Define transpose block
    tran1 = lambda filters, x: relu(norm(tran(x, filters, strides=1)))
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=2)))
    tran1_elu = lambda filters, x: elu(norm(tran(x, filters, strides=1)))
    tran2_elu = lambda filters, x: elu(norm(tran(x, filters, strides=2)))

    norm = lambda x: layers.BatchNormalization(axis=channel_axis)(x)
    relu = lambda x: layers.ReLU()(x)
    elu = lambda x: layers.ELU()(x)
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

    pad = 0

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
        l10 = conv1_lin(2, conv1(32, l9)) # RR:03.012021 recall that old model with same==1 was tuned without linear activation here: conv1(32, conv1(32, l9))

        # --- Create logits
        outputs = l10 # tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=2))
        loss_f = tf.keras.losses.MeanSquaredError()
        lrate = 2e-4 #1e-3

    else:
        # -----------------------------------------------------------------------------
        # when you want another model
        # -----------------------------------------------------------------------------


        # -----------------------------------------------------------------------------

        #RR 220201 nonlinearity trial
        conv1 = conv1_elu
        conv2 = conv2_elu
        tran1 = tran1_elu
        tran2 = tran2_elu

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
        l10 = conv1(32, conv1(32, l9)) #RR 280122 from lin to relu

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

        # model_mu.load_weights("C:/Users/Rudy/Desktop/DL_models/net_type/unet/UNet_tCho_NOwat.best.hdf5")
        outputs = concat(model_mu.output, model_cov.output)

        lrate = 1e-3

        # def wmse_loss_unet():
        #     def loss(ytrue, ypreds):
        #         # n_dims = int(int(ypreds.shape[1]) / 2)
        #         mu = ypreds[:, :, 0]  # 1st channel is mu
        #         logsigma = ypreds[:, :, 1]  # 2nd channel is sigma
        #
        #         loss1 = tf.reduce_mean(tf.exp(-logsigma) * tf.square((mu - ytrue)))
        #         loss2 = tf.reduce_mean(logsigma)
        #         loss = .5 * (loss1 + loss2)
        #
        #         return loss
        #
        #     return loss


        loss_f = wmse_loss_unet()



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


def newModel_deep(type=0):
    """
    :param type: integer, 0: traditional model, 1: variational model
    :return: model function aiming to predict metabolite basis set from real-1-channel-spectra passed as input
    """

    K.set_image_data_format('channels_last')
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 2

    datapoints = 1024
    channels = 2  # number of channels
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

    conv1_elu = lambda filters, x: elu(norm(conv_s(x, filters=filters, strides=1)))
    conv2_elu = lambda filters, x: elu(norm(conv_s(x, filters=filters, strides=2)))
    # --- Define single transpose
    tran = lambda x, filters, strides: layers.Conv1DTranspose(filters=filters, strides=strides, kernel_size=3,
                                                              padding='same')(x)
    # --- Define transpose block
    tran1 = lambda filters, x: relu(norm(tran(x, filters, strides=1)))
    tran2 = lambda filters, x: relu(norm(tran(x, filters, strides=2)))
    tran1_elu = lambda filters, x: elu(norm(tran(x, filters, strides=1)))
    tran2_elu = lambda filters, x: elu(norm(tran(x, filters, strides=2)))

    norm = lambda x: layers.BatchNormalization(axis=channel_axis)(x)
    relu = lambda x: layers.ReLU()(x)
    elu = lambda x: layers.ELU()(x)
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

    pad = 0

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

        l5_0 = conv1(64, conv1(1024, conv2(128, l5)))

        # --- Define expanding layers
        l6 = tran2(64, l5_0)

        # --- Define expanding layers
        l7_0 = tran2(256, tran1(80, tran1(80, concat(l5, l6))))
        l7 = tran2(128, tran1(64, tran1(64, concat(l5, l7_0))))
        l8 = tran2(64, tran1(48, tran1(48, concat(l3, l7))))
        l9 = tran2(32, tran1(32, tran1(32, concat(l2, l8))))
        l10 = conv1_lin(2, conv1(32, l9)) # RR:03.012021 recall that old model with same==1 was tuned without linear activation here: conv1(32, conv1(32, l9))

        # --- Create logits
        outputs = l10 # tf.keras.layers.Cropping1D(cropping=(pad, pad))(conv(l10, kernel_size=3, filters=2))
        loss_f = tf.keras.losses.MeanSquaredError()
        lrate = 1e-4 #1e-3

    else:
        # -----------------------------------------------------------------------------
        # when you want another model
        # -----------------------------------------------------------------------------


        # -----------------------------------------------------------------------------

        #RR 220201 nonlinearity trial
        conv1 = conv1_elu
        conv2 = conv2_elu
        tran1 = tran1_elu
        tran2 = tran2_elu

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
        l10 = conv1(32, conv1(32, l9)) #RR 280122 from lin to relu

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

        # model_mu.load_weights("C:/Users/Rudy/Desktop/DL_models/net_type/unet/UNet_tCho_NOwat.best.hdf5")
        outputs = concat(model_mu.output, model_cov.output)

        lrate = 1e-4

        # def wmse_loss_unet():
        #     def loss(ytrue, ypreds):
        #         # n_dims = int(int(ypreds.shape[1]) / 2)
        #         mu = ypreds[:, :, 0]  # 1st channel is mu
        #         logsigma = ypreds[:, :, 1]  # 2nd channel is sigma
        #
        #         loss1 = tf.reduce_mean(tf.exp(-logsigma) * tf.square((mu - ytrue)))
        #         loss2 = tf.reduce_mean(logsigma)
        #         loss = .5 * (loss1 + loss2)
        #
        #         return loss
        #
        #     return loss


        loss_f = wmse_loss_unet()



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





