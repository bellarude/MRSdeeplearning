from __future__ import print_function
import os
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
import tensorflow.keras.backend as K
from keras import layers
import tensorflow as tf
import numpy as np


def newModel(dim, type, subtype):
    """

    :param dim: typology of data treated by the network:
                2D: bidimensional spectroscopic information: SPECTROGRAMS
                1D: monodimensional spectroscopic information: SPECTRA
    :param type: network main-name -> overall architecture
    :param subtype: network sub-name -> specific parameter design
    :return: compiled CNN model with desired configuration

    STRUCTURE:
    dim = 2D
        type =  shallowCNN
             subtype =  ShallowRELU
                        ShallowRELU_ks3
                        ShallowELU_conv2x
                        ShallowELU_conv2x_hp
                        ShallowELU
                        ShallowELU_hp
                        ShallowELU_hp_MC
                        ShallowELU_hp_nw
                        ShallowInception
                        ShallowInception_v2
                        SrrInception_v2
                        ShallowInception_fact
                        ShallowInception_fact_v2
        type =  deepCNN
             subtype =  deepCNN_2D_ks
                        deepCNN_2D
        type =  ResNet
             subtype =  ResNet_2D_ks
                        ResNet_2D
                        ResNet_2D_deep
                        ResNet50
        type =  inceptionNet
             subtype =  Inceptionv4_rr_2D
                        SrrInception_v2
        type =  EfficientNet
             subtype =  B7
        type =  dualpath_net
             subtype =  shallowELU
    dim = 1D
        type =  ResNet
             subtype =  ResNet_fed
                        ResNet_fed_hp
        type =  deepCNN
             subtype =  DeepCNN_fed
                        DeepCNN_fed_hp
        type =  InceptionNet
             subtype =  InceptionNet-1D
    """

    externalmodel = 0
    customloss = 1
    K.set_image_data_format('channels_last')
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if dim == '2D':

        # --- Define kwargs dictionary
        kwargs = {
            'kernel_size': (3, 3),  # was 2,2 before
            'padding': 'same'}

        # --- Define poolargs dictionary
        poolargs = {
            'pool_size': (2, 2),
            'strides': (2, 2)}

        # --- Define lambda functions
        conv = lambda x, filters, strides: layers.Conv2D(filters=filters, strides=strides, **kwargs)(x)
        norm = lambda x: layers.BatchNormalization(axis=channel_axis)(x)
        normD = lambda x: layers.BatchNormalization(axis=1)(x)
        relu = lambda x: layers.ReLU()(x)
        lin = lambda x: tf.keras.activations.linear(x)
        elu = lambda x: tf.keras.activations.elu(x)
        tanh = lambda x: tf.keras.activations.tanh(x)
        maxP = lambda x, pool_size, strides: layers.MaxPooling2D(pool_size=pool_size, strides=strides)(x)
        # ConvPool = lambda filters, x : conv(x, filters, strides=2)
        add = lambda x, y: layers.Add()([x, y])
        avgP = lambda x, pool_size, strides: layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding='same')(
            x)

        conv_ks = lambda x, filters, strides, kernel_size: layers.Conv2D(filters=filters, strides=strides,
                                                                         kernel_size=kernel_size, padding='same')(x)

        flatten = lambda x: layers.Flatten()(x)
        dense = lambda units, x: layers.Dense(units=units)(x)
        maxP1D = lambda x: tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="same",
                                                        data_format="channels_last")(x)

        convBlock = lambda filters, x: relu(norm(conv(x, filters, strides=1)))
        convBlock2 = lambda filters, x: convBlock(filters, convBlock(filters, x))
        # convBlock4 = lambda filters, x : convBlock2(filters, convBlock2(filters, x))

        convBlock_ks = lambda filters, ks, x: relu(norm(conv_ks(x, filters, strides=1, kernel_size=ks)))
        convBlock_ks_lin = lambda filters, ks, x: lin(norm(conv_ks(x, filters, strides=1, kernel_size=ks)))
        convBlock_ks_elu = lambda filters, ks, x: elu(conv_ks(x, filters, strides=1, kernel_size=ks))
        convBlock_ks_elu_s = lambda filters, ks, s, x: elu(conv_ks(x, filters, strides=s, kernel_size=ks))
        convBlock_2x_ks_elu = lambda filters, ks, x: convBlock_ks_elu(filters, ks, convBlock_ks_elu(filters, ks, x))
        convBlock_ks_tanh = lambda filters, ks, x: tanh(norm(conv_ks(x, filters, strides=1, kernel_size=ks)))

        # --- Concatenate
        concat = lambda a, b: layers.Concatenate(axis=channel_axis)([a, b])

        # --- Dropout
        sDrop = lambda x, rate: layers.SpatialDropout2D(rate, data_format='channels_last')(x)

        class SpatialDropout2DMC(layers.SpatialDropout2D):
            def call(self, inputs):
                return super().call(inputs, training=True)
            
        sDropMonteCarlo = lambda x, rate: SpatialDropout2DMC(rate, data_format='channels_last')(x)
        drop = lambda x, rate: layers.Dropout(rate, noise_shape=None, seed=None)(x)

        # Federico's blocks
        # NB: residual block from Federico's differ from the original one in ResNet paper (see ResNet50)
        bottleneck = lambda filters_input, filters_output, kernel_size, x: norm(conv_ks(relu(norm(
            conv_ks(relu(norm(conv_ks(x, filters_input, strides=1, kernel_size=1))), filters_input, strides=1,
                    kernel_size=kernel_size))), filters_output, strides=1, kernel_size=1))
        ResConvBlock = lambda filters_input, filters_output, kernel_size, x: relu(
            add(bottleneck(filters_input, filters_output, kernel_size, x),
                norm(conv_ks(x, filters_output, strides=1, kernel_size=1))))
        IdeConvBlock = lambda filters_input, filters_output, kernel_size, x: relu(
            add(bottleneck(filters_input, filters_output, kernel_size, x), norm(x)))
        ResidualBlock = lambda filters_input, filters_output, kernel_size, x: maxP(
            IdeConvBlock(filters_input, filters_output, kernel_size[2], maxP(
                IdeConvBlock(filters_input, filters_output, kernel_size[1],
                             maxP(ResConvBlock(filters_input, filters_output, kernel_size[0], x), **poolargs)),
                **poolargs)), **poolargs)

        if type == 'ShallowCNN':
            img_rows, img_cols = 128, 32
            channels = 2
            # input_shape = (channels, img_rows, img_cols)
            input_shape = (img_rows, img_cols, channels)
            inputs = Input(shape=input_shape)

            if subtype == 'ShallowRELU':
                # -----------------------------------------------------------------
                # ShallowRELU
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_ks(50, (11, 11), inputs), **poolargs)
                l2 = sDrop(maxP(convBlock_ks(100, (9, 9), l1), **poolargs), 0.1)
                l3 = sDrop(maxP(convBlock_ks(150, (5, 5), l2), **poolargs), 0.15)
                outputs = lin(norm(dense(17, flatten(l3))))
                lrate = 2e-4
                print('net type: ShallowRELU')
                # -----------------------------------------------------------------

            if subtype == 'ShallowRELU_ks3':
                # -----------------------------------------------------------------
                # ShallowRELU_ks3
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_ks(50, (3,3), inputs), **poolargs)
                l2 = sDrop(maxP(convBlock_ks(100, (3,3), l1), **poolargs), 0.1)
                l3 = sDrop(maxP(convBlock_ks(150, (3,3), l2), **poolargs), 0.15)
                outputs = lin(norm(dense(17, flatten(l3))))
                lrate = 2e-4
                print('net type: ShallowRELU_ks3')
                # -----------------------------------------------------------------

                # l1 = ConvPool(convBlock_ks(50, (11,11), inputs), 25)
                # l2 = sDrop(ConvPool(convBlock_ks(100, (9,9), l1), 50), 0.1)
                # l3 = sDrop(ConvPool(convBlock_ks(150, (5,5), l2), 75), 0.15)

            if subtype == 'ShallowELU_conv2x':
                # -----------------------------------------------------------------
                # shallow ELU-CNN 2xConv
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_2x_ks_elu(50, (9,9), inputs), **poolargs)
                l2 = sDrop(maxP(convBlock_2x_ks_elu(100, (5,5), l1), **poolargs), 0.1)
                l3 = sDrop(maxP(convBlock_2x_ks_elu(150, (3,3), l2), **poolargs), 0.15)
                outputs = lin(norm(dense(17, flatten(l3))))
                lrate = 2e-4
                print('net type: ShallowELU_conv2x')
                # -----------------------------------------------------------------

            if subtype == 'ShallowELU_conv2x_hp':
                # -----------------------------------------------------------------
                # shallow ELU-CNN 2xConv
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_2x_ks_elu(60, (3,3), inputs), **poolargs)
                l2 = sDrop(maxP(convBlock_2x_ks_elu(310, (11,11), l1), **poolargs), 0.2)
                l3 = sDrop(maxP(convBlock_2x_ks_elu(280, (5,5), l2), **poolargs), 0.45)
                outputs = lin(norm(dense(17, flatten(l3))))
                lrate = 1.014e-2
                print('net type: ShallowELU_conv2x')
                # -----------------------------------------------------------------

            if subtype == 'ShallowELU':
                # -----------------------------------------------------------------
                # shallow ELU-CNN from ESMRMB abstract
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_ks_elu(50, (9, 9), inputs), **poolargs)
                l2 = sDrop(maxP(convBlock_ks_elu(100, (5, 5), l1), **poolargs), 0.1)
                l3 = sDrop(maxP(convBlock_ks_elu(150, (3, 3), l2), **poolargs), 0.15)
                outputs = lin(norm(dense(17, flatten(l3))))
                lrate = 2e-4
                print('net type: ShallowELU_ESMRMB')
                # -----------------------------------------------------------------

            if subtype == 'ShallowELU_hp':
                # -----------------------------------------------------------------
                # shallow ELU-CNN from ESMRMB abstract hp
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_ks_elu(300, (7, 7), inputs), **poolargs)
                l2 = sDrop(maxP(convBlock_ks_elu(100, (9, 9), l1), **poolargs), 0.35)
                l3 = sDrop(maxP(convBlock_ks_elu(150, (5, 5), l2), **poolargs), 0.35)
                outputs = lin(normD(dense(17, flatten(l3))))

                lrate = 5.3954e-3
                print('net type: ShallowELU_hp')
                # -----------------------------------------------------------------

            if subtype == 'ShallowELU_hp_MC':
                # -----------------------------------------------------------------
                # shallow ELU-CNN from ESMRMB abstract hp
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_ks_elu(300, (7, 7), inputs), **poolargs)
                l2 = sDropMonteCarlo(maxP(convBlock_ks_elu(100, (9, 9), l1), **poolargs), 0.35)
                l3 = sDropMonteCarlo(maxP(convBlock_ks_elu(150, (5, 5), l2), **poolargs), 0.35)
                outputs = lin(normD(dense(17, flatten(l3))))

                lrate = 5.3954e-3
                print('net type: ShallowELU_hp_MC')
                # -----------------------------------------------------------------

            if subtype == 'ShallowELU_hp_nw':
                # -----------------------------------------------------------------
                # shallow ELU-CNN from ESMRMB abstract hp
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_ks_elu(300, (7, 7), inputs), **poolargs)
                l2 = sDrop(maxP(convBlock_ks_elu(100, (9, 9), l1), **poolargs), 0.35)
                l3 = sDrop(maxP(convBlock_ks_elu(150, (5, 5), l2), **poolargs), 0.35)
                outputs = lin(normD(dense(16, flatten(l3))))

                lrate = 5.3954e-3
                print('net type: ShallowELU_hp')
                # -----------------------------------------------------------------

            if subtype == 'ShallowInception':
                # -----------------------------------------------------------------
                # ShallowInception
                # -----------------------------------------------------------------
                b01 = convBlock_ks_elu(25, (9,9), inputs)
                b11 = convBlock_ks_elu(25, (7,7), inputs)
                b21 = convBlock_ks_elu(25, (5,5), inputs)
                b31 = convBlock_ks_elu(25, (3,3), inputs)
                b1end = maxP(layers.Concatenate(axis=channel_axis)([b01, b11, b21, b31]), **poolargs)
                b02 = convBlock_ks_elu(50, (9,9), b1end)
                b12 = convBlock_ks_elu(50, (7,7), b1end)
                b22 = convBlock_ks_elu(50, (5,5), b1end)
                b32 = convBlock_ks_elu(50, (3,3), b1end)
                b2end = sDrop(maxP(layers.Concatenate(axis=channel_axis)([b02, b12, b22, b32]), **poolargs), 0.1)
                b03 = convBlock_ks_elu(75, (9,9), b2end)
                b13 = convBlock_ks_elu(75, (7,7), b2end)
                b23 = convBlock_ks_elu(75, (5,5), b2end)
                b33 = convBlock_ks_elu(75, (3,3), b2end)
                b3end = sDrop(maxP(layers.Concatenate(axis=channel_axis)([b03, b13, b23, b33]), **poolargs), 0.2)
                outputs = lin(norm(dense(17, flatten(b3end))))

                lrate = 2e-4
                print('net type: ShallowInception')
                # -----------------------------------------------------------------

            if subtype == 'ShallowInception_v2':
                # -----------------------------------------------------------------
                # ShallowInception_v2
                # -----------------------------------------------------------------
                b01 = convBlock_ks_elu(25, (9,9), inputs)
                b11 = convBlock_ks_elu(25, (7,7), inputs)
                b21 = convBlock_ks_elu(25, (5,5), inputs)
                b31 = convBlock_ks_elu(25, (3,3), inputs)
                b41 = avgP(inputs, **poolargs)
                b51 = convBlock_ks_elu(25, (1,1), inputs)
                b1end = maxP(layers.Concatenate(axis=channel_axis)([b01, b11, b21, b31, b51]), **poolargs)
                b1end = layers.Concatenate(axis=channel_axis)([b1end, b41])
                b02 = convBlock_ks_elu(50, (9,9), b1end)
                b12 = convBlock_ks_elu(50, (7,7), b1end)
                b22 = convBlock_ks_elu(50, (5,5), b1end)
                b32 = convBlock_ks_elu(50, (3,3), b1end)
                b42 = avgP(b1end, **poolargs)
                b52 = convBlock_ks_elu(50, (1,1), b1end)
                b2end = maxP(layers.Concatenate(axis=channel_axis)([b02, b12, b22, b32, b52]), **poolargs)
                b2end = sDrop(layers.Concatenate(axis=channel_axis)([b2end, b42]), 0.1)
                b03 = convBlock_ks_elu(75, (9,9), b2end)
                b13 = convBlock_ks_elu(75, (7,7), b2end)
                b23 = convBlock_ks_elu(75, (5,5), b2end)
                b33 = convBlock_ks_elu(75, (3,3), b2end)
                b43 = avgP(b2end, **poolargs)
                b53 = convBlock_ks_elu(75, (1,1), b2end)
                b63 = convBlock_ks_elu(75, (1,3), b53)
                b73 = convBlock_ks_elu(75, (3,1), b53)
                b83 = convBlock_ks_elu(75, (1,3), b33)
                b93 = convBlock_ks_elu(75, (3,1), b33)
                b3end = maxP(layers.Concatenate(axis=channel_axis)([b03, b13, b23, b33, b53, b63, b73, b83, b93]), **poolargs)
                b3end = sDrop(layers.Concatenate(axis=channel_axis)([b3end, b43]), 0.2)
                outputs = lin(norm(dense(17, flatten(b3end))))

                lrate = 2e-4
                print('net type: ShallowInception_v2')

            if subtype == 'SrrInception_v2':
                # -----------------------------------------------------------------
                # SrrInception_v2
                # -----------------------------------------------------------------
                def redBlock1(inp):
                  b0 = convBlock_ks_elu_s(50, (3,3), (2,2), inp)

                  b1 = convBlock_ks_elu(50, (1,1), inp)
                  b1 = convBlock_ks_elu(50, (3,3), b1)
                  b1 = convBlock_ks_elu_s(50, (3,3), (2,2), b1)

                  b2 = maxP(inp, **poolargs)

                  oup = layers.Concatenate(axis=channel_axis)([b0, b1, b2])
                  return oup

                def redBlock2(inp):
                  b0 = convBlock_ks_elu(50, (1,1), inp)
                  b0 = convBlock_ks_elu_s(50, (3,3), (2,2), b0)

                  b1 = convBlock_ks_elu(50, (1,1), inp)
                  b1 = convBlock_ks_elu(50, (1,7), b1)
                  b1 = convBlock_ks_elu(50, (7,1), b1)
                  b1 = convBlock_ks_elu_s(50, (3,3), (2,2), b1)

                  b2 = maxP(inp, **poolargs)

                  oup = layers.Concatenate(axis=channel_axis)([b0, b1, b2])
                  return oup

                b01 = convBlock_ks_elu(25, (9,9), inputs)
                b11 = convBlock_ks_elu(25, (7,7), inputs)
                b21 = convBlock_ks_elu(25, (5,5), inputs)
                b31 = convBlock_ks_elu(25, (3,3), inputs)
                b41 = avgP(inputs, pool_size = (2,2), strides = (1,1))
                b51 = convBlock_ks_elu(25, (1,1), inputs)
                b1end = redBlock1(layers.Concatenate(axis=channel_axis)([b01, b11, b21, b31, b41, b51]))
                b02 = convBlock_ks_elu(50, (9,9), b1end)
                b12 = convBlock_ks_elu(50, (7,7), b1end)
                b22 = convBlock_ks_elu(50, (5,5), b1end)
                b32 = convBlock_ks_elu(50, (3,3), b1end)
                b42 = avgP(b1end, pool_size = (2,2), strides = (1,1))
                b52 = convBlock_ks_elu(50, (1,1), b1end)
                b2end = sDrop(redBlock2(layers.Concatenate(axis=channel_axis)([b02, b12, b22, b32, b42, b52])), 0.1)
                b03 = convBlock_ks_elu(75, (9,9), b2end)
                b13 = convBlock_ks_elu(75, (7,7), b2end)
                b23 = convBlock_ks_elu(75, (5,5), b2end)
                b33 = convBlock_ks_elu(75, (3,3), b2end)
                b43 = avgP(b2end, pool_size = (2,2), strides = (1,1))
                b53 = convBlock_ks_elu(75, (1,1), b2end)
                b63 = convBlock_ks_elu(75, (1,3), b53)
                b73 = convBlock_ks_elu(75, (3,1), b53)
                b83 = convBlock_ks_elu(75, (1,3), b33)
                b93 = convBlock_ks_elu(75, (3,1), b33)
                b3end = sDrop(avgP(layers.Concatenate(axis=channel_axis)([b03, b13, b23, b33, b43, b53, b63, b73, b83, b93]), **poolargs), 0.2)
                last = drop(flatten(b3end), 0.5)
                outputs = lin(norm(dense(17, last)))

                lrate = 2e-4
                print('net type: SrrInception_v2')
            # -----------------------------------------------------------------

            if subtype == 'ShallowInception_fact':
                # -----------------------------------------------------------------
                # ShallowInception_fact
                # -----------------------------------------------------------------
                b01 = convBlock_ks_elu(25, (9,1), inputs)
                b01 = convBlock_ks_elu(25, (1,9), b01)
                b11 = convBlock_ks_elu(25, (7,1), inputs)
                b11 = convBlock_ks_elu(25, (1,7), b11)
                b21 = convBlock_ks_elu(25, (3,3), convBlock_ks_elu(25, (3,3), inputs))
                b31 = convBlock_ks_elu(25, (3,3), inputs)
                b1end = maxP(layers.Concatenate(axis=channel_axis)([b01, b11, b21, b31]), **poolargs)
                b02 = convBlock_ks_elu(50, (9,1), b1end)
                b02 = convBlock_ks_elu(50, (1,9), b02)
                b12 = convBlock_ks_elu(50, (7,1), b1end)
                b12 = convBlock_ks_elu(50, (1,7), b12)
                b22 = convBlock_ks_elu(50, (3,3), convBlock_ks_elu(50, (3,3), b1end))
                b32 = convBlock_ks_elu(50, (3,3), b1end)
                b2end = sDrop(maxP(layers.Concatenate(axis=channel_axis)([b02, b12, b22, b32]), **poolargs), 0.1)
                b03 = convBlock_ks_elu(75, (9,1), b2end)
                b03 = convBlock_ks_elu(75, (1,9), b03)
                b13 = convBlock_ks_elu(75, (7,1), b2end)
                b13 = convBlock_ks_elu(75, (1,7), b13)
                b23 = convBlock_ks_elu(75, (3,3), convBlock_ks_elu(75, (3,3), b2end))
                b33 = convBlock_ks_elu(75, (3,3), b2end)
                b3end = sDrop(maxP(layers.Concatenate(axis=channel_axis)([b03, b13, b23, b33]), **poolargs), 0.2)
                outputs = lin(norm(dense(17, flatten(b3end))))

                lrate = 2e-4
                print('net type: ShallowInception_fact')
                # -----------------------------------------------------------------

            if subtype == 'ShallowInception_fact_v2':
                # -----------------------------------------------------------------
                # ShallowInception_fact_v2
                # -----------------------------------------------------------------
                b01 = convBlock_ks_elu(25, (9,1), inputs)
                b01 = convBlock_ks_elu(25, (1,9), b01)
                b11 = convBlock_ks_elu(25, (7,1), inputs)
                b11 = convBlock_ks_elu(25, (1,7), b11)
                b21 = convBlock_ks_elu(25, (3,3), convBlock_ks_elu(25, (3,3), inputs))
                b31 = convBlock_ks_elu(25, (3,3), inputs)
                b41 = avgP(inputs, **poolargs)
                b51 = convBlock_ks_elu(25, (1,1), inputs)
                b1end = maxP(layers.Concatenate(axis=channel_axis)([b01, b11, b21, b31, b51]), **poolargs)
                b1end = layers.Concatenate(axis=channel_axis)([b1end, b41])
                b02 = convBlock_ks_elu(50, (9,1), b1end)
                b02 = convBlock_ks_elu(50, (1,9), b02)
                b12 = convBlock_ks_elu(50, (7,1), b1end)
                b12 = convBlock_ks_elu(50, (1,7), b12)
                b22 = convBlock_ks_elu(50, (3,3), convBlock_ks_elu(50, (3,3), b1end))
                b32 = convBlock_ks_elu(50, (3,3), b1end)
                b42 = avgP(b1end, **poolargs)
                b52 = convBlock_ks_elu(50, (1,1), b1end)
                b2end = maxP(layers.Concatenate(axis=channel_axis)([b02, b12, b22, b32, b52]), **poolargs)
                b2end = sDrop(layers.Concatenate(axis=channel_axis)([b2end, b42]), 0.1)
                b03 = convBlock_ks_elu(75, (9,1), b2end)
                b03 = convBlock_ks_elu(75, (1,9), b03)
                b13 = convBlock_ks_elu(75, (7,1), b2end)
                b13 = convBlock_ks_elu(75, (1,7), b13)
                b23 = convBlock_ks_elu(75, (3,3), convBlock_ks_elu(75, (3,3), b2end))
                b33 = convBlock_ks_elu(75, (3,3), b2end)
                b43 = avgP(b2end, **poolargs)
                b53 = convBlock_ks_elu(75, (1,1), b2end)
                b63 = convBlock_ks_elu(75, (1,3), b53)
                b73 = convBlock_ks_elu(75, (3,1), b53)
                b83 = convBlock_ks_elu(75, (1,3), b33)
                b93 = convBlock_ks_elu(75, (3,1), b33)
                b3end = maxP(layers.Concatenate(axis=channel_axis)([b03, b13, b23, b33, b53, b63, b73, b83, b93]), **poolargs)
                b3end = sDrop(layers.Concatenate(axis=channel_axis)([b3end, b43]), 0.2)
                outputs = lin(norm(dense(17, flatten(b3end))))

                lrate = 2e-4
                print('net type: ShallowInception_fact_v2')
                # -----------------------------------------------------------------

        if type == 'deepCNN':
            img_rows, img_cols = 128, 64
            channels = 1
            input_shape = (img_rows, img_cols, channels)
            inputs = Input(shape=input_shape)

            if subtype == 'deepCNN_2D_ks':
                # -----------------------------------------------------------------
                # deepCNN_2D_ks
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_ks(8, (13, 13), inputs), **poolargs)
                l2 = maxP(convBlock_ks(16, (11, 11), l1), **poolargs)
                l3 = maxP(convBlock_ks(32, (9, 9), l2), **poolargs)
                l4 = maxP(convBlock_ks(64, (7, 7), l3), **poolargs)
                l5 = maxP(convBlock_ks(128, (5, 5), l4), **poolargs)
                l6 = maxP(convBlock_ks(256, (3, 3), l5), **poolargs)
                l7 = drop(relu(norm(dense(400, flatten(l6)))), 0.1)
                outputs = lin(dense(17, l7))

                lrate = 2e-4
                print('net type: deepCNN_2D_ks')
                # -----------------------------------------------------------------

            if subtype == 'deepCNN_2D':
                # -----------------------------------------------------------------
                # deepCNN_2D
                # -----------------------------------------------------------------
                l1 = maxP(convBlock_ks(8, (3,3), inputs), **poolargs)
                l2 = maxP(convBlock_ks(16, (3,3), l1), **poolargs)
                l3 = maxP(convBlock_ks(32, (3,3), l2), **poolargs)
                l4 = maxP(convBlock_ks(64, (3,3), l3), **poolargs)
                l5 = maxP(convBlock_ks(128, (3,3), l4), **poolargs)
                l6 = maxP(convBlock_ks(256, (3,3), l5), **poolargs)
                l7 = drop(relu(norm(dense(400, flatten(l6)))), 0.1)
                outputs = lin(dense(17,l7))

                lrate = 2e-4
                print('net type: deepCNN_2D')
                # -----------------------------------------------------------------

        if type == 'ResNet':
            img_rows, img_cols = 128, 64
            channels = 1
            input_shape = (img_rows, img_cols, channels)
            inputs = Input(shape=input_shape)

            # general comment: we started from Federico's design but due to the input size we
            # cannot use 3 Residual block (too many MaxP and dimension reduction) and the
            # 2 conv layers at the beginning must be skipped to keep 2 residual blocks.
            # on ResNet_2D_deep a design with these 2 layers is kept, but maxP is skipped
            # on them.

            # NB2: Residual Blocks from Federico's design differ from Residual Blocks on the
            # ResNet original paper.

            if subtype == 'ResNet_2D_ks':
                # -----------------------------------------------------------------
                # ResNet_2D_ks
                # -----------------------------------------------------------------
                l1 = ResidualBlock(32, 128, [13,11,9], inputs)
                l2 = ResidualBlock(64, 256, [7,5,3],l1)
                l3 = drop(relu(norm(dense(400, flatten(l2)))),0.1)
                outputs = lin(dense(17, l3))

                lrate = 2e-4
                print('net type: ResNet_2D_ks')
                # -----------------------------------------------------------------

            if subtype == 'ResNet_2D':
                # ResNet_2D
                # -----------------------------------------------------------------
                l1 = ResidualBlock(32, 128, [3,3,3], inputs)
                l2 = ResidualBlock(64, 256, [3,3,3],l1)
                l3 = drop(relu(norm(dense(400, flatten(l2)))),0.1)
                outputs = lin(dense(17, l3))

                lrate = 2e-4
                print('net type: ResNet_2D')
                # -----------------------------------------------------------------

            if subtype == 'ResNet_2D_deep':
                # -----------------------------------------------------------------
                # ResNet_2D_deep
                # -----------------------------------------------------------------
                l1 = convBlock(8, inputs)
                l2 = convBlock(16, l1)
                l3 = ResidualBlock(32, 128, [13,11,9], l2)
                l4 = ResidualBlock(64, 256, [7,5,3],l3)
                l6 = drop(relu(norm(dense(400, flatten(l4)))),0.1)
                outputs = lin(dense(17, l6))

                lrate = 2e-4
                print('net type: ResNet_2D_deep')
                # -----------------------------------------------------------------

            if subtype == 'ResNet50':
                from resnet50model import ResNet50
                externalmodel = 1
                model = ResNet50(input_shape = input_shape, classes=17)
                lrate = 2e-4

        if type == 'inceptionNet':

            #Inceptionv4 see colab

            if subtype == 'Inceptionv4_rr_2D':
                img_rows, img_cols = 128, 64
                channels = 1
                input_shape = (img_rows, img_cols, channels)
                inputs = Input(shape=input_shape)

                # -----------------------------------------------------------------
                # Inceptionv4_rr_2D
                # -----------------------------------------------------------------
                from InceptionNetV4model import block_inception_a, block_inception_b, block_inception_c, block_reduction_a, block_reduction_b

                l0 = maxP(convBlock_ks(50, (9, 9), inputs), **poolargs)
                l1 = block_inception_a(l0)
                l2 = block_reduction_a(l1)
                l3 = block_inception_b(l2)
                l4 = block_reduction_b(l3)
                l5 = avgP(block_inception_c(l4), **poolargs)
                l6 = drop(flatten(l5), 0.5)
                outputs = lin(norm(dense(17, l6)))

                lrate = 2e-4
                print('net type: Inception_v4_rr_2D')
                # -----------------------------------------------------------------

            if subtype == 'SrrInception_v2':
            # -----------------------------------------------------------------
            # SrrInception_v2
            # -----------------------------------------------------------------

                def redBlock1(inp):
                  b0 = convBlock_ks_elu_s(50, (3,3), (2,2), inp)

                  b1 = convBlock_ks_elu(50, (1,1), inp)
                  b1 = convBlock_ks_elu(50, (3,3), b1)
                  b1 = convBlock_ks_elu_s(50, (3,3), (2,2), b1)

                  b2 = maxP(inp, **poolargs)

                  oup = layers.Concatenate(axis=channel_axis)([b0, b1, b2])
                  return oup

                def redBlock2(inp):
                  b0 = convBlock_ks_elu(50, (1,1), inp)
                  b0 = convBlock_ks_elu_s(50, (3,3), (2,2), b0)

                  b1 = convBlock_ks_elu(50, (1,1), inp)
                  b1 = convBlock_ks_elu(50, (1,7), b1)
                  b1 = convBlock_ks_elu(50, (7,1), b1)
                  b1 = convBlock_ks_elu_s(50, (3,3), (2,2), b1)

                  b2 = maxP(inp, **poolargs)

                  oup = layers.Concatenate(axis=channel_axis)([b0, b1, b2])
                  return oup

                b01 = convBlock_ks_elu(25, (9,9), inputs)
                b11 = convBlock_ks_elu(25, (7,7), inputs)
                b21 = convBlock_ks_elu(25, (5,5), inputs)
                b31 = convBlock_ks_elu(25, (3,3), inputs)
                b41 = avgP(inputs, pool_size = (2,2), strides = (1,1))
                b51 = convBlock_ks_elu(25, (1,1), inputs)
                b1end = redBlock1(layers.Concatenate(axis=channel_axis)([b01, b11, b21, b31, b41, b51]))
                b02 = convBlock_ks_elu(50, (9,9), b1end)
                b12 = convBlock_ks_elu(50, (7,7), b1end)
                b22 = convBlock_ks_elu(50, (5,5), b1end)
                b32 = convBlock_ks_elu(50, (3,3), b1end)
                b42 = avgP(b1end, pool_size = (2,2), strides = (1,1))
                b52 = convBlock_ks_elu(50, (1,1), b1end)
                b2end = sDrop(redBlock2(layers.Concatenate(axis=channel_axis)([b02, b12, b22, b32, b42, b52])), 0.1)
                b03 = convBlock_ks_elu(75, (9,9), b2end)
                b13 = convBlock_ks_elu(75, (7,7), b2end)
                b23 = convBlock_ks_elu(75, (5,5), b2end)
                b33 = convBlock_ks_elu(75, (3,3), b2end)
                b43 = avgP(b2end, pool_size = (2,2), strides = (1,1))
                b53 = convBlock_ks_elu(75, (1,1), b2end)
                b63 = convBlock_ks_elu(75, (1,3), b53)
                b73 = convBlock_ks_elu(75, (3,1), b53)
                b83 = convBlock_ks_elu(75, (1,3), b33)
                b93 = convBlock_ks_elu(75, (3,1), b33)
                b3end = sDrop(avgP(layers.Concatenate(axis=channel_axis)([b03, b13, b23, b33, b43, b53, b63, b73, b83, b93]), **poolargs), 0.2)
                last = drop(flatten(b3end), 0.5)
                outputs = lin(norm(dense(17, last)))

                lrate = 2e-4
                print('net type: SrrInception_v2')

        if type == 'EfficientNet':
            if subtype =='B7':

                IMG_SHAPE = (128, 32, 2)
                model0 = tf.keras.applications.EfficientNetB7(input_shape=IMG_SHAPE, include_top=False, weights=None)
                inputs = model0.input
                outputs = lin(norm(dense(17, flatten(model0.output))))

                lrate = 2e-4

                print('net type: EfficientNetB7')

        if type == 'dualpath_net':
            img_rows, img_cols = 128, 32
            channels = 2
            # input_shape = (channels, img_rows, img_cols)
            input_shape = (img_rows, img_cols, channels)
            inputs = Input(shape=input_shape)

            if subtype == 'ShallowELU':

                # architecture from ESMRMB 2021 abstract
                p = 17  # y_train.shape[1] -> output concentration number of metabolites + water ref.

                l1 = maxP(convBlock_ks_elu(50, (9, 9), inputs), **poolargs)
                l2 = sDrop(maxP(convBlock_ks_elu(100, (5, 5), l1), **poolargs), 0.1)
                l3 = sDrop(maxP(convBlock_ks_elu(150, (3, 3), l2), **poolargs), 0.15)
                mu = lin(norm(dense(p, flatten(l3))))

                l12 = maxP(convBlock_ks_elu(50, (9, 9), inputs), **poolargs)
                l22 = sDrop(maxP(convBlock_ks_elu(100, (5, 5), l12), **poolargs), 0.1)
                l32 = sDrop(maxP(convBlock_ks_elu(150, (3, 3), l22), **poolargs), 0.15)
                sigma = lin(norm(dense(p, flatten(l32))))
                #sigma == T1, extended here in case we assume nott indipendent variables and covariance matrix is not diagonal
                # T2 = lin(norm(dense((p*(p-1)/2), flatten(l32))))
                # T = concat(T1,T2)

                model_mu = Model(inputs=inputs, outputs=mu)
                model_cov = Model(inputs=inputs, outputs=sigma)
                outputs = concat(model_mu.output, model_cov.output)
                # model = Model(inputs=inputs, outputs=out)
                lrate = 2e-3 #donot start at 2e-4 here

                def wmse_loss():
                    def loss(ytrue, ypreds):
                        n_dims = int(int(ypreds.shape[1]) / 2)
                        mu = ypreds[:, 0:n_dims]
                        logsigma = ypreds[:, n_dims:]

                        loss1 = tf.reduce_mean(tf.exp(-logsigma) * tf.square((mu - ytrue)))
                        loss2 = tf.reduce_mean(logsigma)
                        loss = .5 * (loss1 + loss2)

                        return loss

                    return loss

                def gauss_loss():
                    def loss(ytrue, ypreds):
                        """
                            NB: exploration of sigma as measure of STD is not convincing. Whereas results are comparable with wmse_loss

                            Keras implmementation of multivariate Gaussian negative loglikelihood loss function.
                            This implementation implies diagonal covariance matrix.

                            Parameters
                            ----------
                            ytrue: tf.tensor of shape [n_samples, n_dims]
                                ground truth values
                            ypreds: tf.tensor of shape [n_samples, n_dims*2]
                                predicted mu and logsigma values (e.g. by your neural network)

                            Returns
                            -------
                            neg_log_likelihood: float
                                negative loglikelihood averaged over samples

                            This loss can then be used as a target loss for any keras model, e.g.:
                                model.compile(loss=gaussian_nll, optimizer='Adam')

                            """

                        n_dims = int(int(ypreds.shape[1]) / 2)
                        mu = ypreds[:, 0:n_dims]
                        logsigma = ypreds[:, n_dims:]

                        mse = -0.5 * K.sum(K.square((ytrue - mu) / K.exp(logsigma)), axis=1)
                        sigma_trace = -K.sum(logsigma, axis=1)
                        log2pi = -0.5 * n_dims * np.log(2 * np.pi) #constant term that can be removed

                        log_likelihood = mse + sigma_trace + log2pi

                        return K.mean(-log_likelihood)

                    return loss

                model_loss = wmse_loss()

    if dim == '1D':

        # --- Define kwargs dictionary
        kwargs = {
            'kernel_size': (3),
            'padding': 'same'}

        # --- Define poolargs dictionary
        poolargs = {
            'pool_size': (2),
            'strides': (2)}

        # -----------------------------------------------------------------------------
        # Define lambda functions
        # -----------------------------------------------------------------------------
        os.environ["KERAS_BACKEND"] = "theano"
        K.set_image_data_format('channels_last')
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        # channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3

        conv = lambda filters, x, strides: layers.Conv1D(filters=filters, strides=strides, **kwargs)(x)
        conv_ks = lambda filters, x, strides, kernel_size: layers.Conv1D(filters=filters, strides=strides,
                                                                         kernel_size=kernel_size, padding='same')(x)

        norm = lambda x: layers.BatchNormalization(axis=channel_axis)(x)
        normD = lambda x: layers.BatchNormalization(axis=1)(x)
        relu = lambda x: layers.ReLU()(x)
        lin = lambda x: tf.keras.activations.linear(x)
        tanh = lambda x: tf.keras.activations.tanh(x)
        elu = lambda x: tf.keras.layers.ELU(alpha=1.0)(x)
        maxP = lambda x, pool_size, strides: layers.MaxPooling1D(pool_size=pool_size, strides=strides)(x)
        ConvPool = lambda x, filters: conv(filters, x, strides=2)
        avgP = lambda x, pool_size, strides: layers.AveragePooling1D(pool_size=pool_size, strides=strides,
                                                                     padding='same')(x)

        flatten = lambda x: layers.Flatten()(x)
        dense = lambda units, x: layers.Dense(units=units)(x)

        convBlock_ks = lambda filters, ks, x: relu(norm(conv_ks(filters, x, strides=1, kernel_size=ks)))
        convBlock_ks_s = lambda filters, ks, s, x: relu(norm(conv_ks(filters, x, strides=s, kernel_size=ks)))
        convBlock_ks_lin = lambda filters, ks, x: lin(norm(conv_ks(filters, x, strides=1, kernel_size=ks)))
        convBlock_ks_tanh = lambda filters, ks, x: tanh(norm(conv_ks(filters, x, strides=1, kernel_size=ks)))

        convBlock = lambda filters, x: relu(norm(conv(filters, x, strides=1)))
        convBlockELU = lambda filters, x: elu(conv(filters, x, strides=1))
        convBlock2 = lambda filters, x: convBlock(filter, convBlock(filters, x))
        convBlock3 = lambda filters, x: convBlock(filters, convBlock2(filters, x))
        convBlock4 = lambda filters, x: convBlock2(filters, convBlock2(filters, x))

        convShortcut = lambda filters, x: norm(conv(filters, x, strides=1))

        concat = lambda a, b: layers.Concatenate(axis=2)([a, b])

        def concatntimes(x, n):
            output = concat(x, x)
            for i in range(n - 1):
                output = concat(output, output)
            return output

        add = lambda x, y: layers.Add()([x, y])

        # dropout = lambda x, percentage, size : layers.Dropout(percentage, size)(x)
        drop = lambda x, rate: layers.Dropout(rate, noise_shape=None, seed=None)(x)

        # Federico's blocks
        bottleneck = lambda filters_input, filters_output, x: norm(conv_ks(filters_output, relu(norm(
            conv_ks(filters_input, relu(norm(conv_ks(filters_input, x, strides=1, kernel_size=1))), strides=1,
                    kernel_size=3))), strides=1, kernel_size=1))
        ResConvBlock = lambda filters_input, filters_output, x: relu(add(bottleneck(filters_input, filters_output, x),
                                                                         norm(conv_ks(filters_output, x, strides=1,
                                                                                      kernel_size=1))))
        IdeConvBlock = lambda filters_input, filters_output, x: relu(
            add(bottleneck(filters_input, filters_output, x), norm(x)))
        ResidualBlock = lambda filters_input, filters_output, x: maxP(IdeConvBlock(filters_input, filters_output, maxP(
            IdeConvBlock(filters_input, filters_output,
                         maxP(ResConvBlock(filters_input, filters_output, x), **poolargs)), **poolargs)), **poolargs)

        # ResNet paper's block
        conv2id = lambda filters, x: relu(add(norm(conv(filters, convBlock(filters, x), strides=1)), norm(x)))
        conv2conv = lambda filters, x: relu(
            add(norm(conv(filters, convBlock(filters, x), strides=2)), norm(conv(filters, x, strides=1))))
        DRB1 = lambda filters, x: conv2id(filters, conv2id(filters, conv2id(filters, x)))


        if type == 'ResNet':
            datapoints = 2048
            channels = 1

            input_shape = (datapoints, channels)
            inputs = Input(shape=input_shape)

            if subtype == 'ResNet_fed':
                # -----------------------------------------------------------------
                # ResNet_fed
                # -----------------------------------------------------------------
                l1 = maxP(convBlock(8, inputs), **poolargs)
                l2 = maxP(convBlock(16, l1), **poolargs)
                l3 = ResidualBlock(32, 128, l2)
                l4 = ResidualBlock(64, 256, l3)
                l5 = ResidualBlock(128, 512, l4)
                l6 = drop(relu(normD(dense(400, flatten(l5)))), 0.3)
                outputs = lin(dense(17, l6))

                lrate = 2e-4
                # -----------------------------------------------------------------

            if subtype == 'ResNet_fed_hp':
                # -----------------------------------------------------------------
                # ResNet_fed_hp
                # -----------------------------------------------------------------
                l1 = maxP(convBlock(15, inputs), **poolargs)
                l2 = maxP(convBlock(40, l1), **poolargs)
                l3 = ResidualBlock(90, 150, l2)
                l4 = ResidualBlock(150, 230, l3)
                l5 = ResidualBlock(140, 600, l4)
                l6 = drop(relu(normD(dense(110, flatten(l5)))), 0.25)
                outputs = lin(dense(17, l6))

                lrate = 3.1938e-4
                # -----------------------------------------------------------------

        if type == 'DeepCNN':
            datapoints = 2048
            channels = 1

            input_shape = (datapoints, channels)
            inputs = Input(shape=input_shape)

            if subtype == 'DeepCNN_fed':
                # -----------------------------------------------------------------
                # DeepCNN_fed
                # -----------------------------------------------------------------
                l1 = maxP(convBlock(8, inputs), **poolargs)
                l2 = maxP(convBlock(16, l1), **poolargs)
                l3 = maxP(convBlock(32, l2), **poolargs)
                l4 = maxP(convBlock(64, l3), **poolargs)
                l5 = maxP(convBlock(128, l4), **poolargs)
                l6 = maxP(convBlock(256, l5), **poolargs)
                l7 = drop(relu(normD(dense(400, flatten(l6)))), 0.1)
                outputs = lin(dense(17, l7))

                lrate = 2e-4
                # -----------------------------------------------------------------

            if subtype == 'DeepCNN_fed_hp':
                # -----------------------------------------------------------------
                # DeepCNN_fed_hp
                # -----------------------------------------------------------------
                l1 = maxP(convBlock(15, inputs), **poolargs)
                l2 = maxP(convBlock(40, l1), **poolargs)
                l3 = maxP(convBlock(90, l2), **poolargs)
                l4 = maxP(convBlock(150, l3), **poolargs)
                l5 = maxP(convBlock(160, l4), **poolargs)
                l6 = maxP(convBlock(280, l5), **poolargs)
                l7 = drop(relu(normD(dense(780, flatten(l6)))), 0.3)
                outputs = lin(dense(17, l7))

                lrate = 0.003275
                # -----------------------------------------------------------------
        if type == 'InceptionNet':
            datapoints = 2048
            channels = 1

            input_shape = (datapoints, channels)
            inputs = Input(shape=input_shape)
            if subtype == 'InceptionNet-1D':
                # -----------------------------------------------------------------
                # InceptionNet-1D
                # -----------------------------------------------------------------
                def redBlock1(inp):
                    b0 = convBlock_ks_s(50, 3, 2, inp)

                    b1 = convBlock_ks(50, 1, inp)
                    b1 = convBlock_ks(50, 3, b1)
                    b1 = convBlock_ks_s(50, 3, 2, b1)

                    b2 = maxP(inp, **poolargs)

                    oup = layers.Concatenate(axis=channel_axis)([b0, b1, b2])
                    return oup

                def redBlock2(inp):
                    b0 = convBlock_ks(50, 1, inp)
                    b0 = convBlock_ks_s(50, 3, 2, b0)

                    b1 = convBlock_ks(50, 1, inp)
                    b1 = convBlock_ks(50, 7, b1)
                    b1 = convBlock_ks_s(50, 3, 2, b1)

                    b2 = maxP(inp, **poolargs)

                    oup = layers.Concatenate(axis=channel_axis)([b0, b1, b2])
                    return oup

                b01 = convBlock_ks(96, 9, inputs)
                b11 = convBlock_ks(96, 7, inputs)
                b21 = convBlock_ks(96, 5, inputs)
                b31 = convBlock_ks(96, 3, inputs)
                b41 = avgP(inputs, pool_size=2, strides=1)
                b51 = convBlock_ks(96, 1, inputs)
                b1end = redBlock1(layers.Concatenate(axis=channel_axis)([b01, b11, b21, b31, b41, b51]))
                b02 = convBlock_ks(128, 9, b1end)
                b12 = convBlock_ks(128, 7, b1end)
                b22 = convBlock_ks(128, 5, b1end)
                b32 = convBlock_ks(128, 3, b1end)
                b42 = avgP(b1end, pool_size=2, strides=1)
                b52 = convBlock_ks(128, 1, b1end)
                b2end = redBlock2(layers.Concatenate(axis=channel_axis)([b02, b12, b22, b32, b42, b52]))
                b03 = convBlock_ks(256, 9, b2end)
                b13 = convBlock_ks(256, 7, b2end)
                b23 = convBlock_ks(256, 5, b2end)
                b33 = convBlock_ks(256, 3, b2end)
                b43 = avgP(b2end, pool_size=2, strides=1)
                b53 = convBlock_ks(256, 1, b2end)
                b63 = convBlock_ks(256, 3, b53)
                b73 = convBlock_ks(256, 3, b33)
                b3end = avgP(layers.Concatenate(axis=channel_axis)([b03, b13, b23, b33, b43, b53, b63, b73]),
                             **poolargs)
                last = drop(flatten(b3end), 0.8)
                outputs = lin(normD(dense(17, last)))

                lrate = 2e-4
                # -----------------------------------------------------------------
    # else:
    #     print('Model dimensionality of data is wrong')

    # --- Create model
    if externalmodel == 0:
        model = Model(inputs=inputs, outputs=outputs)

    # --- Compile model
    if customloss == 0:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
            loss=tf.keras.losses.MeanSquaredError(),
            experimental_run_tf_function=False)
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
            loss=model_loss,
            experimental_run_tf_function=False)


    print(model.summary())
    return model