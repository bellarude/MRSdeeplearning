from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
from keras.models import Model, load_model, Sequential
from keras.layers import Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Reshape, Dense, Flatten, Input, BatchNormalization, ELU, Conv2D, Conv1D, Dropout, SpatialDropout2D, Concatenate
from keras import backend as K
from data_load_norm import dataimport2D, labelsimport, labelsNorm, ilabelsNorm, inputConcat2D, dataimport2Dhres, labelsNorm, ilabelsNorm, inputConcat2D, labelsNormREDdataset
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import xlsxwriter

folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/'
dataname = 'dataset_spgram.mat'
labelsname = 'labels_c.mat'

doTrain = 1
doTest = 1
hres = 0
cifar_example = 0
# ----------------------------------------------------------------
# TRANSFORMER PARAMETERS
# ----------------------------------------------------------------
num_classes = 17
img_rows, img_cols = 128, 32
channels = 2
input_shape = (img_rows, img_cols, channels)
# input_shape = (32, 32, 3)

learning_rate = 2e-4 #0.001 #0.001
weight_decay = 0.0001
batch_size = 30
num_epochs = 200
image_size = [128,32]  # We'll resize input images to this size
patch_hight = 8
patch_width = 8
patch_size = [patch_hight,patch_width]  # Size of the patches to be extract from the input images
num_patches = 64 #image_size[0]/patch_size[0] * image_size[1]/patch_size[1]
projection_dim = 256 #256
num_heads = 64 #4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8 #8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
# ----------------------------------------------------------------

if hres:
    X_train, X_val = dataimport2Dhres(folder, dataname, 'dataset')
    y_train, y_val = labelsimporthres(folder, labelsname, 'labels_c')
else:
    X_train, X_val = dataimport2D(folder, dataname, 'dataset')
    y_train, y_val = labelsimport(folder, labelsname, 'labels_c')

ny_train, w_y_train = labelsNorm(y_train)
ny_val, w_y_val = labelsNorm(y_val)


class Patches(layers.Layer):
    def __init__(self, patch_hight, patch_width):
        super(Patches, self).__init__()
        self.patch_hight = patch_hight
        self.patch_width = patch_width

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_hight, self.patch_width, 1],
            strides=[1, self.patch_hight, self.patch_width, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

print(X_train[0, :, :, 0].shape)
d = np.empty((1,128,32,1))
d[0,:,:,0] = X_train[0, :, :, 0]
print(d.shape)

b = tf.image.extract_patches(
            images=d,
            sizes=[1, patch_hight, patch_width, 1],
            strides=[1, patch_hight, patch_width, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

print(X_train[0, :, :, 0].shape[-1])
print(b.shape)
# pp = tf.reshape(b, [128, -1, b.shape[-1]])
# print(pp.shape)


patch = b[0,0,0,:]
patch_img = tf.reshape(patch, (patch_size[0], patch_size[1]))
plt.imshow(patch_img)
plt.imshow(d[0,:,:,0])

plt.figure(figsize=(16, 8))
k=1
for idx in range(b.shape[1]):
  for jdx in range(b.shape[2]):
      patch = b[0, idx, jdx, :]
      ax = plt.subplot(b.shape[1], b.shape[2], k)
      patch_img = tf.reshape(patch, (patch_size[0], patch_size[1]))
      plt.imshow(patch_img)
      plt.axis("off")
      k+=1

if cifar_example:
    # ----------------------------------------------------------------
    # example of how patches selection work on cifar dataset
    (xx_train, yy_train), (xx_test, yy_test) = tf.keras.datasets.cifar100.load_data()
    print(f"x_train shape: {xx_train.shape} - y_train shape: {yy_train.shape}")
    print(f"x_test shape: {xx_test.shape} - y_test shape: {yy_test.shape}")

    print(xx_train[0, :, :, 0].shape)
    d = np.empty((1,32,32,1))
    d[0,:,:,0] = xx_train[10, :, :, 0]
    print(d.shape)

    b = tf.image.extract_patches(
                images=d,
                sizes=[1, 4, 4, 1],
                strides=[1, 4, 4, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )

    print(b.shape)
    patch = b[0,0,0,:]
    patch_img = tf.reshape(patch, (4, 4))
    plt.imshow(patch_img)

    plt.figure(figsize=(4, 4))
    k=1
    for idx in range(b.shape[1]):
      for jdx in range(b.shape[2]):
          patch = b[0, idx, jdx, :]
          ax = plt.subplot(b.shape[1], b.shape[2], k)
          patch_img = tf.reshape(patch, (4, 4))
          plt.imshow(patch_img)
          plt.axis("off")
          k+=1

    plt.figure(figsize=(4, 4))
    plt.imshow(xx_train[10, :, :, 0].astype("uint8"))
    # ----------------------------------------------------------------

# plt.imshow(X_train[0, 1, :, :])
# plt.colorbar()
#
#
# patches = Patches(patch_hight, patch_width)(d)
# print(f"Image size: {image_size[0]} X {image_size[1]}")
# print(f"Patch size: {patch_size[0]} X {patch_size[1]}")
# print(f"Patches per image: {patches.shape[1]}")
# print(f"Elements per patch: {patches.shape[-1]}")
#
# n = int(np.sqrt(patches.shape[1]))
# plt.figure(figsize=(4, 4))
# for i, patch in enumerate(patches[0]):
#     ax = plt.subplot(n, n, i + 1)
#     patch_img = tf.reshape(patch, (patch_size[0], patch_size[1]))
#     plt.imshow(patch_img.numpy().astype("uint8"))
#     plt.axis("off")

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        # x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dense(units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    # augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_hight, patch_width)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # x1 = encoded_patches

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # x3 = x2

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # representation = encoded_patches

    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation) #0.5
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5) #0.5
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)

    # optimizer = tfa.optimizers.AdamW(
    #     learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer = tf.optimizers.Adam(
          learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        experimental_run_tf_function=False
    )

    return model

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = ""
subfolder = ""
net_name = "Transformers_try"
def run_experiment(model):


    checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"
    mc = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True,
                         save_weights_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    history = model.fit(
        x=X_train,
        y=ny_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_val, ny_val),
        validation_freq=1,
        callbacks=[es, mc],
        verbose=1
    )

    # model.load_weights(checkpoint_path)
    # _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


vit_classifier = create_vit_classifier()
vit_classifier.summary()

if doTrain:
    history = run_experiment(vit_classifier)

    fig = plt.figure(figsize=(10, 10))
    # summarize history for loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model losses')
    plt.xlabel('epoch')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


if doTest:

    flat_input = 0
    test_diff_conc_bounds = 0

    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_20/test dataset/'


    # dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'

    def datatestimport():
        global dataset2D, nlabels, w_nlabels, snr_v, shim_v

        # labels = np.transpose(labels,(1,0))
        if test_diff_conc_bounds == 0:
            snr_v = sio.loadmat(dest_folder + 'snr_v_TEST')
            readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST.mat')
            labels_import = sio.loadmat(dest_folder + 'labels_c_TEST.mat')

            labels = labels_import['labels_c'] * 64.5
            snr_v = snr_v['snr_v']
            shim_v = readme_SHIM['shim_v']

            nlabels, w_nlabels = labelsNorm(labels)
        else:
            snr_v = sio.loadmat(dest_folder + 'snr_v_TEST_0406')
            readme_SHIM = sio.loadmat(dest_folder + 'shim_v_TEST_0406.mat')
            labels_import = sio.loadmat(dest_folder + 'labels_c_TEST_0406.mat')

            labels = labels_import['labels_c'] * 64.5
            snr_v = snr_v['snr_v']
            shim_v = readme_SHIM['shim_v']

            labels_import_orig = sio.loadmat(dest_folder + 'labels_c_TEST.mat')
            labels_orig = labels_import_orig['labels_c'] * 64.5
            nlabels, w_nlabels = labelsNormREDdataset(labels, labels_orig)

        if test_diff_conc_bounds == 0:
            data_import2D = sio.loadmat(dest_folder + 'dataset_spgram_TEST.mat')
        else:
            data_import2D = sio.loadmat(dest_folder + 'dataset_spgram_TEST_0406.mat')
        dataset2D = data_import2D['output']

        if flat_input:
            dataset2D = inputConcat2D(dataset2D)

        return dataset2D, nlabels, w_nlabels, snr_v, shim_v


    datatestimport()

    checkpoint_path = outpath + folder + subfolder + net_name + ".best.hdf5"
    vit_classifier = create_vit_classifier()
    vit_classifier.load_weights(checkpoint_path)
    loss = vit_classifier.evaluate(dataset2D, nlabels, verbose=2)

    pred_abs = vit_classifier.predict(dataset2D)  # normalized [0-1] absolute concentrations prediction
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


    # -------------------------------------------------------------
    # plot joint distribution of regression
    # -------------------------------------------------------------
    def jointregression(index, met, outer=None, sharey=0, sharex=0):
        from matplotlib import gridspec
        import seaborn as sns
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.patches as mpatches
        from matplotlib.ticker import FormatStrFormatter

        # fig = plt.figure()
        if outer == None:
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                                  wspace=0.05, hspace=0.05)
        else:
            gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer, width_ratios=[3, 1], height_ratios=[1, 3],
                                                  wspace=0.05, hspace=0.05)
        # ----------------------------------------------
        x = y_test[:, index].reshape(-1, 1)
        y = pred[:, index]
        regr.fit(x, y)
        lin = regr.predict(np.arange(np.min(y_test[:, index]), np.max(y_test[:, index]), 0.01).reshape(-1, 1))
        mse = mean_squared_error(x, y)
        r_sq = regr.score(x, y)

        # ----------------------------------------------

        ax2 = plt.subplot(gs[2])
        p1 = ax2.scatter(y_test[:, index], pred[:, index], c=snr_v, cmap='summer', label='observation')
        # p1 = ax2.scatter(y_test[:, index], pred[:, index], c='darkgreen', cmap='summer', label='observation')
        m = np.max(y_test[:, index])
        ax2.plot(np.arange(np.min(y_test[:, index]), m, 0.01), lin, color='tab:olive', linewidth=3)
        ident = [np.min(y_test[:, index]), m]
        ax2.plot(ident, ident, '--', linewidth=3, color='k')
        # ax1 = plt.subplot(gs[1])

        if outer == None:
            cbaxes = inset_axes(ax2, width="30%", height="3%", loc=2)
            plt.colorbar(p1, cax=cbaxes, orientation='horizontal')

        if outer != None:
            if sharex:
                ax2.set_xlabel('Ground Truth [mM]')
            if sharey:
                ax2.set_ylabel('Predictions [mM]')

        # ax2.plot(np.arange(0, m, 0.01), lin - np.sqrt(mse), color = 'tab:orange', linewidth=3)
        # ax2.plot(np.arange(0, m, 0.01), lin + np.sqrt(mse), color = 'tab:orange', linewidth=3)

        mP = np.min(y)
        MP = np.max(y)
        ax2.set_xlim(np.min(y_test[:, index]) - (0.05 * m), m + (0.05 * m))
        ax2.set_ylim(mP - (0.05 * MP), MP + (0.05 * MP))

        ax0 = plt.subplot(gs[0])
        ax0.set_title(met, fontweight="bold")
        sns.distplot(y_test[:, index], ax=ax0, color='tab:olive')
        ax0.set_xlim(-0.250, m + 0.250)
        ax0.xaxis.set_visible(False)
        ax0.yaxis.set_visible(False)
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
        ax0.set_xlim(np.min(y_test[:, index]) - (0.05 * m), m + (0.05 * m))

        ax3 = plt.subplot(gs[3])
        sns.distplot(y, ax=ax3, vertical=True, color='tab:olive')
        ax3.set_ylim(mP - (0.05 * MP), MP + (0.05 * MP))
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

        patch_t1 = mpatches.Patch(facecolor='w', label=r'$a=%.3f$' % (regr.coef_[0],))
        patch_t2 = mpatches.Patch(facecolor='w', label=r'$q=%.3f$' % (regr.intercept_,))
        patch_t3 = mpatches.Patch(facecolor='w', label=r'$R^{2}=%.3f$' % (r_sq,))
        patch_t4 = mpatches.Patch(facecolor='w', label=r'$std.=%.3f$ [mM]' % (np.sqrt(mse),))
        patch2 = mpatches.Patch(facecolor='tab:red', label='$y=ax+q$', linestyle='-')
        patch3 = mpatches.Patch(facecolor='k', label='$y=x$', linestyle='--')
        patch4 = mpatches.Patch(facecolor='tab:orange', label='$y=\pm std. \dot x$', linestyle='-')

        # ax1.legend(handles = [p1, patch2, patch3, patch4, patch_t1, patch_t2, patch_t3, patch_t4],bbox_to_anchor=(0.5, 0.3, 0.5, 0.5))

        ax1.axis('off')
        # gs.tight_layout()


    def blandAltmann_Shim(index, met, outer=None, sharey=0, sharex=0):
        from matplotlib import gridspec
        from matplotlib.ticker import FormatStrFormatter
        x = y_test[:, index]
        diff = pred[:, index] - x
        shim = shim_v[:, 0]

        idx_s = np.argsort(shim)
        sort = np.sort(shim)

        s_diff = diff[idx_s]
        std_diff = np.empty((y_test.shape[0], 1))
        bsize = 125

        nbins = np.int(y_test.shape[0] / bsize)

        m_bin = np.empty((nbins, 1))
        m_std = np.empty((nbins, 1))
        vlines = np.empty((nbins, 1))
        for i in range(nbins):
            bin = idx_s[i * bsize:((i + 1) * bsize) - 1]
            std_diff[i * bsize:((i + 1) * bsize)] = np.std(diff[bin])

            m_bin[i] = (np.max(sort[i * bsize:((i + 1) * bsize)]) - np.min(
                sort[i * bsize:((i + 1) * bsize)])) / 2 + np.min(sort[i * bsize:((i + 1) * bsize)])
            vlines[i] = np.max(sort[i * bsize:((i + 1) * bsize)])
            m_std[i] = np.std(diff[bin])

        if outer == None:
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3],
                                  wspace=0.05, hspace=0.05)
        else:
            gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, height_ratios=[1, 3],
                                                  wspace=0.05, hspace=0.05)

        ax0 = plt.subplot(gs[0])
        ax0.plot(sort, std_diff[:, 0], 'lightgray', linewidth=0.5)
        ax0.plot(m_bin[:, 0], m_std[:, 0], 'tab:green')
        ax0.scatter(m_bin[:, 0], m_std[:, 0], c='tab:green', s=10)
        for i in np.arange(vlines.shape[0]):
            ax0.axvline(vlines[i, 0], 0, 1, color='gray', alpha=0.5, linewidth=0.5)
        ax0.xaxis.set_visible(False)
        ax0.set_title(met, fontweight="bold")

        mm = np.mean(std_diff[:, 0])
        ax0.set_ylim(mm - 0.8 * mm, mm + 0.8 * mm)

        ax1 = plt.subplot(gs[1])
        ax1.scatter(sort, s_diff, c=snr_v[idx_s], cmap='summer')
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


    def blandAltmann_SNR(index, met, outer=None, xlabel='noise', sharey=0, sharex=0):
        from matplotlib import gridspec
        from matplotlib.ticker import FormatStrFormatter
        x = y_test[:, index]
        diff = pred[:, index] - x
        # rel_noise = np.multiply(1/x, 1/snr_v[:,0])

        snr = snr_v[:, 0]
        noise = 1 / snr_v[:, 0]
        noise_over_gt = noise / x

        if xlabel == 'noise':
            idx_s = np.argsort(noise)
            sort = np.sort(noise)
        elif xlabel == 'snr':
            idx_s = np.argsort(snr)
            sort = np.sort(snr)
        else:
            idx_s = np.argsort(noise_over_gt)
            sort = np.sort(noise_over_gt)

        s_diff = diff[idx_s]
        std_diff = np.empty((y_test.shape[0], 1))
        bsize = 125
        nbins = np.int(y_test.shape[0] / bsize)

        m_bin = np.empty((nbins, 1))
        m_std = np.empty((nbins, 1))
        vlines = np.empty((nbins, 1))
        for i in range(nbins):
            bin = idx_s[i * bsize:((i + 1) * bsize)]
            m_bin[i] = (np.max(sort[i * bsize:((i + 1) * bsize)]) - np.min(
                sort[i * bsize:((i + 1) * bsize)])) / 2 + np.min(sort[i * bsize:((i + 1) * bsize)])
            vlines[i] = np.max(sort[i * bsize:((i + 1) * bsize)])
            # m_bin[i] = np.mean(sort[i * bsize:((i + 1) * bsize)])
            std_diff[i * bsize:((i + 1) * bsize)] = np.std(diff[bin])
            m_std[i] = np.std(diff[bin])

        if outer == None:
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 3],
                                  wspace=0.05, hspace=0.05)
        else:
            gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer, height_ratios=[1, 3],
                                                  wspace=0.05, hspace=0.05)

        ax0 = plt.subplot(gs[0])
        ax0.plot(sort, std_diff[:, 0], 'lightgray', linewidth=0.5)
        ax0.plot(m_bin[:, 0], m_std[:, 0], 'tab:green')
        ax0.scatter(m_bin[:, 0], m_std[:, 0], c='tab:green', s=10)
        for i in np.arange(len(m_bin)):
            ax0.axvline(vlines[i, 0], 0, 1, color='gray', alpha=0.5, linewidth=0.5)
        ax0.xaxis.set_visible(False)
        ax0.set_title(met, fontweight="bold")

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


    order = [2, 0, 4, 12, 7, 6, 1, 8, 9, 14, 10, 3, 13, 15, 11, 5]  # to order metabolites plot from good to bad

    doSNR = 1
    doShim = 1


    # single plots to check
    # fig = plt.figure()
    # jointregression(0, metnames[0])
    # fig = plt.figure()
    # blandAltmann_SNR(3, metnames[3], None, 'noisegt')

    # fig = plt.figure()
    # blandAltmann_SNR(3, metnames[3], None, 'noise')
    # fig = plt.figure()
    # blandAltmann_SNR(3, metnames[3], None, 'snr')
    # fig = plt.figure()
    # blandAltmann_Shim(3, metnames[3])

    # -------------------------------------------------------------
    # plot regression 2x4
    # -------------------------------------------------------------
    def plotREGR2x4fromindex(i):
        fig = plt.figure(figsize=(40, 10))

        widths = 2 * np.ones(4)
        heights = 2 * np.ones(2)
        spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                height_ratios=heights)
        for row in range(2):
            for col in range(4):
                ax = fig.add_subplot(spec[row, col])
                if (i == 0) or (i == 8):
                    jointregression(order[i], metnames[order[i]], spec[row, col], sharey=1)
                elif (i == 4) or (i == 12):
                    jointregression(order[i], metnames[order[i]], spec[row, col], sharex=1, sharey=1)
                elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                    jointregression(order[i], metnames[order[i]], spec[row, col], sharex=1)
                else:
                    jointregression(order[i], metnames[order[i]], spec[row, col])

                i += 1


    # -------------------------------------------------------------
    # plot regression 4x4
    # -------------------------------------------------------------
    def plotREGR4x4fromindex(i):
        fig = plt.figure(figsize=(40, 30))

        widths = 2 * np.ones(4)
        heights = 2 * np.ones(4)
        spec = fig.add_gridspec(ncols=4, nrows=4, width_ratios=widths,
                                height_ratios=heights,
                                top=0.97,
                                bottom=0.06,
                                left=0.06,
                                right=0.97,
                                hspace=0.2,
                                wspace=0.2)

        for row in range(4):
            for col in range(4):
                ax = fig.add_subplot(spec[row, col])
                if ((col == 1) or (col == 2) or (col == 3)) and (row == 3):
                    jointregression(order[i], metnames[order[i]], spec[row, col], sharex=1)
                elif (col == 0) and ((row == 0) or (row == 1) or (row == 2)):
                    jointregression(order[i], metnames[order[i]], spec[row, col], sharey=1)
                elif (col == 0) and (row == 3):
                    jointregression(order[i], metnames[order[i]], spec[row, col], sharex=1, sharey=1)
                else:
                    jointregression(order[i], metnames[order[i]], spec[row, col])

                i += 1


    #
    # def plotREGR_paper_fromindex(i):
    #     fig = plt.figure(figsize = (10,40))
    #
    #     widths = 2*np.ones(3)
    #     heights = 2*np.ones(4)
    #     spec = fig.add_gridspec(ncols=3, nrows=4, width_ratios=widths,
    #                               height_ratios=heights)
    #     for row in range(4):
    #         for col in range(3):
    #             ax = fig.add_subplot(spec[row,col])
    #             if (i==0) or (i==8):
    #                 jointregression(order[i], metnames[order[i]], spec[row,col], sharey=1)
    #             elif (i==4) or (i==12):
    #                 jointregression(order[i], metnames[order[i]], spec[row,col], sharex=1, sharey=1)
    #             elif (i==5) or (i==6) or (i==7) or (i==13) or (i==14) or (i==15):
    #                 jointregression(order[i], metnames[order[i]], spec[row, col], sharex=1)
    #             else:
    #                 jointregression(order[i], metnames[order[i]], spec[row, col])
    #
    #             i += 1

    # plotREGR_paper_fromindex(0)
    plotREGR2x4fromindex(0)
    plotREGR2x4fromindex(8)
    plotREGR4x4fromindex(0)

    if doSNR:
        def plotSNR2x4fromindex(i):
            fig = plt.figure(figsize=(40, 20))

            widths = 2 * np.ones(4)
            heights = 2 * np.ones(2)
            spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                    height_ratios=heights)
            for row in range(2):
                for col in range(4):
                    ax = fig.add_subplot(spec[row, col])

                    if (i == 0) or (i == 8):
                        blandAltmann_SNR(order[i], metnames[order[i]], outer=spec[row, col], xlabel='noise', sharey=1)
                    elif (i == 4) or (i == 12):
                        blandAltmann_SNR(order[i], metnames[order[i]], outer=spec[row, col], xlabel='noise', sharex=1,
                                         sharey=1)
                    elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                        blandAltmann_SNR(order[i], metnames[order[i]], outer=spec[row, col], xlabel='noise', sharex=1)
                    else:
                        blandAltmann_SNR(order[i], metnames[order[i]], outer=spec[row, col], xlabel='noise')

                    i += 1


        plotSNR2x4fromindex(0)
        plotSNR2x4fromindex(8)

    if doShim:
        def plotSHIM2x4fromindex(i):
            fig = plt.figure(figsize=(40, 20))

            widths = 2 * np.ones(4)
            heights = 2 * np.ones(2)
            spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                    height_ratios=heights)

            for row in range(2):
                for col in range(4):
                    ax = fig.add_subplot(spec[row, col])

                    if (i == 0) or (i == 8):
                        blandAltmann_Shim(order[i], metnames[order[i]], outer=spec[row, col], sharey=1)
                    elif (i == 4) or (i == 12):
                        blandAltmann_Shim(order[i], metnames[order[i]], outer=spec[row, col], sharex=1, sharey=1)
                    elif (i == 5) or (i == 6) or (i == 7) or (i == 13) or (i == 14) or (i == 15):
                        blandAltmann_Shim(order[i], metnames[order[i]], outer=spec[row, col], sharex=1)
                    else:
                        blandAltmann_Shim(order[i], metnames[order[i]], outer=spec[row, col])

                    i += 1


        plotSHIM2x4fromindex(0)
        plotSHIM2x4fromindex(8)


    def scores(index):
        # ----------------------------------------------
        x = y_test[:, index].reshape(-1, 1)
        y = pred[:, index]
        regr.fit(x, y)
        lin = regr.predict(np.arange(0, np.max(y_test[:, index]), 0.01).reshape(-1, 1))
        mse = mean_squared_error(x, y)
        r_sq = regr.score(x, y)

        return regr.coef_[0], regr.intercept_, r_sq, mse


    excelname = '/' + net_name + '0mu_test_eval.xlsx'
    workbook = xlsxwriter.Workbook(outpath + folder + subfolder + excelname)
    worksheet = workbook.add_worksheet()
    for i in range(16):
        a, q, r2, mse = scores(i)
        s = 'A' + str(i * 4 + 1)
        worksheet.write(s, a)
        s = 'A' + str(i * 4 + 2)
        worksheet.write(s, q)
        s = 'A' + str(i * 4 + 3)
        worksheet.write(s, r2)
        s = 'A' + str(i * 4 + 4)
        worksheet.write(s, mse)

    workbook.close()
    print('xlsx SAVED')
