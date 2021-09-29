import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from data_load_norm import labelsimport
import h5py
from matplotlib.ticker import FormatStrFormatter

def dataimport2D(folder, filename):
    data_import = sio.loadmat(folder + filename)
    dataset = data_import['output']

    X_train = dataset[0:18000, :, :, :]
    X_val   = dataset[18000:20000, :, :, :]
    # X_test  = dataset[18000:20000, :, :, :]

    return X_train, X_val

def dataimport1D(folder, filename):
    data_import = sio.loadmat(folder + filename)
    dataset = data_import['dataset_spectra']

    X_train = dataset[0:18000, :]
    X_val = dataset[18000:20000, :]
    # X_test = dataset[19000:20000, :]  # unused

    return X_train, X_val

def readme_import(folder, filename):
    data_import = sio.loadmat(folder + filename)
    readme = data_import['readme']
    snr = readme['SNR'][0][0]
    shim = readme["shim"][0][0]
    mmbgw = readme['w_mmbl'][0][0]

    return snr, shim, mmbgw

def dataimportUNET(index):
    global y_train, y_val, y_test, X_train, X_val, X_test

    os.environ["KERAS_BACKEND"] = "theano"
    K.set_image_data_format('channels_last')


    dest_folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/'

    data_import = sio.loadmat(dest_folder + 'spectra_kor_wat.mat')
    labels_import = sio.loadmat(dest_folder + 'labels_kor_' + str(index) + '_NOwat.mat')

    dataset = data_import['spectra_kor']
    labels = labels_import['labels_kor_' + str(index)]

    # X_train = dataset[0:18000, :]
    # X_val = dataset[18000:20000, :]
    X_test = dataset[19000:20000, :]  # unused

    # y_train = labels[0:18000, :]
    # y_val = labels[18000:20000, :]
    y_test = labels[19000:20000, :]

    return X_test, y_test

folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'
dataname = 'dataset_spectra_TEST.mat'
X_train1d, X_val1d = dataimport1D(folder, dataname)
snr, shim, mmbgw = readme_import(folder, 'readme_TEST.mat')

dataname = 'dataset_spgram_TEST.mat'
X_train2d, X_val2d = dataimport2D(folder, dataname)

labelsname = 'labels_c_TEST.mat'
y_train, y_val = labelsimport(folder, labelsname, 'labels_c')

f = h5py.File(folder + 'signals_TEST.mat', 'r')
signals = f['signals']
time = signals['time']
# time.visititems(lambda n,o:print(n, o)) #to print the struct names
t = time['fid_shim_mmbl_snr'][()]

treal = np.empty((4096,2500))
timag = np.empty((4096,2500))
for i in range(2500):
    treal[:,i] = np.array([x[0] for x in t[i]])
    timag[:, i] = np.array([x[1] for x in t[i]])


# fig = plt.figure()
# ax = fig.add_subplot(211, projection='3d')
# ax.plot3D(np.linspace(0,1,1024), 0*np.ones(1024), X_train1d[0,0,:], 'b')
# ax.plot3D(np.linspace(0,1,1024), 1*np.ones(1024), X_train1d[0,1,:], 'r')
# ax.set_box_aspect(aspect = (3,1,1))
# ax = fig.add_subplot(212, projection='3d')
# ax.plot3D(np.linspace(0,1,1024), 2*np.ones(1024), X_train1d[1,0,:], 'b')
# ax.plot3D(np.linspace(0,1,1024), 3*np.ones(1024), X_train1d[1,1,:], 'r')
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(np.linspace(0,1024,1024), 1*np.ones(1024), X_train1d[0,1,:], 'r')
# ax.plot3D(np.linspace(0,1024,1024), 0*np.ones(1024), X_train1d[0,0,:], 'b')
# ax.set_box_aspect(aspect = (3,1,1))
# ax.grid(False)
#
# fig= plt.figure()
# fig.add_subplot(211)
# plt.plot(np.flip(X_train1d[0,0,:]), 'b')
# fig.get_axes()[0].yaxis.set_visible(False)
# fig.get_axes()[0].xaxis.set_visible(False)
# fig.add_subplot(212)
# plt.plot(np.flip(X_train1d[0,1,:]), 'r')
# fig.get_axes()[1].yaxis.set_visible(False)
#

#
# fig = plt.figure()
# plt.imshow(X_train2d[0,:,:,0], cmap=plt.get_cmap('Greys'))
#

fig = plt.figure()
fig.add_subplot(211)
plt.plot(treal[0:1024, 0], 'b')
fig.get_axes()[0].set_title('Real')
fig.get_axes()[0].yaxis.set_visible(False)
fig.get_axes()[0].xaxis.set_visible(False)
fig.add_subplot(212)
plt.plot(timag[0:1024, 0], 'r')
fig.get_axes()[1].set_title('Imag')
fig.get_axes()[1].yaxis.set_visible(False)

fig= plt.figure()
fig.add_subplot(211)
plt.plot(np.flip(X_train1d[0,0, 255:785]), 'b')
fig.get_axes()[0].set_title('Real')
fig.get_axes()[0].yaxis.set_visible(False)
fig.get_axes()[0].xaxis.set_visible(False)
fig.add_subplot(212)
plt.plot(np.flip(X_train1d[0,1,:]), 'r')
fig.get_axes()[1].yaxis.set_visible(False)
fig.get_axes()[1].set_title('Imag')


# -----
# plot of multiple spectra with concentrations and other parameters
# fig = plt.figure()
fig, ax = plt.subplots(3,3,sharey='row')

n = 0
for i in range(3):
    for j in range(3):
        # ax = fig.add_subplot(3, 3, i+1)
        ax[i, j].plot(np.flip(X_train1d[n, 0, 200:785]), 'b') # 300 without water, 200 with water !!!
        # ax.get_axes()[0].set_title('Real')
        # ax.get_axes()[0].yaxis.set_visible(False)
        # ax.get_axes()[0].xaxis.set_visible(False)
        # text
        # textstr = '\n'.join((
        #     r'$SNR: %.2f$' % (snr[n,0],),
        #     r'$SHIM: %.2f$' % (shim[n,0],),
        #     r'$MMBG_w : %.2f$' % (mmbgw[n,0],)))
        # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        # ax[i, j].text(0.05, 0.95, textstr, transform=ax[i, j].transAxes, fontsize=10,
        #          verticalalignment='top', bbox=props)
        #
        # textstr = '\n'.join((
        #     r'$NAA: %.2f$' % (y_train[n,2],),
        #     r'$tCr: %.2f$' % (y_train[n,4],),
        #     r'$tCho : %.2f$' % (y_train[n,0],)))
        # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        # ax[i, j].text(0.95, 0.95, textstr, transform=ax[i, j].transAxes, fontsize=10,
        #         verticalalignment='top', horizontalalignment = 'right', bbox=props)
        ax[i, j].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax[i, j].xaxis.set_visible(False)

        n+=1

fig = plt.figure()
plt.plot(np.linspace(0,1024,1024), np.flip(X_train1d[0,0,:]), 'b', label = 'Real')
plt.plot(np.linspace(1024,2048, 1024), np.flip(X_train1d[0,1,:]), 'r', label = 'Imag')
fig.get_axes()[0].yaxis.set_visible(False)
fig.get_axes()[0].legend()

fig = plt.figure()
fig.add_subplot(211)
plt.imshow(np.flipud(X_train2d[0,:,:,0]).T, cmap=plt.get_cmap('Greys'))
fig.get_axes()[0].xaxis.set_visible(False)
fig.get_axes()[0].set_title('Real')
fig.add_subplot(212)
plt.imshow(np.flipud(X_train2d[0,:,:,1]).T, cmap=plt.get_cmap('Greys'))
fig.get_axes()[1].set_title('Imag')