import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from data_load_norm import labelsimport
import h5py

def dataimport2D(folder, filename):
    data_import = sio.loadmat(folder + filename)
    dataset = data_import['output_nw']

    X_train = dataset[0:18000, :, :, :]
    X_val   = dataset[18000:20000, :, :, :]
    # X_test  = dataset[18000:20000, :, :, :]

    return X_train, X_val

def dataimport1D(folder, filename):
    data_import = sio.loadmat(folder + filename)
    dataset = data_import['dataset_spectra_nw']

    X_train = dataset[0:18000, :]
    X_val = dataset[18000:20000, :]
    # X_test = dataset[19000:20000, :]  # unused

    return X_train, X_val


folder = 'C:/Users/Rudy/Desktop/datasets/dataset_31/test dataset/'
dataname = 'dataset_spectra_nw_TEST.mat'
X_train1d, X_val1d = dataimport1D(folder, dataname)

dataname = 'dataset_spgram_nw_TEST.mat'
X_train2d, X_val2d = dataimport2D(folder, dataname)

labelsname = 'labels_c_TEST.mat'
y_train, y_val = labelsimport(folder, labelsname)


f = h5py.File(folder + 'signals_TEST.mat', 'r')
signals = f['signals']
time = signals['time']
# time.visititems(lambda n,o:print(n, o)) #to print the struct names
t = time['fid_nowater_shim_mmbl_snr'][()]

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
plt.plot(np.flip(X_train1d[0,0,:]), 'b')
fig.get_axes()[0].set_title('Real')
fig.get_axes()[0].yaxis.set_visible(False)
fig.get_axes()[0].xaxis.set_visible(False)
fig.add_subplot(212)
plt.plot(np.flip(X_train1d[0,1,:]), 'r')
fig.get_axes()[1].yaxis.set_visible(False)
fig.get_axes()[1].set_title('Imag')


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