import pickle
import matplotlib.pyplot as plt
import numpy as np

outpath = 'C:/Users/Rudy/Desktop/DL_models/'
folder = "net_type/"
subfolder = ""
net_name = "ShallowNet-2D2c-hp-miccai-noiseless2"
file_name = outpath + folder + subfolder + net_name + "_history"

net_name2 = "ShallowNet-2D2c-hp-miccai2"
file_name2 = outpath + folder + subfolder + net_name2 + "_history"

open_file = open(file_name, "rb")
history = pickle.load(open_file)
open_file.close()

open_file = open(file_name2, "rb")
history2 = pickle.load(open_file)
open_file.close()

fig = plt.figure()
plt.subplot(121)
plt.plot(np.log(history['loss']), label='loss')
plt.plot(np.log(history['val_loss']), label='val_loss')
plt.subplot(122)
plt.plot(np.log(history2['loss']), label='loss')
plt.plot(np.log(history2['val_loss']), label='val_loss')
plt.title('model losses')
plt.xlabel('epoch')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

fig = plt.figure()
plt.subplot(211)
plt.plot(history['loss'], label='loss', linewidth=7.0)
plt.plot(history['val_loss'], label='val_loss', linewidth=7.0)
plt.subplot(212)
plt.plot(history2['loss'], label='loss', linewidth=7.0)
plt.plot(history2['val_loss'], label='val_loss', linewidth=7.0)
plt.title('model losses')
plt.xlabel('epoch')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)