
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


#matabolite = sio.loadmat('C:\Users\Rudy\Desktop\DL code\simulations\0R_vespa_sim)
dataLoc = 'C:/Users/Rudy/Desktop/DL code/simulations/2_WATER/WATER__4HZ_conc_T2_scaled.txt'
matabolite = sio.loadmat('C:/Users/Rudy/Desktop/DL code/simulations/2_WATER/WATER__4HZ_conc_T2_scaled.txt'


lines[21:]  ## all data, as sig(real)\t sig(imag)\tfft(real)\tfft(imag)\n'

sig_ = np.zeros([2048,2])
fft_ = np.zeros([2048,2])
i = 0
for line in lines[21:]:
    #print(line)
    x = line[:-1].split('\t')
    sig_[i, 0] = x[0]
    sig_[i, 1] =x[1]
    fft_[i, 0] =x[2]
    fft_[i, 1] =x[3]

    i+=1

norm_fft_ = np.divide(fft_, np.max(fft_))  # (2048,2)

def white_noise(rho, sr, n, mu=0):
    sigma = rho * np.sqrt(sr/2)
    noise = np.random.normal(mu, sigma, n)
    return noise

rho = 0.1
sr = 0.1
n = 2048
noise = white_noise(rho, sr, n)

## multiple noise by the GT
GT = norm_fft_
pred = np.zeros([2048,2])
X_noisy = np.zeros([2048,2])
pred[:,0] = np.multiply(GT[:,0], noise)
pred[:,1] = np.multiply(GT[:,0], noise)
X_noisy[:,0] = (GT[:,0] + noise)
X_noisy[:,1] = (GT[:,0] + noise)

X_noisy_ = np.zeros([1,2048,2])
pred_ = np.zeros([1,2048,2])
GT_ = np.zeros([1,2048,2])

X_noisy_[0,:,:] = X_noisy
pred_[0,:,:] = pred
GT_[0,:,:] = GT



# take a spectrum from standard dataset and see the DQ DQw

fig, axs = plt.subplots(3)
fig.suptitle('test result')
axs[0].plot(X_noisy_[0,:,0])
#axs[0].title('X_noisy')
axs[1].plot(pred_[0,:,0])
#axs[1].title('pred_')
axs[2].plot(GT_[0,:,0])
#axs[2].title('GT_')



DQ, DQw, Dqcompl, DQwcompl = calculate_DQ_DQw(pred_, X_noisy_, GT_)

plt.figure()
plt.plot(pred_[:,:,1])
plt.show()