import numpy as np
import matplotlib.pyplot as plt

cam0 = np.loadtxt('/home/sid/NeuROAM_data/omega_cam0_prior.csv', delimiter=',', skiprows=1)
cam1 = np.loadtxt('/home/sid/NeuROAM_data/omega_cam1_prior.csv', delimiter=',', skiprows=1)

t       = cam0[:, 0]
sig0    = cam0[:, 1]
sig1    = cam1[:, 1]
dT      = t[1] - t[0]

corr = np.correlate(sig1, sig0, 'full')
lags = np.arange(-(len(sig0)-1), len(sig0)) * dT

peak_idx = corr.argmax()
peak_lag = lags[peak_idx]

# correlation value at lag=0 and at lag=+50ms
idx_zero   = len(sig0) - 1          # lag=0 index in full correlation
idx_50ms   = idx_zero + 5           # lag=+50ms
idx_n50ms  = idx_zero - 5           # lag=-50ms

print("Correlation at lag=  0ms : %.4f" % corr[idx_zero])
print("Correlation at lag=+50ms : %.4f" % corr[idx_50ms])
print("Correlation at lag=-50ms : %.4f" % corr[idx_n50ms])
print("Difference 0ms vs +50ms  : %.4f  (%.4f%%)" % (
    corr[idx_zero] - corr[idx_50ms],
    100.0 * abs(corr[idx_zero] - corr[idx_50ms]) / corr[idx_zero]))
print("Peak lag found           : %.4fs" % peak_lag)
print("SNR                      : %.4f" % (corr[peak_idx] / np.sort(corr)[-2]))

# zoom plot around lag=0 ± 500ms
window = int(0.5 / dT)
center = len(sig0) - 1
plt.figure(figsize=(10, 4))
plt.plot(lags[center-window:center+window] * 1000,
         corr[center-window:center+window])
plt.axvline(x=0,    color='gray', linestyle='--', label='lag=0')
plt.axvline(x=50,   color='red',  linestyle='--', label='lag=+50ms (true)')
plt.axvline(x=-50,  color='red',  linestyle=':',  label='lag=-50ms')
plt.xlabel('Lag (ms)')
plt.ylabel('Cross-correlation')
plt.title('Correlation zoomed ±500ms — code is correct, signal is uninformative')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/sid/NeuROAM_data/xcorr_zoom.png')
plt.show()
