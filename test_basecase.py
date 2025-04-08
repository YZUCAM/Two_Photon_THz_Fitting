# all tensor is not in the same device. need tweak the code

from util import *
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('yzoxford-v2')

data_dir = '//home/z/zhuy/Yi_Zhu/Two_Photon_THz_Fitting/Data/'
file1 = 'THz_high_gate_fluence.csv'     # 3.77 mW
file2 = 'THz_low_gate_fluence.csv'      # 0.73 mW

df = pd.read_csv(data_dir+file1, skiprows=[0,1])
df2 = pd.read_csv(data_dir+file2, skiprows=[0,1])

# load x-axis data
position = df.iloc[:-1, 1]
T1 = pd_2_tensor(df.iloc[:-1, 2])
T2 = pd_2_tensor(df2.iloc[:-1, 2])
times = pd_2_tensor(position * 2e-3 / c)

freqs = gen_freqs(times)
w = gen_angular_freqs(freqs)
FTY1_raw = gen_THz_complex_spectrum(T1)
FTY2_raw = gen_THz_complex_spectrum(T2)

fshift, w, FTY1 = single_side_fft(freqs, w, FTY1_raw)
fshift, w, FTY2 = single_side_fft(freqs, w, FTY2_raw)

ROI = [1, 90]
ROI_freqs, ROI_w, ROI_FTY1 = window_specturm(ROI, fshift, w, FTY1)
ROI_freqs2, ROI_w2, ROI_FTY2 = window_specturm(ROI, fshift, w, FTY2)

R = cal_detector_response(ROI_w, epsiloni)
corrected_FTY1 = ROI_FTY1 / R
corrected_FTY2 = ROI_FTY2 / R           # complex numbers

# extract mod just before plotting
FTY1 = torch.abs(FTY1)
FTY2 = torch.abs(FTY2)
ROI_FTY1 = torch.abs(ROI_FTY1)
ROI_FTY2 = torch.abs(ROI_FTY2)
corrected_FTY1 = torch.abs(corrected_FTY1)
corrected_FTY2 = torch.abs(corrected_FTY2)

# plot the data
plt.figure(0)
plt.plot(fshift*1e-12, FTY1/torch.max(FTY1), linewidth=3, label=f"3.77 mW high gate") 
plt.plot(fshift*1e-12, FTY2/torch.max(FTY2), linewidth=3, label=f"0.73 mW low gate") 
plt.yscale("log") 
plt.title(f'Raw THz Spectrum')
plt.xlabel("Frequency (THz)")
plt.ylabel("E (a.u.)")
plt.xlim(left=0)
plt.legend(loc='upper right')
plt.savefig("Raw_E_frequency.png")
# plt.show()

plt.figure(1)
plt.plot(ROI_freqs*1e-12, ROI_FTY1/torch.max(ROI_FTY1), linewidth=3, label=f"3.77 mW high gate") 
plt.plot(ROI_freqs*1e-12, ROI_FTY2/torch.max(ROI_FTY2), linewidth=3, label=f"0.73 mW low gate") 
plt.plot(ROI_freqs*1e-12, corrected_FTY1/torch.max(corrected_FTY1), linewidth=3, label=f"3.73 mW high gate corrected") 
plt.plot(ROI_freqs*1e-12, corrected_FTY2/torch.max(corrected_FTY2), linewidth=3, label=f"0.73 mW low gate corrected") 
plt.yscale("log") 
plt.title(f'Raw THz Spectrum')
plt.xlabel("Frequency (THz)")
plt.ylabel("E (a.u.)")
plt.xlim(left=0)
plt.legend(loc='upper right')
plt.savefig("Raw_E_frequency_windowed.png")