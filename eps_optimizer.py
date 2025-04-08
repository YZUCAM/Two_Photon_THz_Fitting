import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from opt_network import *
from util import *
plt.style.use('yzoxford-v2')

# loading data
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

ROI = [1, 100]
ROI_freqs, ROI_w, ROI_FTY1 = window_specturm(ROI, fshift, w, FTY1)
ROI_freqs2, ROI_w2, ROI_FTY2 = window_specturm(ROI, fshift, w, FTY2)

R = cal_detector_response(ROI_w, epsiloni).to(device)
ROI_FTY2 = ROI_FTY2.to(device)
ROI_corrected_FTY2 = ROI_FTY2 / R           # complex numbers

eps = torch.ones(len(ROI_w), dtype=torch.float32) * epsiloni


if __name__=="__main__":

    lr = 0.1
    max_epochs = 5000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROI_w = ROI_w.to(device)
    ROI_FTY1 = ROI_FTY1.to(device)
    ROI_corrected_FTY2 = ROI_corrected_FTY2.to(device)

    # prameters send to the model: eps, w, FTY1, corrected_FTY2
    eps_opt_model = OptNetwork(eps, ROI_w, ROI_FTY1, ROI_corrected_FTY2)
    optimizer = torch.optim.Adam(params=eps_opt_model.parameters(), lr=lr)

    results, parameters = train_step(eps_opt_model, optimizer, max_epochs, device)


    opt_eps = parameters[0]
    print(f"quick check of opt eps value: {opt_eps[:20]}")

    # all tensor data in cpu
    ROI_w = ROI_w.cpu().detach()
    opt_R = cal_detector_response(ROI_w, opt_eps).cpu()
    ROI_FTY1 = ROI_FTY1.cpu().detach()
    ROI_corrected_FTY1 = ROI_FTY1 / opt_R
    ROI_corrected_FTY2 = ROI_corrected_FTY2.cpu().detach()

    ROI_corrected_FTY1 = torch.abs(ROI_corrected_FTY1)
    ROI_corrected_FTY2 = torch.abs(ROI_corrected_FTY2)

    plt.figure(0)
    # plt.plot(ROI_freqs*1e-12, ROI_corrected_FTY1/torch.max(ROI_corrected_FTY1), linewidth=3, label=f"3.77 mW high gate") 
    # plt.plot(ROI_freqs*1e-12, ROI_corrected_FTY2/torch.max(ROI_corrected_FTY2), linewidth=3, label=f"0.73 mW low gate") 
    plt.plot(ROI_freqs*1e-12, ROI_corrected_FTY1, linewidth=3, label=f"3.77 mW high gate") 
    plt.plot(ROI_freqs*1e-12, ROI_corrected_FTY2, linewidth=3, label=f"0.73 mW low gate")
    plt.yscale("log") 
    plt.title(f'Corrected THz Spectrum with fre_dep_eps')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("E (a.u.)")
    plt.xlim(left=0)
    plt.legend(loc='upper right')
    plt.savefig("fre_dep_eps_E_frequency.png")
