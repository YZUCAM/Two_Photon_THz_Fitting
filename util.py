import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
c= 299792458  # m/s
hbar= 6.582119569e-16  #eV.s
wavenumber2eV=0.000124

hWto = 367.3 * wavenumber2eV   # eV  367.3
hWlo = 403 * wavenumber2eV     # 403
gamma = 4.3 * wavenumber2eV    # 4.3

# initial parameters value
# epsiloni = 9.075              # 9.075    original value
epsiloni = 9.033256530761719 
# C = -0.47                   # Original value
C = -0.5165228247642517 
# ng = 3.556                    # n =Group velocity for laser gate =3.556 @835nm
# d = 200e-6                      # original value
d = 0.00018944150360766798

# group velocity has 5 points
# ng = [3.7120037, 3.622208,  3.6213386, 3.5292218, 3.529887,  3.527322, 3.4351704]     # original value
ng = [3.752796, 3.6570277, 3.6570172, 3.56134, 3.5613406, 3.5613346, 3.4630003]
# the weight of each group velocity, extracted by the 800nm spectrum 
weight = torch.asarray([0.3, 0.6, 0.8, 1, 0.8, 0.6, 0.3]).to(device)
wl = [772.6, 784.5, 790.9, 800.08, 808.1, 813.6, 820.9]




# functions
def pd_2_tensor(arr):
    return torch.tensor(arr.values)

def calculate_r41(w):
    r41=(1+C*(1-((hbar*w)**2-1j*hbar*w*gamma)/hWto**2)**(-1)).to(device)
    return r41

def calculate_n(w, eps): # epsilon need to optimized
    n = torch.sqrt((1+(hWlo**2-hWto**2)/(hWto**2-(hbar*w)**2-1j*gamma*hbar*w))*eps)
    return n

def calculate_G(w, n):
    G = []
    for i in range(len(ng)):
        # G_temp = 2/(n+1)*self.C*(torch.exp(-1j*self.w*self.d*(self.ng[i]-n)/self.c)-1)/(-1j*self.w*self.d*(self.ng[i]-n))
        G_temp = 2/(n+1)*C*(torch.exp(-1j*w*d*(ng[i]-n)/c)-1)/(-1j*w*d*(ng[i]-n))
        G.append(G_temp)
    G_tensor = torch.stack(G).to(device)
    weighted_G = G_tensor * weight[:, torch.newaxis]
    cal_G = torch.sum(weighted_G, axis=0)
    return cal_G

def gen_freqs(times):
    freqs = torch.linspace(0, len(times)/(max(times)-min(times)), len(times))
    return freqs

def gen_angular_freqs(freqs):
    w = 2 * torch.pi * freqs
    return w

def gen_THz_spectrum(T):
    fty = torch.fft.fft(T)
    fty = torch.abs(fty).to(torch.float32)
    return fty

def gen_THz_complex_spectrum(T):
    fty = torch.fft.fft(T)
    return fty

def single_side_fft(freqs, w, fty):
    pts = len(freqs)
    fshift = freqs[0:pts//2]
    fty = fty[0:pts//2]
    w = w[0:pts//2]
    w[0] = 1e-12        # change the first 0 frequency
    w = torch.asarray(w)
    return fshift, w, fty

def window_specturm(ROI: list, freqs, w, fty):
    s, t = ROI
    fty = fty[s:t]
    w = w[s:t]
    freqs = freqs[s:t]
    return freqs, w, fty

def cal_detector_response(w, eps):
    # Calculate n
    n = calculate_n(w, eps)
    # Calculate G
    G = calculate_G(w, n)
    # Calculate Response
    R = G * calculate_r41(w)
    return R

# training function
def train_step(model: nn.Module,
               optimizer: torch.optim.Optimizer,
               max_epochs: int,
               device: torch.device):
    
    train_loss, train_acc = 0, 0
    results = {"train_loss": []}

    model.to(device)

    for epoch in range(max_epochs):
        loss = model()
        train_loss = loss

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch+1}/{max_epochs} | Train loss: {train_loss:.5f}")

    results["train_loss"].append(train_loss.cpu().detach().numpy())
    # results["eps"].append(model.calspec.epsiloni.cpu().detach().numpy())
    # results["C"].append(model.calspec.C.cpu().detach().numpy())
    # results["ng"].append(model.calspec.ng.cpu().detach().numpy())
    # results["d"].append(model.calspec.d.cpu().detach().numpy())

    print('---------------------------------------------------------')
    # print(f'eps : {model.eps.cpu().detach().numpy()} \n')
    # print(f'C : {model.calspec.C.cpu().detach().numpy()} \n')
    # print(f'ng : {model.calspec.ng.cpu().detach().numpy()} \n')
    # print(f'd : {model.calspec.d.cpu().detach().numpy()} \n')
    # print(f'gamma : {(model.calspec.gamma.cpu().detach().numpy())/0.000124} \n')
    # print(f'gamma : {model.calspec.gamma} \n')

    return results, [model.eps.cpu().detach()]