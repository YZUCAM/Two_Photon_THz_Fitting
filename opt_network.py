from util import *
import torch
import torch.nn as nn

class OptNetwork(nn.Module):
    def __init__(self, eps, w, FTY1, corrected_FTY2):
        super().__init__()

        self.eps = nn.Parameter(eps, requires_grad=True)
        self.w = w
        self.FTY1 = FTY1
        self.corrected_FTY2 = corrected_FTY2

        self.r41 = calculate_r41(self.w)

    def forward(self):
        # Calculate n
        n = calculate_n(self.w, self.eps)
        # Calculate G
        G = calculate_G(self.w, n)
        # Calculate Response
        R = G*self.r41

        # Calculate corrected FTY
        FTY1_corrected = self.FTY1/R

        # Calculate loss
        loss = torch.mean((torch.abs(FTY1_corrected)/torch.max(torch.abs(FTY1_corrected)) - torch.abs(self.corrected_FTY2)/torch.max(torch.abs(self.corrected_FTY2)))**2)
        return loss
    

