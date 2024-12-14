from argparse import Namespace
from collections import Counter
import csv
import gc
from itertools import product
from re import S
from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from numpy import array, pad
import os
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import torch
import torch.nn as nn

# Fourier Transform Block
class FT_block(nn.Module):
    """
    A module that applies a Fourier Transform to the input tensor, multiplies it by learnable weights,
    and then applies an inverse Fourier Transform. This block can be used for frequency domain operations.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(FT_block, self).__init__()
        self.out_channels = out_channels
        # Learnable Fourier transform weights initialized with scaled random values
        self.FT_W = nn.Parameter(torch.rand(in_channels, out_channels, dtype=torch.cfloat)/ (in_channels * out_channels))

    def FT(self, x):
        # Apply Fourier Transform along the time dimension
        x_FT = torch.fft.fft(x, dim=1)
        # Multiply the transformed tensor by the learnable weights using einsum for batch matrix multiplication
        x_FT = torch.einsum("bix,io->box", x_FT, self.FT_W)
        # Apply Inverse Fourier Transform along the time dimension, limiting the output size to out_channels
        x = torch.fft.ifft(x_FT, dim=1, n=self.out_channels)
        # Take the real part of the result as the final output
        x = x.real 
        return x
    
    def forward(self, x):
        x = self.FT(x)
        return x

# Sparse Matrix Block
class SM_block(nn.Module):
    """
    A module that transforms the input tensor into a sparse representation by expanding it in the feature dimension,
    then applies a learnable transformation matrix to this expanded tensor.

    Args:
        in_channels (int): Number of input channels.
        pred_num (int): Prediction number, not directly used but kept for compatibility.
        Features (int): Number of feature channels.
        k (int): Factor by which the time dimension is expanded.
    """
    def __init__(self, in_channels, Features, k):
        super(SM_block, self).__init__()
        self.k = k
        # Learnable transformation matrix initialized with scaled random values
        self.W_SM = nn.Parameter(torch.rand((in_channels, k * Features, Features), dtype=torch.float) / (k * Features), requires_grad=True)
    
    def trans_input(self, x):
        B, T, C = x.shape
        x_trans = torch.zeros(B, T, C * self.k).to(x.device)
        for i in range(self.k):
            x_trans[:, :(T - i), i * C:(i + 1) * C] = x[:, i:, :]
        return x_trans

    def forward(self, x):
        x_Tran = self.trans_input(x)
        out = torch.einsum("bni,nio->bno", x_Tran, self.W_SM)
        return out

# Combined FT and SM Block
class FT_SM_block(nn.Module):
    def __init__(self, in_channels, Features, k_size):
        super(FT_SM_block, self).__init__()
        self.FT = FT_block(in_channels, in_channels)
        self.SM = SM_block(in_channels, Features, k_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_SM = self.SM(x)
        out_FT = self.FT(x)
        out = out_FT + self.relu(out_SM)
        return out

class FT_SMNet(nn.Module):
    def __init__(self, in_channels, out_channels, Features, k_size):
        super(FT_SMNet, self).__init__()
        self.FT_SM = FT_SM_block(in_channels, Features, k_size)
        self.FT = FT_block(in_channels, out_channels)

    def forward(self, x):
        x_ = x[:, -1:, :].detach()
        x = x - x_
        out = self.FT_SM(x)
        out = self.FT(out)
        out = out + x_
        return out

