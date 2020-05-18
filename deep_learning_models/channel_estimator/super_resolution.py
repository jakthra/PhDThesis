import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np

class SuperRes(nn.Module):
    def __init__(self, zd, xd, td, channels=2, dropout=0.1):
        super(SuperRes, self).__init__()
        z_dim = zd
        x_dim = xd
        t_dim = td
        

        # Encoder
        conv1_kernel_size = (9,9)
        conv2_kernel_size = (1,1)

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=conv1_kernel_size, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=conv2_kernel_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.conv_out = nn.Conv2d(32, channels, kernel_size=(5,5), padding=2)
        


    def forward(self, g):

        x = self.conv1(g)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        out = self.conv_out(x)

        return out




    