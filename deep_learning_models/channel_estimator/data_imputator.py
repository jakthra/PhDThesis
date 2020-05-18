import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, **kwargs):
        super(BasicBlock, self).__init__()
        self.kernel_size = kwargs.get('kernel_size') if kwargs.get('kernel_size') else (1,1)
        self.padding = kwargs.get('padding') if kwargs.get('padding') else 0
        self.stride = kwargs.get('stride') if kwargs.get('stride') else 1
        self.dilation = kwargs.get('dilation') if kwargs.get('dilation') else 1
        self.max_pool = kwargs.get('max_pool') if kwargs.get('max_pool') else (2,2)
    
    def get_output_size(self, h_w):
        from math import floor
        if type(self.kernel_size) is not tuple:
            kernel_size = (self.kernel_size, self.kernel_size)
        h = floor( ((h_w[0] + (2 * self.padding) - ( self.dilation * (self.kernel_size[0] - 1) ) - 1 )/ self.stride) + 1)
        w = floor( ((h_w[1] + (2 * self.padding) - ( self.dilation * (self.kernel_size[1] - 1) ) - 1 )/ self.stride) + 1)
        size = [int(h/self.max_pool[0]), int(w/self.max_pool[1])]
        return size


class BasicConvBlock(BasicBlock):

    def __init__(self, channels, z_dim, max_pool, leaky_relu,  **kwargs):
        super(BasicConvBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(channels, z_dim, **kwargs)
        self.relu = nn.LeakyReLU(negative_slope=leaky_relu)
        self.batchnorm = nn.BatchNorm2d(z_dim)
        self.pool = nn.MaxPool2d(kernel_size=max_pool)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.pool(x)
        return x

class BasicDeconvBlock(nn.Module):

    def __init__(self, z_dim):
        super(BasicDeconvBlock, self).__init__()
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(z_dim)
    
    def forward(self, x, upsample_shape):
        x = torch.nn.functional.interpolate(x, size=(upsample_shape[2], upsample_shape[3]), mode='bilinear')
        x = self.relu(x)
        x = self.batchnorm(x)
        return x


class DataImputator(nn.Module):
    def __init__(self, channels, z_dim, x_dim, t_dim, dropout=0.1):
        super(DataImputator, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Encoder side
        self.block1 = BasicConvBlock(channels, z_dim, (2,2), 0.2, kernel_size=(9,9), padding=4, stride=1)
        self.block2 = BasicConvBlock(z_dim, z_dim, (2,2), 0.1, kernel_size=(1,1))
        self.block3 = BasicConvBlock(z_dim, 1, (2,2), 0.2, kernel_size=(5,5), padding=2)

        l1_size = self.block1.get_output_size([x_dim, t_dim])
        l2_size = self.block2.get_output_size(l1_size)
        l3_size = self.block3.get_output_size(l2_size)
        print("Calculated l3_size {}".format(l3_size))

        # Latent layer
        self.Z = nn.Linear(l3_size[0]*l3_size[1], l3_size[0]*l3_size[1])

        # Decoder side
        self.deblock1 = BasicDeconvBlock(1)
        self.deblock2 = BasicDeconvBlock(z_dim)
        self.deblock3 = BasicDeconvBlock(z_dim)
        self.conv_out = nn.Conv2d(z_dim, channels, kernel_size=(1,1))
        
    def MC(self, g, num_samples=10):

        self.dropout.train(mode=True)
        pred = torch.empty((num_samples, g.shape[0], g.shape[1], g.shape[2], g.shape[3]))
        for sample in range(num_samples):
            pred[sample] = self.forward(g)

        std = pred.std(dim=0)
        mean = pred.mean(dim=0)
        self.dropout.train(mode=False)

        return mean, std

    def forward(self, g):

        batch_size = g.shape[0]
        # Encoder

        l1_out = self.block1(g)
        l2_out = self.block2(l1_out)
        l3_out = self.block3(l2_out)
        x = l3_out.view(batch_size, -1)
  
        x = self.Z(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = x.view(l3_out.shape)
        x = x + l3_out
        x = self.deblock1(x, l2_out.shape)
        x = x + l2_out
        x = self.deblock2(x, l1_out.shape)
        x = x + l1_out
        x = self.deblock3(x, g.shape)
        out = self.conv_out(x)

        return out