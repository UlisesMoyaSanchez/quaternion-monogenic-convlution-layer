#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["E. Ulises Moya","Abraham Sanchez"]
__copyright__ = "Copyright 2021, Gobierno de Jalisco"
__credits__ = ["Abraham Sanchez", "E. Ulises Moya"]
__license__ = "MIT"
__version__ = "0.0.2"
__maintainer__ = ["Abraham Sanchez", "E. Ulises Moya"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"


import torch
import torch.nn as nn
from torch.nn.modules import Module


class Monogenic(Module):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, sx1=1.1, sy1=1.1, wave_length1=.1, sx2=1.1, sy2=1.1, wave_length2=.1,
                 sx3=1.1, sy3=1.1, wave_length3=.1):
        super(Monogenic, self).__init__()
        self.sigmax1 = nn.Parameter(data=torch.as_tensor(sx1, dtype=torch.float64, device=self.DEVICE))
        self.sigmay1 = nn.Parameter(data=torch.as_tensor(sy1, dtype=torch.float64, device=self.DEVICE))
        self.wave_length1 = nn.Parameter(data=torch.as_tensor(wave_length1, dtype=torch.float64, device=self.DEVICE))
        #--
        self.sigmax2 = nn.Parameter(data=torch.as_tensor(sx2, dtype=torch.float64, device=self.DEVICE))
        self.sigmay2 = nn.Parameter(data=torch.as_tensor(sy2, dtype=torch.float64, device=self.DEVICE))
        self.wave_length2= nn.Parameter(data=torch.as_tensor(wave_length2, dtype=torch.float64, device=self.DEVICE))
        #--
        self.sigmax3 = nn.Parameter(data=torch.as_tensor(sx3, dtype=torch.float64, device=self.DEVICE))
        self.sigmay3 = nn.Parameter(data=torch.as_tensor(sy3, dtype=torch.float64, device=self.DEVICE))
        self.wave_length3= nn.Parameter(data=torch.as_tensor(wave_length3, dtype=torch.float64, device=self.DEVICE))

    def forward(self, inputs):
        #x = torch.mean(inputs, dim=1)
        x1 = inputs[...,0,:,:]
        x2 = inputs[...,1,:,:]
        x3 = inputs[...,2,:,:]
        _, channels, cols, rows = inputs.shape
        if channels != 3:
            raise ValueError('Monogenic layer only supports 3 channels.')

        gaborgenic_1 = self.gabor(cols=cols, rows=rows, central_frequency=self.wave_length1, 
                                sx=self.sigmax1, sy=self.sigmay1, theta=0, offset=0)
        gaborgenic_2 = self.gabor(cols=cols, rows=rows, central_frequency=self.wave_length2, 
                                sx=self.sigmax2, sy=self.sigmay2, theta=0, offset=0)
        gaborgenic_3 = self.gabor(cols=cols, rows=rows, central_frequency=self.wave_length3, 
                                sx=self.sigmax3, sy=self.sigmay3, theta=0, offset=0)
        
        output1 = self.compute_gaborgenic(inputs=x1, gaborgenic=gaborgenic_1, cols=cols, rows=rows)
        output2 = self.compute_gaborgenic(inputs=x2, gaborgenic=gaborgenic_2, cols=cols, rows=rows)
        output3 = self.compute_gaborgenic(inputs=x3, gaborgenic=gaborgenic_3, cols=cols, rows=rows)
        output = torch.cat([output1,output2,output3], dim=1)
        return output
        
    def compute_gaborgenic(self, inputs, gaborgenic, cols, rows):
        im = torch.fft.fftn(inputs)
        gaborgenic = torch.fft.fftn(gaborgenic)
        hr1, hr2 = self.riesz_trans(cols, rows)
        #Convolution in Fourier Domain
        imf = im * gaborgenic
        imh1 = im * gaborgenic*hr1
        imh2 = im * gaborgenic*hr2
        f = torch.fft.ifftn(imf).real
        h1 = torch.fft.ifftn(imh1).real
        h2 = torch.fft.ifftn(imh2).real
        ori = torch.atan(torch.divide(-h2, h1 + 1e-6))
        fr = torch.sqrt(h1 ** 2 + h2 ** 2) + 1e-6
        ft = torch.atan2(f, fr)
        ones = torch.ones_like(inputs)
        fts = self.scale_max_min(ft)
        frs = self.scale_max_min(fr)
        oris = self.scale_max_min(ori)
        hsv_tensor_v = torch.stack((fts, frs, ones), dim=1)
        rgb_tensor_v = self.hsv_to_rgb(tensor=hsv_tensor_v, shape=inputs.shape)
        hsv_tensor_o = torch.stack((oris, frs, ones), dim=1)
        rgb_tensor_o = self.hsv_to_rgb(tensor=hsv_tensor_o, shape=inputs.shape)
        rgb_tensor = torch.cat([rgb_tensor_o, rgb_tensor_v], dim=1)
        return rgb_tensor

    def hsv_to_rgb(self, tensor, shape):
        h = tensor[:, 0, :, :]
        s = tensor[:, 1, :, :]
        v = tensor[:, 2, :, :]
        c = s * v
        m = v - c
        dh = h * 6.
        h_category = torch.as_tensor(dh, dtype=torch.int32, device=self.DEVICE)
        fmodu = dh % 2
        x = c * (1. - torch.abs(fmodu - 1))
        dtype = tensor.dtype
        rr = torch.zeros(shape, dtype=dtype, device=self.DEVICE)
        gg = torch.zeros(shape, dtype=dtype, device=self.DEVICE)
        bb = torch.zeros(shape, dtype=dtype, device=self.DEVICE)
        h0 = torch.eq(h_category, 0)
        rr = torch.where(h0, c, rr)
        gg = torch.where(h0, x, gg)
        h1 = torch.eq(h_category, 1)
        rr = torch.where(h1, x, rr)
        gg = torch.where(h1, c, gg)
        h2 = torch.eq(h_category, 2)
        gg = torch.where(h2, c, gg)
        bb = torch.where(h2, x, bb)
        h3 = torch.eq(h_category, 3)
        gg = torch.where(h3, x, gg)
        bb = torch.where(h3, c, bb)
        h4 = torch.eq(h_category, 4)
        rr = torch.where(h4, x, rr)
        bb = torch.where(h4, c, bb)
        h5 = torch.eq(h_category, 5)
        rr = torch.where(h5, c, rr)
        bb = torch.where(h5, x, bb)
        r = rr + m
        g = gg + m
        b = bb + m
        return torch.stack([r, g, b], dim=1)

    @classmethod
    def scale_max_min(cls, x):
        x_min = torch.amin(x, dim=(1, 2), keepdim=True)
        x_max = torch.amax(x, dim=(1, 2), keepdim=True)
        scale = torch.nan_to_num((x - x_min) / (x_max - x_min),1e-5)
        return scale

    def meshs(self, size):
        x, y = self.mesh_range(size)
        radius = torch.fft.ifftshift(torch.sqrt(x * x + y * y)).type(torch.complex64)
        x = torch.fft.ifftshift(x).type(torch.complex64)
        y = torch.fft.ifftshift(y).type(torch.complex64)
        return x, y, radius



    def riesz_trans(self, cols, rows):
        u1, u2, qs = self.meshs((rows, cols))
        qs = torch.sqrt(u1 * u1 + u2 * u2)
        qs[0, 0] = 1.
        h1 = (1j * u1) / qs
        h2 = (1j * u2) / qs
        return h1, h2

    def gabor(self, cols, rows, central_frequency, sx, sy, theta=0, offset=0):
        #size = rows // 2
        #grid_val = torch.arange(-size, size+1, dtype=torch.float)
        #x, y = torch.meshgrid(grid_val, grid_val)
        x, y = self.mesh_range((cols,rows))
        x = torch.fft.ifftshift(x)
        y = torch.fft.ifftshift(y)
        rotx = x * torch.cos(torch.tensor(theta)) + y * torch.sin(torch.tensor(theta))
        roty = -x * torch.sin(torch.tensor(theta)) + y * torch.cos(torch.tensor(theta))
        g = torch.zeros(y.shape, device=self.DEVICE)
        g[:] = torch.div(torch.exp(-0.5 * (rotx ** 2 / sx ** 2 + roty ** 2 / sy ** 2)),  2 * torch.pi * sx * sy)*torch.cos(2 * torch.tensor(torch.pi) * central_frequency * rotx + offset)
        #g /= 2 * torch.pi * sx * sy         
        #g *= torch.cos(2 * torch.tensor(torch.pi) * central_frequency * rotx + offset)
        return g    



    def mesh_range(self, size):
        rows, cols = size
        if cols % 2:
            x_values = torch.arange(-(cols - 1) / 2., ((cols - 1) / 2.) + 1, device=self.DEVICE) 
        else:
            x_values = torch.arange(-cols / 2., cols / 2., device=self.DEVICE) 
        if rows % 2:
            y_values = torch.arange(-(rows - 1) / 2., ((rows - 1) / 2.) + 1, device=self.DEVICE) 
        else:
            y_values = torch.arange(-rows / 2., rows / 2., device=self.DEVICE)
        return torch.meshgrid(x_values, y_values)
