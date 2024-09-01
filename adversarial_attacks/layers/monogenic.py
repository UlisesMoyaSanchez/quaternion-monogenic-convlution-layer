#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["E. Ulises Moya","Abraham Sanchez"]
__copyright__ = "Copyright 2021, Gobierno de Jalisco, Universidad Autonoma de Guadalajara"
__credits__ = ["Abraham Sanchez", "E. Ulises Moya"]
__license__ = "MIT"
__version__ = "0.0.2"
__maintainer__ = ["Abraham Sanchez", "E. Ulises Moya"]
__email__ = "eduardo.moya@jalisco.gob.mx"
__status__ = "Development"

import time

import torch, os
import torch.nn as nn
from torch.nn.modules import Module
from utils import Utils

class Monogenic(Module):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #sx=0.5382136341359487, sy=0.556393863323947, wave_length1=0.1
    #sx=1.1, sy=1.1, wave_length1=0.1
    #sx = 0.2334, sy=-0.1834, wl=102.4516
    def __init__(self, sx=1.1, sy=1.1, wave_length1=0.1, trainable=False):
        super(Monogenic, self).__init__()
        self.sigmax = nn.Parameter(data=torch.as_tensor(sx, dtype=torch.float64, device=self.DEVICE), requires_grad=trainable)
        self.sigmay = nn.Parameter(data=torch.as_tensor(sy, dtype=torch.float64, device=self.DEVICE), requires_grad=trainable)
        self.wave_length1 = nn.Parameter(data=torch.as_tensor(wave_length1, dtype=torch.float64, device=self.DEVICE), requires_grad=False)

        self.plotDir = f'./mon_storage/plots'
        self.step_trigger = False
        self.utils = Utils()

    def forward(self, inputs):
        # if self.step_trigger:
        #     self.utils.imshow(inputs[0], 'inputs', self.plotDir, unNormalize=True)

        x = torch.mean(inputs, dim=1)

        # if self.step_trigger:
        #     self.utils.imshow(x[0], 'x', self.plotDir)

        _, channels, cols, rows = inputs.shape
        gaborgenic = self.gabor(cols=cols, rows=rows, central_frequency=self.wave_length1, sx=self.sigmax, sy=self.sigmay, theta=0, offset=0)

        # if self.step_trigger:
        #     self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(gaborgenic))), f'log gabor\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)

        output = self.compute_gaborgenic(inputs=x, gaborgenic=gaborgenic, cols=cols, rows=rows)

        # if self.step_trigger:
        #     self.utils.imshow(output[0][0:3], 'output 1', self.plotDir)
        #     self.utils.imshow(output[0][3:6], 'output 2', self.plotDir)

        return output
        
    def compute_gaborgenic(self, inputs, gaborgenic, cols, rows):
        im = torch.fft.fftn(inputs).to(self.DEVICE)
        # self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(im[0]))), f'log im', self.plotDir)

        gaborgenic = torch.fft.fftn(gaborgenic)

        # if self.step_trigger:
        #     self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(gaborgenic))), f'log gaborgenic\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)

        hr1, hr2 = self.riesz_trans(cols, rows)
        # self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(hr1))), f'hr1\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        # self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(hr2))), f'hr2\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)

        #Convolution in Fourier Domain
        imf = im * gaborgenic
        # imh1_ = gaborgenic * hr1
        imh1 = im * gaborgenic * hr1
        # self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(imh1_))), f'log gaborgenic * hr1\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        # self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(imh1[0]))), f'log imh1\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        # imh2_ = gaborgenic * hr2
        imh2 = im * gaborgenic * hr2
        # self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(imh2_))), f'log gaborgenic * hr2\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        # self.utils.imshow(torch.log(torch.abs(torch.fft.fftshift(imh2[0]))), f'log imh2\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        f = torch.fft.ifftn(imf).real

        # print(f'f type: {type(f)} {f.shape}')

        h1 = torch.fft.ifftn(imh1).real
        # self.utils.imshow(torch.abs(torch.fft.fftshift(h1[0])), f'log h1\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        h2 = torch.fft.ifftn(imh2).real
        # self.utils.imshow(torch.abs(torch.fft.fftshift(h2[0])), f'log h2\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        ### Local orientation
        ori = torch.atan(torch.divide(-h2, h1 + 1e-6))
        # self.utils.imshow(torch.abs(torch.fft.fftshift(ori[0])), f'log ori\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        fr = torch.as_tensor(torch.sqrt(h1 ** 2 + h2 ** 2) + 1e-6, device=self.DEVICE)
        # self.utils.imshow(torch.abs(torch.fft.fftshift(fr[0])), f'log fr\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        ### Local phase
        ft = torch.as_tensor(torch.atan2(f, fr), device=self.DEVICE)
        # self.utils.imshow(torch.abs(torch.fft.fftshift(ft[0])), f'log ft\n{self.sigmax}_{self.sigmay}_{self.wave_length1}', self.plotDir)
        ones = torch.ones_like(inputs, device=self.DEVICE)
        fts = self.scale_max_min(ft)
        frs = self.scale_max_min(fr)
        oris = self.scale_max_min(ori)
        hsv_tensor_v = torch.stack((fts, frs, ones), dim=1)
        rgb_tensor_v = self.hsv_to_rgb(tensor=hsv_tensor_v, shape=inputs.shape)
        hsv_tensor_o = torch.stack((oris, frs, ones), dim=1)
        rgb_tensor_o = self.hsv_to_rgb(tensor=hsv_tensor_o, shape=inputs.shape)
        rgb_tensor = torch.cat([rgb_tensor_o, rgb_tensor_v], dim=1)
        # rgb_tensor = (rgb_tensor_o + rgb_tensor_v) / 2

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
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_min = torch.amin(x, dim=(1, 2), keepdim=True)
        x_max = torch.amax(x, dim=(1, 2), keepdim=True)
        scale = torch.nan_to_num((x - x_min) / (x_max - x_min), 1e-5)
        return torch.as_tensor(scale, device=DEVICE)

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
        g[:] = torch.div(torch.exp(-0.5 * (rotx ** 2 / sx ** 2 + roty ** 2 / sy ** 2)),  2 * torch.pi * sx * sy) * \
            torch.cos(2 * torch.tensor(torch.pi) * central_frequency * rotx + offset)
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