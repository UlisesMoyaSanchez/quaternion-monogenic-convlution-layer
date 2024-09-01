#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = ["Abraham Sanchez", "E. Ulises Moya", "Guillermo Mendoza"]
__copyright__ = "Copyright 2021, Gobierno de Jalisco, Universidad Autonoma de Guadalajara"
__credits__ = ["Abraham Sanchez", "E. Ulises Moya"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = ["Abraham Sanchez", "E. Ulises Moya", "Guillermo Mendoza"]
__email__ = "abraham.sanchez@jalisco.gob.mx"
__status__ = "Development"


import torch
import torch.nn as nn

from torch.nn.modules import Module


class Monogenic(Module):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, sigma=0.33, wave_length=1.):
        super(Monogenic, self).__init__()
        self.sigma = nn.Parameter(data=torch.as_tensor(sigma, dtype=torch.complex64, device=self.DEVICE))
        self.wave_length = nn.Parameter(data=torch.as_tensor(wave_length, dtype=torch.complex64, device=self.DEVICE))

    def forward(self, inputs):
        x = torch.mean(inputs, dim=1)
        _, channels, cols, rows = inputs.shape
        monogenic = self.monogenic_scale(cols=cols, rows=rows, central_frequency=self.wave_length, sigma=self.sigma)
        output = self.compute_monogenic(inputs=x, monogenic=monogenic)
        return output

    def compute_monogenic(self, inputs, monogenic):
        im = torch.fft.fftn(inputs)
        imf = im * monogenic[0, ...]
        imh1 = im * monogenic[1, ...]
        imh2 = im * monogenic[2, ...]
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
        scale = (x - x_min) / (x_max - x_min)
        return scale

    def meshs(self, size):
        x, y = self.mesh_range(size)
        radius = torch.fft.ifftshift(torch.sqrt(x * x + y * y)).type(torch.complex64)
        x = torch.fft.ifftshift(x).type(torch.complex64)
        y = torch.fft.ifftshift(y).type(torch.complex64)
        return x, y, radius

    def low_pass_filter(self, size, cutoff, n):
        x, y = self.mesh_range(size)
        radius = torch.sqrt(x * x + y * y)
        lpf = torch.fft.ifftshift(1. / (1. + (radius / cutoff) ** (2. * n))).type(torch.complex64)
        return lpf

    def riesz_trans(self, cols, rows):
        u1, u2, qs = self.meshs((rows, cols))
        qs = torch.sqrt(u1 * u1 + u2 * u2)
        qs[0, 0] = 1.
        h1 = (1j * u1) / qs
        h2 = (1j * u2) / qs
        return h1, h2

    def log_gabor_scale(self, cols, rows, wl, c):
        u1, u2, radius = self.meshs((rows, cols))
        radius[0, 0] = 1.
        lp = self.low_pass_filter((rows, cols), .45, 15.)
        log_gabor_denominator = (2. * torch.log(c) ** 2.).type(torch.complex64)
        fo = 1. / wl
        log_rad_over_fo = torch.log(radius / fo)
        log_gabor = torch.exp(-(log_rad_over_fo * log_rad_over_fo) / log_gabor_denominator)
        log_gabor = lp * log_gabor
        return log_gabor

    def monogenic_scale(self, cols, rows, central_frequency, sigma):
        h1, h2 = self.riesz_trans(cols, rows)
        lg = self.log_gabor_scale(cols, rows, central_frequency, sigma)
        lg_h1 = lg * h1
        lg_h2 = lg * h2
        monogenic = torch.stack([lg, lg_h1, lg_h2], dim=0)
        return monogenic

    def mesh_range(self, size):
        rows, cols = size
        if cols % 2:
            x_values = torch.arange(-(cols - 1) / 2., ((cols - 1) / 2.) + 1, device=self.DEVICE) / float(cols - 1)
        else:
            x_values = torch.arange(-cols / 2., cols / 2., device=self.DEVICE) / float(cols)
        if rows % 2:
            y_values = torch.arange(-(rows - 1) / 2., ((rows - 1) / 2.) + 1, device=self.DEVICE) / float(rows - 1)
        else:
            y_values = torch.arange(-rows / 2., rows / 2., device=self.DEVICE) / float(rows)
        return torch.meshgrid(x_values, y_values)
