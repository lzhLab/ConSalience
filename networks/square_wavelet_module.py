import os
import cv2
import datetime
import torch
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim, nn


def gabor_square(sigma, theta, lambd, gamma, ksize, psi):
    """psi=0 return sin square_wave.
       psi=1 return cos square_wave."""
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    xmax = int(ksize[0] / 2)
    ymax = int(ksize[1] / 2)
    xmax = np.ceil(max(1, xmax))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (x, y) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gabor_triangle = np.zeros([ksize[0], ksize[1]])

    if psi == 0:
        for j in range(0, 30, 1):
            gabor_triangle_tmp = 4 / math.pi * (1 / (2 * j + 1)) * np.cos(2 * np.pi / lambd * ((2 * j + 1)) * x_theta + j * math.pi )
            gabor_triangle = gabor_triangle + gabor_triangle_tmp
    if psi == 1:
        for j in range(0, 30, 1):
            gabor_triangle_tmp = - 4 / math.pi * (1 / (2 * j + 1)) * np.cos(2 * np.pi / lambd * ((2 * j + 1)) * x_theta + math.pi/2)
            gabor_triangle = gabor_triangle + gabor_triangle_tmp
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * gabor_triangle
    return gb



def get_wavelet_weight_tensor(sigma,lambd,gamm,kernel_size,parity):
    wavlet_kernel_list = []
    theta = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    if parity == 0 :
        for i, t in enumerate(theta):
            theta_wavlet_list = []
            wavlet_kernel = gabor_square(ksize=kernel_size, sigma=sigma, theta=t * math.pi / 180 * 2, lambd=lambd, gamma=gamm,
                                              psi=0)
            new_wavlet_kernel = np.where(wavlet_kernel > 0, wavlet_kernel, 0)
            count1 = np.sum(new_wavlet_kernel)
            wavlet_kernel = wavlet_kernel / count1
            for count_theta in range(len(theta)):
                theta_wavlet_list.append(wavlet_kernel)
            theta_wavlet_numpy = np.array(theta_wavlet_list)
            wavlet_kernel_list.append(theta_wavlet_numpy)

    if parity == 1 :
        for i, t in enumerate(theta):
            theta_wavlet_list = []
            wavlet_kernel = gabor_square(ksize=kernel_size, sigma=sigma, theta=t * math.pi / 180 * 2, lambd=lambd,
                                              gamma=gamm,psi=1)
            new_wavlet_kernel = np.where(wavlet_kernel > 0, wavlet_kernel, 0)
            count1 = np.sum(new_wavlet_kernel)
            wavlet_kernel = wavlet_kernel / count1
            for count_theta in range(len(theta)):
                theta_wavlet_list.append(wavlet_kernel)
            theta_wavlet_numpy = np.array(theta_wavlet_list)
            wavlet_kernel_list.append(theta_wavlet_numpy)

    wavlet_kernel_list = np.array(wavlet_kernel_list)
    channel_wavlet_kernel_tensor = torch.tensor(wavlet_kernel_list)
    channel_wavlet_kernel_tensor = channel_wavlet_kernel_tensor.double()
    channel_wavlet_kernel_tensor = channel_wavlet_kernel_tensor.type(torch.FloatTensor)
    return channel_wavlet_kernel_tensor

class Wavelet_block(nn.Module):
    def __init__(self, planes, sigma, lambd, gamm, kernel_size, dilations,strides,paddings):
        super(Wavelet_block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bias1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(1, 1), stride=1, padding=0,
                               padding_mode='zeros',
                               bias=True)
        bias1_weight_tensor = torch.ones(6, 1, 1, 1)
        bias1_weight_tensor = bias1_weight_tensor.type(torch.FloatTensor)
        self.bias1.weight = nn.Parameter(bias1_weight_tensor)

        self.bias2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(1, 1), stride=1, padding=0,
                               padding_mode='zeros',
                               bias=True, groups=6)
        self.wavelet_conv_odd = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=kernel_size, stride=strides, padding=paddings,
                                          padding_mode='zeros',
                                          bias=False, dilation=dilations)
        self.wavelet_conv_odd.weight = nn.Parameter(
            get_wavelet_weight_tensor(sigma, lambd, gamm, kernel_size, parity=1))
        self.wavelet_conv_even = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=kernel_size, stride=strides, padding=paddings,
                                           padding_mode='zeros',
                                           bias=False, dilation=dilations)
        self.wavelet_conv_even.weight = nn.Parameter(
            get_wavelet_weight_tensor(sigma, lambd, gamm, kernel_size, parity=0))
        self.wavelet_conv_odd.weight.requires_grad = False
        self.wavelet_conv_even.weight.requires_grad = False

        self.last_conv_odd = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=(1, 1), stride=1, padding=0,
                                   groups=1,
                                   bias=False)
        self.last_conv_even = nn.Conv2d(in_channels=12, out_channels=1, kernel_size=(1, 1), stride=1, padding=0,
                                   groups=1,
                                   bias=False)
        self.last_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1), stride=1, padding=0,
                                   groups=1,
                                   bias=False)

    def forward(self, x):
        out = self.bias1(x)
        out = self.bias2(out)
        out_odd = self.wavelet_conv_odd(out)
        out_odd = self.last_conv_odd(out_odd)
        out_even = self.wavelet_conv_even(out)
        out_even = self.last_conv_even(out_even)
        out = torch.cat([out_odd, out_even], 1)
        out = self.last_conv(out)

        return out

def conv_double(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 4, kernel_size= (3,3), stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 4, kernel_size=(3, 3), stride=1, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, out_planes, kernel_size= (3,3), stride=1, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )

class Wavelet_square_FPN(nn.Module): 
    def __init__(self):
        super(Wavelet_square_FPN, self).__init__()
        self.wavelet_1 = nn.Sequential(*[Wavelet_block(planes=1,sigma=5,lambd=5,gamm=1,kernel_size=(5,5),dilations= 1,strides=1,paddings=2)])
        self.wavelet_2 = nn.Sequential(*[Wavelet_block(planes=1,sigma=5,lambd=2.5,gamm=1,kernel_size=(5,5),dilations= 1,strides=1,paddings=2)])
        self.wavelet_3 = nn.Sequential(*[Wavelet_block(planes=1,sigma=5,lambd=7.5,gamm=1,kernel_size=(5,5),dilations= 1,strides=1,paddings=2)])

        self.wavelet_4 = nn.Sequential(*[
            Wavelet_block(planes=1, sigma=3, lambd=3, gamm=1, kernel_size=(3, 3), dilations=1, strides=1, paddings=1)])
        self.wavelet_5 = nn.Sequential(*[
            Wavelet_block(planes=1, sigma=3, lambd=1.5, gamm=1, kernel_size=(3, 3), dilations=1, strides=1, paddings=1)])
        self.wavelet_6 = nn.Sequential(*[
            Wavelet_block(planes=1, sigma=3, lambd=6, gamm=1, kernel_size=(3, 3), dilations=1, strides=1, paddings=1)])

        self.wavelet_7 = nn.Sequential(*[
            Wavelet_block(planes=1, sigma=3, lambd=3, gamm=1, kernel_size=(5, 5), dilations=1, strides=1, paddings=2)])
        self.wavelet_8 = nn.Sequential(*[
            Wavelet_block(planes=1, sigma=3, lambd=3, gamm=1, kernel_size=(3, 3), dilations=2, strides=1, paddings=2)])
        self.wavelet_9 = nn.Sequential(*[
            Wavelet_block(planes=1, sigma=5, lambd=10, gamm=1, kernel_size=(5, 5), dilations=2, strides=1, paddings=4)])
        self.up_conv = conv_double(9, 1)


    def forward(self, x):

        w1 = self.wavelet_1(x)
        w2 = self.wavelet_2(x)
        w3 = self.wavelet_3(x)
        w4 = self.wavelet_4(x)
        w5 = self.wavelet_5(x)
        w6 = self.wavelet_6(x)
        w7 = self.wavelet_7(x)
        w8 = self.wavelet_8(x)
        w9 = self.wavelet_9(x)

        w_all = torch.cat([w1, w2,w3,w4,w5,w6,w7,w8,w9], 1)
        out = self.up_conv(w_all)

        return out
