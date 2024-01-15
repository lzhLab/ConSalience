# coding=utf-8
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


def gabor_square(sigma=5, theta=0.0, gamma=1, ksize=(5, 5), psi=1):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    lambd = ksize[0] - 1
    xmax = int(ksize[0] / 2)
    xmax = np.ceil(max(1, xmax))
    ymax = int(ksize[1] / 2)
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (x, y) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # 旋转
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    square_wave = np.zeros([ksize[0], ksize[1]])
    # 偶方波分解
    if psi == 0:
        for j in range(0, 30, 1):
            odd_n = 2 * j + 1
            omega = 2 * np.pi / lambd
            wave = (
                4
                / math.pi
                * odd_n**-1
                * np.cos(odd_n * omega * x_theta + j * math.pi)
            )
            square_wave = square_wave + wave
    # 奇方波分解
    if psi == 1:
        for j in range(0, 30, 1):
            odd_n = 2 * j + 1
            omega = 2 * np.pi / lambd
            wave = (
                -4
                / math.pi
                * odd_n**-1
                * np.cos(odd_n * omega * x_theta + math.pi / 2)
            )
            square_wave = square_wave + wave
    # 窗函数
    gb_square_wave = (
        np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
        * square_wave
    )
    # 小波
    return gb_square_wave


def get_weight(in_channels=3, kernel_size=(5, 5), sigma=5, psi=0):
    # 180度等分为12份
    rotation_step = math.pi / 12
    theta_lst = [i * rotation_step for i in range(12)]
    # 生成`out_channels`为12的卷积核
    weight = []
    for t in theta_lst:
        kernel = gabor_square(sigma, theta=t, ksize=kernel_size, psi=psi)
        kernel = torch.from_numpy(kernel).float()
        weight.append(kernel)
    weight = torch.stack(weight)
    weight = weight.unsqueeze(1).repeat(1, in_channels, 1, 1)
    return weight  # (12, in_channels, 5, 5)


class GaborSquare(nn.Module):
    def __init__(self, in_channels, kernel_size=(5, 5), sigma=5):
        super(GaborSquare, self).__init__()
        self.kernel_size = kernel_size
        # 偶波
        self.even_kernel = get_weight(
            in_channels=in_channels, kernel_size=kernel_size, sigma=sigma, psi=0
        ).cuda()
        # 奇波
        self.odd_kernel = get_weight(
            in_channels=in_channels, kernel_size=kernel_size, sigma=sigma, psi=1
        ).cuda()

    def forward(self, x):
        """x:(B, in_channels, H, W)"""
        padding = self.kernel_size[0] // 2
        even_out = F.conv2d(x, self.even_kernel, padding=padding)
        odd_out = F.conv2d(x, self.odd_kernel, padding=padding)
        return even_out + odd_out  # (B, 12, H, W)


class Wavelet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, scales=5):
        super(Wavelet, self).__init__()
        # 3, 5, 7, ..., 2*scales+1
        k_list = [2 * i + 1 for i in range(1, scales+1)]  
        self.gabor_list = [
            GaborSquare(in_channels, kernel_size=(k, k), sigma=k) for k in k_list
        ]
        self.out_conv = nn.Sequential(
            nn.Conv2d(12*scales, 6*scales, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(6*scales, 6*scales, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(6*scales, out_channels, kernel_size=(1, 1), stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 1, H, W, D)
        b, c, h, w, d = x.shape
        x = x.permute(4, 0, 1, 2, 3).contiguous().view(d * b, 1, h, w)
        out_lst = []
        for model in self.gabor_list:
            out_lst.append(model(x))
        # 多尺度特征融合
        out = self.out_conv(torch.cat(out_lst, dim=1))
        return out.view(d, b, 1, h, w).permute(1, 2, 3, 4, 0)


if __name__ == "__main__":
    k = gabor_square(sigma=5, theta=0 * math.pi / 180, ksize=(5, 5), psi=0)
    print(k.round(4))
    k = gabor_square(sigma=5, theta=0 * math.pi / 180, ksize=(5, 5), psi=1)
    print(k.round(4))

    inputs = torch.randn(1, 1, 96, 96, 96).cuda()
    model = Wavelet(in_channels=1, out_channels=1).cuda()
    outputs = model(inputs)
    print(outputs.shape)

    from thop import profile
    from thop import clever_format

    macs, params = profile(model, inputs=(inputs, ))
    macs, params = clever_format([macs, params], '%.3f')

    print('模型参数：', params)
    print('每一个样本浮点运算量：', macs)
