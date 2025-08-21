# -*- coding: utf-8 -*-
"""
Implementation of the Attention U-Net network for image segmentation tasks.
The Attention U-Net incorporates attention gates to focus on important regions and suppress irrelevant regions.

Original Attention U-Net paper:
    Ozan Oktay, Jacopo Folgocci, Marko Davatzikos, et al.:
    Attention U-Net: Learning Where to Look for the Pancreas. 
    arXiv:1804.03999 [cs.CV]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_weights


class AttnUNet(nn.Module):
    """
       Class Definition and Initialization.
       The constructor initializes the network with parameters that control the network's structure and behavior.
    """
    def __init__(self, in_channels, n_classes, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(AttnUNet, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling path
        self.conv1 = UnetConv2(in_channels, filters[0], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm, kernel_size=3, padding_size=1)

        # Attention gates and upsampling path
        self.up_concat4 = AttnGatedUp(filters[4], filters[3], filters[3], self.is_deconv)
        self.up_concat3 = AttnGatedUp(filters[3], filters[2], filters[2], self.is_deconv)
        self.up_concat2 = AttnGatedUp(filters[2], filters[1], filters[1], self.is_deconv)
        self.up_concat1 = AttnGatedUp(filters[1], filters[0], filters[0], self.is_deconv)

        # Final convolution layer
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        # Downsampling
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Center
        center = self.center(maxpool4)

        # Upsampling with attention gates
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Final output
        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


class AttnGatedUp(nn.Module):
    """
        Attention gates are used to weigh the importance of different regions in the skip connections.
    """
    def __init__(self, in_size, skip_size, out_size, is_deconv):
        super(AttnGatedUp, self).__init__()
        self.conv = UnetConv2(in_size + out_size, out_size, False, kernel_size=3, padding_size=1)
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Attention gate
        self.attention = AttentionGate(F_l=in_size, F_g=in_size, F_int=in_size // 2)

        # Initialize weights
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, skip_conn, inputs):
        # Apply attention gate to skip connection
        attn_skip = self.attention(skip_conn, inputs)

        # Upsample inputs
        up = self.up(inputs)

        # Handle differences in feature map sizes
        if up.size()[2:] != attn_skip.size()[2:]:
            up = F.interpolate(up, size=attn_skip.size()[2:], mode='bilinear', align_corners=True)

        return self.conv(torch.cat([attn_skip, up], 1))


class AttentionGate(nn.Module):
    """
        Attention gates are used to weigh the importance of different regions in the skip connections.
    """
    def __init__(self, F_l, F_g, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Initialize weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x, g):
        # Apply convolution to the downsampled features
        g = self.W_g(g)

        # Apply convolution to the skip connection features
        x = self.W_x(x)

        # Resize features to match dimensions
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        # Compute attention coefficients
        psi = self.psi(torch.relu(g + x))

        # Apply attention coefficients to skip connection
        return x * psi


class UnetConv2(nn.Module):
    """
        A helper class for standard convolutional blocks.
    """
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=3, padding_size=1, init_stride=1):
        super(UnetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # Initialize weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
