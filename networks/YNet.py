# -*- coding: utf-8 -*-
"""
Implementation of the Y-Net network for image segmentation tasks.
Y-Net is a multimodal network that can handle multiple input modalities.

Original Y-Net paper:
    Farshad A, Yeganeh Y, Gehlbach P, et al. :
    Y-net: A spatiospectral dual-encoder network for medical image segmentation.
    International conference on medical image computing and computer-assisted intervention, 2022: 582-592.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_weights


class YNet(nn.Module):
    """
      Class Definition and Initialization
      Dual Downsampling Paths:
            Two separate downsampling paths are defined for different modalities (conv1_m1 to conv4_m1 for modality 1 and conv1_m2 to conv4_m2 for modality 2).
            Max pooling layers (maxpool1_m1 to maxpool4_m1 for modality 1 and maxpool1_m2 to maxpool4_m2 for modality 2) reduce the spatial dimensions of the feature maps.
      Fusion Layers:
            Fusion layers (fuse1 to fuse4) combine features from both modalities at different levels.

    """
    def __init__(self, in_channels, n_classes, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(YNet, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling path for modality 1
        self.conv1_m1 = UnetConv2(in_channels, filters[0], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool1_m1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_m1 = UnetConv2(filters[0], filters[1], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool2_m1 = nn.MaxPool2d(kernel_size=2)

        self.conv3_m1 = UnetConv2(filters[1], filters[2], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool3_m1 = nn.MaxPool2d(kernel_size=2)

        self.conv4_m1 = UnetConv2(filters[2], filters[3], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool4_m1 = nn.MaxPool2d(kernel_size=2)

        # Downsampling path for modality 2
        self.conv1_m2 = UnetConv2(in_channels, filters[0], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool1_m2 = nn.MaxPool2d(kernel_size=2)

        self.conv2_m2 = UnetConv2(filters[0], filters[1], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool2_m2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_m2 = UnetConv2(filters[1], filters[2], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool3_m2 = nn.MaxPool2d(kernel_size=2)

        self.conv4_m2 = UnetConv2(filters[2], filters[3], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool4_m2 = nn.MaxPool2d(kernel_size=2)

        # Fusion layers
        self.fuse1 = UnetConv2(filters[0] * 2, filters[0], self.is_batchnorm, kernel_size=1, padding_size=0)
        self.fuse2 = UnetConv2(filters[1] * 2, filters[1], self.is_batchnorm, kernel_size=1, padding_size=0)
        self.fuse3 = UnetConv2(filters[2] * 2, filters[2], self.is_batchnorm, kernel_size=1, padding_size=0)
        self.fuse4 = UnetConv2(filters[3] * 2, filters[3], self.is_batchnorm, kernel_size=1, padding_size=0)

        # Center block
        self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm, kernel_size=3, padding_size=1)

        # Upsampling path
        self.up_concat4 = UnetUp2(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp2(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp2(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp2(filters[1], filters[0], self.is_deconv)

        # Final convolution layer
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs_m1, inputs_m2):
        # Downsampling path for modality 1
        conv1_m1 = self.conv1_m1(inputs_m1)
        maxpool1_m1 = self.maxpool1_m1(conv1_m1)

        conv2_m1 = self.conv2_m1(maxpool1_m1)
        maxpool2_m1 = self.maxpool2_m1(conv2_m1)

        conv3_m1 = self.conv3_m1(maxpool2_m1)
        maxpool3_m1 = self.maxpool3_m1(conv3_m1)

        conv4_m1 = self.conv4_m1(maxpool3_m1)
        maxpool4_m1 = self.maxpool4_m1(conv4_m1)

        # Downsampling path for modality 2
        conv1_m2 = self.conv1_m2(inputs_m2)
        maxpool1_m2 = self.maxpool1_m2(conv1_m2)

        conv2_m2 = self.conv2_m2(maxpool1_m2)
        maxpool2_m2 = self.maxpool2_m2(conv2_m2)

        conv3_m2 = self.conv3_m2(maxpool2_m2)
        maxpool3_m2 = self.maxpool3_m2(conv3_m2)

        conv4_m2 = self.conv4_m2(maxpool3_m2)
        maxpool4_m2 = self.maxpool4_m2(conv4_m2)

        # Fusion of features from both modalities
        fuse1 = self.fuse1(torch.cat([conv1_m1, conv1_m2], 1))
        fuse2 = self.fuse2(torch.cat([conv2_m1, conv2_m2], 1))
        fuse3 = self.fuse3(torch.cat([conv3_m1, conv3_m2], 1))
        fuse4 = self.fuse4(torch.cat([conv4_m1, conv4_m2], 1))

        # Center block
        center = self.center(maxpool4_m1 + maxpool4_m2)

        # Upsampling path
        up4 = self.up_concat4(fuse4, center)
        up3 = self.up_concat3(fuse3, up4)
        up2 = self.up_concat2(fuse2, up3)
        up1 = self.up_concat1(fuse1, up2)

        # Final output
        final = self.final(up1)

        return final


class UnetUp2(nn.Module):
    """
       UnetUp2 blocks combine features from the fusion layers and the center block.
    """
    def __init__(self, in_size, out_size, is_deconv):
        super(UnetUp2, self).__init__()
        self.conv = UnetConv2(in_size + out_size, out_size, False, kernel_size=3, padding_size=1)
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Initialize weights
        for m in self.children():
            if m.__class__.__name__.find('UnetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        # Handle differences in feature map sizes
        if outputs2.size()[2:] != inputs1.size()[2:]:
            outputs2 = F.interpolate(outputs2, size=inputs1.size()[2:], mode='bilinear', align_corners=True)
        return self.conv(torch.cat([inputs1, outputs2], 1))


class UnetConv2(nn.Module):
    """
        Helper classes for convolution operations.
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
