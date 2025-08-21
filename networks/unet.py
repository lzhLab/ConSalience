# -*- coding: utf-8 -*-
"""
Implementation of the 2D U-Net network for image segmentation tasks.
The code follows a similar structure to the 3D U-Net implementation but is adapted for 2D data.

Original U-Net paper:
    Olaf Ronneberger, Philipp Fischer, and Thomas Brox:
    U-Net: Convolutional Networks for Biomedical Image Segmentation. 
    MICCAI 2015: 234-241
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_weights


class UNet_2D(nn.Module):
    """
       The network has an encoder-decoder architecture with skip connections.
       Parameters:
            in_channels: Number of input data channels.
            n_classes: Number of output classes.
            feature_scale: Scaling ratio of the feature channels in the network.
            is_deconv: Indicates whether to use deconvolution for upsampling.
            is_batchnorm: Indicates whether to use batch normalization.
    """
    def __init__(self, in_channels, n_classes, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_2D, self).__init__()
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

        # Upsampling path
        self.up_concat4 = UnetUp2(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp2(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp2(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp2(filters[1], filters[0], self.is_deconv)

        # Final convolution layer
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

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
        center = self.dropout1(center)

        # Upsampling with concatenation
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        # Final output
        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


class UnetUp2(nn.Module):
    """
        upsampling path gradually increases the spatial resolution while combining features from different levels
    """
    def __init__(self, in_size, out_size, is_deconv):
        super(UnetUp2, self).__init__()
        self.conv = UnetConv2(in_size + out_size, out_size, False, kernel_size=3, padding_size=1)
        
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, in_size, kernel_size=2, stride=2)
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
        The final convolutional layer maps the features to the desired number of output classes.
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
