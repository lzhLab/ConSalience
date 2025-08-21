# -*- coding: utf-8 -*-
"""
Implementation of the U-Net with Salience Generator for image segmentation tasks.
This network incorporates a salience generator to highlight important regions in the input data.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_weights
from .salience_generator import Salience


class UNet_WithSalience(nn.Module):
    """
       1) Class Definition and Initialization
       2) Downsampling Path: Multiple conv layers extract features at different levels. 
       3) Center Block: further processes the features from the deepest level
       4) Upsampling Path: gradually restore the spatial dimensions through upsampling or transposed convolutions.
       5) The salience map is combined with the input to enhance the network's focus on relevant regions.
    """
    def __init__(self, in_channels, n_classes, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_WithSalience, self).__init__()
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

        # Salience generator
        self.salience = Salience(1, 1)

    def forward(self, inputs):
        # Salience generation
        salience_map = self.salience(inputs)
        inputs = inputs + inputs * salience_map

        # Downsampling path
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Center block
        center = self.center(maxpool4)

        # extract the features
        self.featuremap_center = center.detach()

        center = self.dropout1(center)
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
       UnetUp2 blocks combine features from the downsampling path and the center block.
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
       UnetConv2 blocks extract features at different levels.
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
