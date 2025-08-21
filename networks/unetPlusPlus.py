# -*- coding: utf-8 -*-
"""
Implementation of the U-Net++ network for image segmentation tasks.
U-Net++ is an enhanced version of U-Net with more powerful feature fusion capabilities.

Original U-Net++ paper:
    Zhou Z, Rahman Siddiquee M M, Tajbakhsh N, et al. :
    "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
    DLMIA and MLHM 2018: 3-11
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_weights


class UNetPlusPlus(nn.Module):
    """
       U-Net++ architecture, which is an enhanced version of the original U-Net with a more sophisticated feature fusion mechanism
       Filters are calculated based on the feature_scale to control network complexity.

       Parameters:
            in_channels: Number of input data channels.
            n_classes: Number of output classes.
            feature_scale: Scaling ratio of the feature channels in the network.
            is_deconv: Indicates whether to use deconvolution for upsampling.
            is_batchnorm: Indicates whether to use batch normalization.
    """
    def __init__(self, in_channels, n_classes, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNetPlusPlus, self).__init__()
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

        # Upsampling path with nested concatenation
        self.up_concat41 = UnetUp2(filters[4], filters[3], self.is_deconv)
        self.up_concat31 = UnetUp2(filters[3], filters[2], self.is_deconv)
        self.up_concat21 = UnetUp2(filters[2], filters[1], self.is_deconv)
        self.up_concat11 = UnetUp2(filters[1], filters[0], self.is_deconv)

        self.up_concat42 = UnetUp2(filters[4] + filters[3], filters[3], self.is_deconv)
        self.up_concat32 = UnetUp2(filters[3] + filters[2], filters[2], self.is_deconv)
        self.up_concat22 = UnetUp2(filters[2] + filters[1], filters[1], self.is_deconv)

        self.up_concat43 = UnetUp2(filters[4] + 2 * filters[3], filters[3], self.is_deconv)
        self.up_concat33 = UnetUp2(filters[3] + 2 * filters[2], filters[2], self.is_deconv)

        self.up_concat44 = UnetUp2(filters[4] + 3 * filters[3], filters[3], self.is_deconv)

        # Final convolution layers
        self.final1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final2 = nn.Conv2d(filters[1], n_classes, 1)
        self.final3 = nn.Conv2d(filters[2], n_classes, 1)
        self.final4 = nn.Conv2d(filters[3], n_classes, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        # Downsampling path
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

        # Upsampling path with multiple nested levels
        # First level up
        up41 = self.up_concat41(conv4, center)
        up31 = self.up_concat31(conv3, up41)
        up21 = self.up_concat21(conv2, up31)
        up11 = self.up_concat11(conv1, up21)

        # Second level up
        up42 = self.up_concat42(torch.cat([center, up41], 1), center)
        up32 = self.up_concat32(torch.cat([conv3, up31], 1), up42)
        up22 = self.up_concat22(torch.cat([conv2, up21], 1), up32)

        # Third level up
        up43 = self.up_concat43(torch.cat([center, up41, up42], 1), center)
        up33 = self.up_concat33(torch.cat([conv3, up31, up32], 1), up43)

        # Fourth level up
        up44 = self.up_concat44(torch.cat([center, up41, up42, up43], 1), center)

        # Final outputs from different levels
        final1 = self.final1(up11)
        final2 = self.final2(up22)
        final3 = self.final3(up33)
        final4 = self.final4(up44)

        # Fuse outputs from different levels
        final = (final1 + final2 + final3 + final4) / 4

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p


class UnetUp2(nn.Module):
    """
       Multiple upsampling blocks are defined to gradually restore spatial dimensions.
       These blocks utilize nested concatenation to combine features from different levels.

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
       The center block further processes the deepest features using a UnetConv2 block.
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
