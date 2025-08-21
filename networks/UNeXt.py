# -*- coding: utf-8 -*-
"""
Implementation of the UNeXt network for image segmentation tasks.
UNeXt combines the strengths of U-Net and Transformer architectures.

Original UNeXt paper:
     Valanarasu J M J, Patel V M. :
    "Unext: Mlp-based rapid medical image segmentation network"
    International conference on medical image computing and computer-assisted intervention, 2022: 23-33.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import init_weights
from .transformer import TransformerEncoderLayer, TransformerEncoder


class UNeXt(nn.Module):
    """
       Class Definition and Initialization.
    """
    def __init__(self, in_channels, n_classes, img_dim, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNeXt, self).__init__()
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        self.img_dim = img_dim

        # Downsampling path
        self.conv1 = UnetConv2(in_channels, filters[0], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm, kernel_size=3, padding_size=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # Transformer encoder
        self.transformer = self._create_transformer(filters[3], num_layers=2, num_heads=8)

        # Upsampling path
        self.up_concat4 = UnetUp2(filters[3], filters[2], self.is_deconv)
        self.up_concat3 = UnetUp2(filters[2], filters[1], self.is_deconv)
        self.up_concat2 = UnetUp2(filters[1], filters[0], self.is_deconv)
        self.up_concat1 = UnetUp2(filters[0], filters[0], self.is_deconv)

        # Final convolution layer
        self.final = nn.Conv2d(filters[0], n_classes, 1)

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

        # Transformer encoder
        batch_size = maxpool4.size(0)
        transformer_input = maxpool4.view(batch_size, -1, self.img_dim // 16, self.img_dim // 16)
        transformer_input = transformer_input.permute(0, 2, 3, 1).contiguous()
        transformer_output = self.transformer(transformer_input)
        transformer_output = transformer_output.permute(0, 3, 1, 2).contiguous()

        # Upsampling path
        up4 = self.up_concat4(conv4, transformer_output)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Final output
        final = self.final(up1)

        return final

    def _create_transformer(self, input_dim, num_layers, num_heads):
        encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=input_dim * 2,
                dropout=0.1,
                activation='relu'
            ) for _ in range(num_layers)
        ])
        return TransformerEncoder(encoder_layers, num_layers=num_layers)


class UnetUp2(nn.Module):
    """
        Helper classes for upsampling operations
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
