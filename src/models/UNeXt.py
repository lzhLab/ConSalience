# src/models/UNeXt.py
# -*- coding: utf-8 -*-

"""
UNeXt-style 3D segmentation network.

Input:
    (B, in_channels, D, H, W)

Output:
    (B, out_channels, D, H, W)

The network uses:
    - anisotropic 3D downsampling: only H/W are downsampled
    - depthwise separable 3D convolution
    - channel expansion and residual MLP-style blocks
    - U-Net skip connections

This implementation is compatible with the existing trainer and evaluator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class LayerNorm3d(nn.Module):
    """
    Channel-wise LayerNorm for tensors shaped as (B, C, D, H, W).
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, D, H, W) -> (B, D, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        # (B, D, H, W, C) -> (B, C, D, H, W)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class DepthwiseSeparableConv3d(nn.Module):
    """
    Depthwise separable 3D convolution.

    The depthwise convolution extracts local spatial features, followed by
    a pointwise convolution that mixes channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 3, 3),
        padding=(1, 1, 1),
        stride=(1, 1, 1),
    ):
        super().__init__()

        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.pointwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class ConvStem3D(nn.Module):
    """
    Initial feature projection.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        groups = _valid_num_groups(out_channels)

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNeXtBlock3D(nn.Module):
    """
    UNeXt-style residual block.

    Structure:
        LayerNorm
        depthwise 3D convolution
        channel expansion
        GELU
        channel projection
        residual connection

    The expansion is implemented with 1x1x1 convolutions, analogous to the
    token/channel mixing MLP used in UNeXt-like architectures.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_channels = max(channels, int(channels * expansion))

        self.norm = LayerNorm3d(channels)

        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )

        self.expand = nn.Conv3d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )

        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        self.project = nn.Conv3d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.dropout2 = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        self.layer_scale = nn.Parameter(
            torch.ones(channels) * 1e-2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm(x)
        x = self.depthwise(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.project(x)
        x = self.dropout2(x)

        scale = self.layer_scale.view(1, -1, 1, 1, 1)
        return residual + scale * x


class EncoderStage(nn.Module):
    """
    Encoder stage consisting of several UNeXt blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        dropout: float = 0.0,
        downsample: bool = False,
    ):
        super().__init__()

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                    bias=False,
                ),
                nn.GroupNorm(
                    _valid_num_groups(out_channels),
                    out_channels,
                ),
                nn.GELU(),
            )
        else:
            self.downsample = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        _valid_num_groups(out_channels),
                        out_channels,
                    ),
                    nn.GELU(),
                )
            )

        self.blocks = nn.Sequential(
            *[
                UNeXtBlock3D(
                    channels=out_channels,
                    expansion=4.0,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.blocks(x)


class DecoderStage(nn.Module):
    """
    Decoder stage.

    Upsampling affects only H/W, preserving the depth dimension.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        depth: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
            bias=False,
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                _valid_num_groups(out_channels),
                out_channels,
            ),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[
                UNeXtBlock3D(
                    channels=out_channels,
                    expansion=4.0,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.blocks(x)


class UNeXt(nn.Module):
    """
    UNeXt-style 3D segmentation model.

    Parameters
    ----------
    in_channels:
        Number of input channels.
        - 1: original CT
        - 3: salience-only input
        - 4: CT + three salience channels

    out_channels:
        Number of output channels. Use 1 for binary segmentation.

    base_channels:
        Base feature width. Recommended value: 16.

    dropout:
        Dropout probability in UNeXt blocks.

    depths:
        Number of blocks in each encoder/decoder stage.
        Default is suitable for the current 3D experiments.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        dropout: float = 0.0,
        depths=(2, 2, 2, 2),
    ):
        super().__init__()

        if in_channels < 1:
            raise ValueError(
                f"in_channels must be >= 1, got {in_channels}"
            )

        if out_channels < 1:
            raise ValueError(
                f"out_channels must be >= 1, got {out_channels}"
            )

        if len(depths) != 4:
            raise ValueError(
                f"depths must contain four values, got {depths}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depths = tuple(depths)

        b = base_channels

        # Shared encoder.
        self.stem = ConvStem3D(
            in_channels=in_channels,
            out_channels=b,
        )

        self.enc1 = EncoderStage(
            in_channels=b,
            out_channels=b,
            depth=depths[0],
            dropout=dropout,
            downsample=False,
        )

        self.enc2 = EncoderStage(
            in_channels=b,
            out_channels=b * 2,
            depth=depths[1],
            dropout=dropout,
            downsample=True,
        )

        self.enc3 = EncoderStage(
            in_channels=b * 2,
            out_channels=b * 4,
            depth=depths[2],
            dropout=dropout,
            downsample=True,
        )

        self.enc4 = EncoderStage(
            in_channels=b * 4,
            out_channels=b * 8,
            depth=depths[3],
            dropout=dropout,
            downsample=True,
        )

        # Decoder.
        self.dec3 = DecoderStage(
            in_channels=b * 8,
            skip_channels=b * 4,
            out_channels=b * 4,
            depth=depths[2],
            dropout=dropout,
        )

        self.dec2 = DecoderStage(
            in_channels=b * 4,
            skip_channels=b * 2,
            out_channels=b * 2,
            depth=depths[1],
            dropout=dropout,
        )

        self.dec1 = DecoderStage(
            in_channels=b * 2,
            skip_channels=b,
            out_channels=b,
            depth=depths[0],
            dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.Conv3d(
                b,
                b,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_valid_num_groups(b), b),
            nn.GELU(),
            nn.Conv3d(
                b,
                out_channels,
                kernel_size=1,
            ),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "UNeXt expects input with shape "
                f"(B, C, D, H, W), got {tuple(x.shape)}"
            )

        if x.size(1) != self.in_channels:
            raise ValueError(
                f"UNeXt was built for {self.in_channels} input channels, "
                f"but received {x.size(1)}"
            )

        x1 = self.stem(x)
        x1 = self.enc1(x1)

        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        x = self.dec3(x4, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== UNeXt sanity check ===")
    print(f"Device: {device}")

    for channels in (1, 3, 4):
        model = UNeXt(
            in_channels=channels,
            out_channels=1,
            base_channels=16,
        ).to(device)

        x = torch.randn(
            1,
            channels,
            4,
            64,
            64,
            device=device,
        )

        with torch.no_grad():
            y = model(x)

        print(
            f"in_channels={channels}: "
            f"input={tuple(x.shape)}, "
            f"output={tuple(y.shape)}, "
            f"parameters={count_parameters(model):,}"
        )

        assert y.shape == (1, 1, 4, 64, 64)

    print("UNeXt sanity check passed.")
