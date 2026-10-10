# src/models/YNet.py
# -*- coding: utf-8 -*-
"""
3D Y-Net for vessel segmentation.

This implementation is compatible with the existing training/evaluation
pipeline in this project.

Input:
    x: (B, C, D, H, W)

Output:
    logits: (B, out_channels, D, H, W)

Design:
    - Shared 3D encoder.
    - Two decoder branches:
        1) vessel decoder
        2) context/boundary decoder
    - Final fusion head combines the two decoder features.
    - Uses anisotropic down/up sampling: only H/W are downsampled, D is preserved.
      This is safer for your current input shape like (B, C, 4, 256, 256).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    Choose a valid GroupNorm group number.
    """
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1


class ConvNormAct(nn.Module):
    """
    Conv3d + GroupNorm + LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        padding=1,
        dropout: float = 0.0,
    ):
        super().__init__()

        groups = _get_num_groups(out_channels)

        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """
    Two consecutive ConvNormAct blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.block = nn.Sequential(
            ConvNormAct(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dropout=dropout,
            ),
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dropout=dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsample only H/W, preserve D.

    Input:
        (B, C, D, H, W)

    Output:
        (B, C_out, D, H/2, W/2)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """
    Upsample only H/W, preserve D.

    Uses ConvTranspose3d with kernel_size=(1,2,2), stride=(1,2,2).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )

        self.conv = DoubleConv(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # In case H/W are odd after preprocessing, align spatial size.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class DecoderBranch(nn.Module):
    """
    One decoder branch of Y-Net.

    Shared encoder features:
        x1: base
        x2: base*2
        x3: base*4
        x4: base*8

    Output:
        feature: base channels, same spatial shape as input.
    """

    def __init__(
        self,
        base_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        b = base_channels

        self.up3 = UpBlock(
            in_channels=b * 8,
            skip_channels=b * 4,
            out_channels=b * 4,
            dropout=dropout,
        )

        self.up2 = UpBlock(
            in_channels=b * 4,
            skip_channels=b * 2,
            out_channels=b * 2,
            dropout=dropout,
        )

        self.up1 = UpBlock(
            in_channels=b * 2,
            skip_channels=b,
            out_channels=b,
            dropout=dropout,
        )

    def forward(
        self,
        x4: torch.Tensor,
        x3: torch.Tensor,
        x2: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return x


class SpatialAttention3D(nn.Module):
    """
    Lightweight 3D spatial attention.

    This is used to refine the context/boundary branch before fusion.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv3d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_map, max_map], dim=1)
        attn = self.conv(attn)
        attn = self.sigmoid(attn)

        return x * attn


class YNet(nn.Module):
    """
    3D Y-Net.

    Parameters
    ----------
    in_channels:
        Input channels.
        - baseline CT: 1
        - salience-only: 3
    out_channels:
        Usually 1 for binary vessel segmentation.
    base_channels:
        Base channel width. Recommended: 16.
    dropout:
        Optional dropout in conv blocks.
    return_aux:
        If True, return (main_logits, vessel_aux_logits, context_aux_logits).
        For compatibility with your current trainer/eval, keep False.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        dropout: float = 0.0,
        return_aux: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.return_aux = return_aux

        b = base_channels

        # Shared encoder
        self.enc1 = DoubleConv(in_channels, b, dropout=dropout)
        self.enc2 = DownBlock(b, b * 2, dropout=dropout)
        self.enc3 = DownBlock(b * 2, b * 4, dropout=dropout)
        self.enc4 = DownBlock(b * 4, b * 8, dropout=dropout)

        # Two decoder branches
        self.vessel_decoder = DecoderBranch(base_channels=b, dropout=dropout)
        self.context_decoder = DecoderBranch(base_channels=b, dropout=dropout)

        # Lightweight attention on context branch
        self.context_attention = SpatialAttention3D(kernel_size=7)

        # Fusion head
        self.fusion = nn.Sequential(
            DoubleConv(b * 2, b, dropout=dropout),
            nn.Conv3d(b, out_channels, kernel_size=1),
        )

        # Optional auxiliary heads
        self.vessel_head = nn.Conv3d(b, out_channels, kernel_size=1)
        self.context_head = nn.Conv3d(b, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        Kaiming initialization for Conv3d / ConvTranspose3d.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x:
            (B, in_channels, D, H, W)

        return:
            logits: (B, out_channels, D, H, W)
        """

        # Shared encoder
        x1 = self.enc1(x)   # (B, b, D, H, W)
        x2 = self.enc2(x1)  # (B, 2b, D, H/2, W/2)
        x3 = self.enc3(x2)  # (B, 4b, D, H/4, W/4)
        x4 = self.enc4(x3)  # (B, 8b, D, H/8, W/8)

        # Branch 1: vessel decoder
        vessel_feat = self.vessel_decoder(x4, x3, x2, x1)

        # Branch 2: context/boundary decoder
        context_feat = self.context_decoder(x4, x3, x2, x1)
        context_feat = self.context_attention(context_feat)

        # Fusion
        fused = torch.cat([vessel_feat, context_feat], dim=1)
        main_logits = self.fusion(fused)

        if self.return_aux:
            vessel_logits = self.vessel_head(vessel_feat)
            context_logits = self.context_head(context_feat)
            return main_logits, vessel_logits, context_logits

        return main_logits


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=== YNet Sanity Check ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for in_ch in [1, 3]:
        model = YNet(
            in_channels=in_ch,
            out_channels=1,
            base_channels=16,
        ).to(device)

        x = torch.randn(2, in_ch, 4, 256, 256).to(device)

        with torch.no_grad():
            y = model(x)

        print()
        print(f"Input channels: {in_ch}")
        print(f"Input shape : {tuple(x.shape)}")
        print(f"Output shape: {tuple(y.shape)}")
        print(f"Parameters  : {_count_parameters(model):,}")

        assert y.shape == (2, 1, 4, 256, 256), (
            f"Unexpected output shape: {y.shape}"
        )

    print()
    print("YNet sanity check passed.")
