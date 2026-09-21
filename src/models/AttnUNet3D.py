# src/models/AttnUNet3D.py
# -*- coding: utf-8 -*-

"""
3D Attention U-Net.

Input:
    x: (B, C, D, H, W)

Output:
    logits: (B, out_channels, D, H, W)

This implementation is designed to be compatible with the existing UNet3D
training pipeline in this project.

Important:
    Downsampling and upsampling are only applied on H/W dimensions.
    The D dimension is kept unchanged by using kernel_size=(1, 2, 2)
    and stride=(1, 2, 2).

This is important for medical 3D volumes where D can be small.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
    Double 3D convolution block:
        Conv3d -> BatchNorm3d -> ReLU
        Conv3d -> BatchNorm3d -> ReLU
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    """
    Downsampling block.

    Only downsample H/W:
        input : (B, C, D, H, W)
        output: (B, C_out, D, H/2, W/2)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.pool = nn.MaxPool3d(
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class AttentionGate3D(nn.Module):
    """
    3D Attention Gate.

    This is the 3D version of the attention gate used in Attention U-Net.

    g:
        gating signal from decoder, shape (B, F_g, D, H, W)

    x:
        skip feature from encoder, shape (B, F_l, D, H, W)

    output:
        attention-weighted skip feature, same shape as x
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm3d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(
                F_l,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm3d(F_int),
        )

        # Keep this structure unchanged if you want to load old checkpoints.
        # Checkpoint-compatible structure:
        #     psi.0 = Conv3d
        #     psi.1 = BatchNorm3d
        #     psi.2 = Sigmoid
        self.psi = nn.Sequential(
            nn.Conv3d(
                F_int,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UpAttention3D(nn.Module):
    """
    Upsampling block with Attention Gate.

    Steps:
        1. ConvTranspose3d upsampling on H/W only.
        2. Align spatial size with skip feature.
        3. Apply AttentionGate3D on skip feature.
        4. Concatenate attended skip and decoder feature.
        5. Apply DoubleConv3D.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()

        # Important:
        # This must be (1, 2, 2), not (2, 2, 2),
        # otherwise old checkpoints cannot be loaded.
        self.up = nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )

        self.att_gate = AttentionGate3D(
            F_g=out_ch,
            F_l=skip_ch,
            F_int=max(out_ch // 2, 1),
        )

        self.conv = DoubleConv3D(
            in_ch=out_ch + skip_ch,
            out_ch=out_ch,
        )

    @staticmethod
    def _align_to_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Align decoder feature x to skip feature size.

        x:
            decoder feature after upsampling

        skip:
            encoder skip feature

        Returns:
            padded/cropped x with the same D/H/W as skip.
        """
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)

        # Usually diff values are >= 0.
        # F.pad supports positive padding.
        # If negative values appear, crop first.
        if diff_d < 0:
            crop_before = (-diff_d) // 2
            crop_after = crop_before + skip.size(2)
            x = x[:, :, crop_before:crop_after, :, :]
            diff_d = 0

        if diff_h < 0:
            crop_before = (-diff_h) // 2
            crop_after = crop_before + skip.size(3)
            x = x[:, :, :, crop_before:crop_after, :]
            diff_h = 0

        if diff_w < 0:
            crop_before = (-diff_w) // 2
            crop_after = crop_before + skip.size(4)
            x = x[:, :, :, :, crop_before:crop_after]
            diff_w = 0

        x = F.pad(
            x,
            (
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
                diff_d // 2,
                diff_d - diff_d // 2,
            ),
        )

        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._align_to_skip(x, skip)

        skip_att = self.att_gate(g=x, x=skip)

        x = torch.cat([skip_att, x], dim=1)
        x = self.conv(x)

        return x


class AttnUNet3D(nn.Module):
    """
    3D Attention U-Net.

    Args:
        in_channels:
            1 for baseline CT input.
            3 for salience-enhanced input.

        out_channels:
            Usually 1 for binary vessel segmentation.

        base_channels:
            Base channel number.
            Common values: 16 or 32.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        c = base_channels

        # Encoder
        self.inc = DoubleConv3D(in_channels, c)
        self.down1 = Down3D(c, c * 2)
        self.down2 = Down3D(c * 2, c * 4)
        self.down3 = Down3D(c * 4, c * 8)

        # Decoder with attention gates
        self.up1 = UpAttention3D(
            in_ch=c * 8,
            skip_ch=c * 4,
            out_ch=c * 4,
        )

        self.up2 = UpAttention3D(
            in_ch=c * 4,
            skip_ch=c * 2,
            out_ch=c * 2,
        )

        self.up3 = UpAttention3D(
            in_ch=c * 2,
            skip_ch=c,
            out_ch=c,
        )

        self.outc = nn.Conv3d(
            c,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward.

        Input:
            x: (B, C, D, H, W)

        Output:
            logits: (B, out_channels, D, H, W)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        logits = self.outc(x)
        return logits


def _test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== AttnUNet3D Sanity Check ===")
    print(f"Device: {device}")

    for in_channels in (1, 3):
        model = AttnUNet3D(
            in_channels=in_channels,
            out_channels=1,
            base_channels=16,
        ).to(device)

        x = torch.randn(
            2,
            in_channels,
            4,
            256,
            256,
            device=device,
        )

        with torch.no_grad():
            y = model(x)

        n_params = sum(p.numel() for p in model.parameters())

        print(f"\nInput channels: {in_channels}")
        print(f"Input shape : {tuple(x.shape)}")
        print(f"Output shape: {tuple(y.shape)}")
        print(f"Parameters  : {n_params:,}")

        assert y.shape == (2, 1, 4, 256, 256), (
            f"Unexpected output shape: {y.shape}"
        )

    print("\nAttnUNet3D sanity check passed.")


if __name__ == "__main__":
    _test()

