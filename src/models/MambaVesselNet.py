#!/usr/bin/env python3
"""PyTorch implementation of MambaVesselNet++ for 3-D segmentation.

The original project uses a five-level CNN encoder, Mamba blocks at the
bottleneck, and UNETR-style convolutional skip decoding.  This adapter keeps
that layout while matching the model-factory interface used by the
ConSalience experiments::

    model = MambaVesselNet(in_channels=1, out_channels=1, base_channels=8)
    logits = model(volume)                 # (B, C, D, H, W)

For a Salience experiment, the existing trainer should generate the three
Salience channels and construct this model with ``in_channels=3`` (or pass
``use_salience=True`` with the default ``in_channels=1``).  Salience is kept
outside this model so the baseline and Salience runs share exactly the same
network and training loop; the model only consumes the resulting channels.

``mamba_ssm`` is optional.  When it is available, ``MambaTokenMixer`` uses it
in both scan directions.  On CPU-only or minimal installations, a lightweight
bidirectional depthwise sequence mixer is used instead, so the file remains
executable with PyTorch alone.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Int3 = Tuple[int, int, int]


def _valid_groups(channels: int, requested: int = 8) -> int:
    groups = min(int(channels), int(requested))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * noise.floor_()


class LayerNorm3D(nn.Module):
    """LayerNorm over the channel axis for NCDHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight[:, None, None, None] + self.bias[:, None, None, None]


class ConvNormAct3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_valid_groups(out_channels), out_channels),
        ]
        if activation:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock3D(nn.Module):
    """MONAI UnetrBasicBlock-like two-convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_block: bool = True,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct3D(in_channels, out_channels)
        self.conv2 = ConvNormAct3D(out_channels, out_channels, activation=False)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.res_block = bool(res_block)
        self.drop_path = DropPath(drop_path)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv2(self.conv1(x))
        return self.act(self.drop_path(x) + residual) if self.res_block else self.act(x)


class ConvStage3D(nn.Module):
    """A stack of MONAI UnetrBasicBlock-like convolutional blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        drop_path_rates: Sequence[float],
        res_block: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("Each encoder stage must contain at least one block")
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock3D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    res_block=res_block,
                    drop_path=float(drop_path_rates[index]),
                )
                for index in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ChannelMlp3D(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(channels, channels * expansion)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class _FallbackBidirectionalMixer(nn.Module):
    """Small PyTorch-only substitute for Mamba when mamba_ssm is unavailable."""

    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        self.forward_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.backward_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        sequence = value.transpose(1, 2)
        forward = self.forward_conv(sequence)
        reverse = torch.flip(self.backward_conv(torch.flip(sequence, dims=(2,))), dims=(2,))
        mixed = 0.5 * (forward + reverse).transpose(1, 2)
        return self.out_proj(mixed * torch.sigmoid(gate))


class MambaTokenMixer(nn.Module):
    """Apply a bidirectional Mamba-like mixer to flattened 3-D features."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.using_external_mamba = False

        if use_mamba:
            try:
                from mamba_ssm import Mamba  # type: ignore

                self.forward_mamba = Mamba(
                    d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
                )
                self.backward_mamba = Mamba(
                    d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
                )
                self.using_external_mamba = True
            except Exception:
                self.forward_mamba = None
                self.backward_mamba = None
        else:
            self.forward_mamba = None
            self.backward_mamba = None

        if not self.using_external_mamba:
            self.fallback = _FallbackBidirectionalMixer(dim, kernel_size=max(3, d_conv | 1))
        else:
            self.fallback = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) -> tokens: (B, N, C)
        batch, channels = x.shape[:2]
        spatial = x.shape[2:]
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)

        if self.using_external_mamba:
            forward = self.forward_mamba(tokens)
            reverse = torch.flip(
                self.backward_mamba(torch.flip(tokens, dims=(1,))), dims=(1,)
            )
            tokens = 0.5 * (forward + reverse)
        else:
            tokens = self.fallback(tokens)

        return tokens.transpose(1, 2).reshape(batch, channels, *spatial)


class MambaResidualBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        drop_path: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba: bool = True,
    ) -> None:
        super().__init__()
        self.mixer = MambaTokenMixer(
            channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_mamba=use_mamba,
        )
        self.norm = LayerNorm3D(channels)
        self.mlp = ChannelMlp3D(channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mixer(x))
        x = x + self.drop_path(self.mlp(self.norm(x)))
        return x


class MambaStack3D(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        drop_path_rates: Sequence[float],
        d_state: int,
        d_conv: int,
        expand: int,
        use_mamba: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MambaResidualBlock3D(
                    channels,
                    drop_path=float(drop_path_rates[i]),
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_mamba=use_mamba,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = LayerNorm3D(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv3d(kernel=2) cannot process a singleton spatial dimension.
        pads = []
        for size in reversed(x.shape[-3:]):
            pads.extend((0, int(size < 2)))
        if any(pads):
            x = F.pad(x, tuple(pads))
        return self.norm(self.conv(x))


class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.fuse = ResidualConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat((x, skip), dim=1))


def _five_values(values: Optional[Sequence[int]], default: Sequence[int], name: str) -> Tuple[int, ...]:
    if values is None:
        return tuple(int(v) for v in default)
    values = tuple(int(v) for v in values)
    if len(values) == 4:
        values = values + (values[-1],)
    if len(values) != 5:
        raise ValueError(f"{name} must contain four or five values")
    if any(v < 1 for v in values):
        raise ValueError(f"{name} values must be positive")
    return values


class MambaVesselNet(nn.Module):
    """MambaVesselNet++ adapted to the ConSalience model-factory API.

    Parameters named by the previous H-vmunet/DSwinUNet adapters are accepted
    where practical.  Unused compatibility options are deliberately accepted
    through ``**kwargs`` so existing YAML files do not need to be rewritten.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        feature_dims: Optional[Sequence[int]] = None,
        c_list: Optional[Sequence[int]] = None,
        depths: Optional[Sequence[int]] = None,
        bottleneck_depth: int = 4,
        decoder_depth: int = 4,
        drop_path_rate: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_mamba: bool = True,
        use_salience: Optional[bool] = None,
        bridge: bool = True,
        norm_name: str = "group",
        spatial_dims: int = 3,
        res_block: bool = True,
        **kwargs: object,
    ) -> None:
        super().__init__()
        del norm_name, kwargs

        if spatial_dims != 3:
            raise ValueError("MambaVesselNet is a 3-D model and requires spatial_dims=3")
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if base_channels < 1:
            raise ValueError("base_channels must be positive")

        # The existing Salience trainer produces [G_hat_c, G_hat_s, G_hat_t].
        # This convenience keeps model construction safe if only the boolean
        # switch is overridden in YAML.
        self.use_salience = bool(use_salience) if use_salience is not None else in_channels == 3
        if self.use_salience and in_channels == 1:
            in_channels = 3

        default_dims = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        )
        raw_dims = feature_dims if feature_dims is not None else c_list
        if raw_dims is not None and len(tuple(raw_dims)) == 4:
            raw_dims = tuple(raw_dims) + (int(tuple(raw_dims)[-1]) * 2,)
        dims = _five_values(raw_dims, default_dims, "feature_dims/c_list")
        default_depths = (1, 1, 1, 1, 1)
        stage_depths = _five_values(depths, default_depths, "depths")
        if bottleneck_depth < 1 or decoder_depth < 0:
            raise ValueError("bottleneck_depth must be positive and decoder_depth non-negative")

        total_blocks = sum(stage_depths) + bottleneck_depth + decoder_depth
        rates = torch.linspace(0.0, float(drop_path_rate), total_blocks).tolist()
        rate_index = 0

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.feature_dims = dims

        self.enc1 = ConvStage3D(
            in_channels,
            dims[0],
            stage_depths[0],
            rates[rate_index : rate_index + stage_depths[0]],
            res_block=res_block,
        )
        rate_index += stage_depths[0]
        self.down1 = Downsample3D(dims[0], dims[0])
        self.enc2 = ConvStage3D(
            dims[0],
            dims[1],
            stage_depths[1],
            rates[rate_index : rate_index + stage_depths[1]],
            res_block=res_block,
        )
        rate_index += stage_depths[1]
        self.down2 = Downsample3D(dims[1], dims[1])
        self.enc3 = ConvStage3D(
            dims[1],
            dims[2],
            stage_depths[2],
            rates[rate_index : rate_index + stage_depths[2]],
            res_block=res_block,
        )
        rate_index += stage_depths[2]
        self.down3 = Downsample3D(dims[2], dims[2])
        self.enc4 = ConvStage3D(
            dims[2],
            dims[3],
            stage_depths[3],
            rates[rate_index : rate_index + stage_depths[3]],
            res_block=res_block,
        )
        rate_index += stage_depths[3]
        self.down4 = Downsample3D(dims[3], dims[3])
        self.enc5 = ConvStage3D(
            dims[3],
            dims[4],
            stage_depths[4],
            rates[rate_index : rate_index + stage_depths[4]],
            res_block=res_block,
        )
        rate_index += stage_depths[4]

        self.enc_mamba = MambaStack3D(
            dims[4],
            bottleneck_depth,
            rates[rate_index : rate_index + bottleneck_depth],
            d_state,
            d_conv,
            expand,
            use_mamba,
        )
        rate_index += bottleneck_depth

        self.dec_mamba = MambaStack3D(
            dims[4],
            decoder_depth,
            rates[rate_index : rate_index + decoder_depth],
            d_state,
            d_conv,
            expand,
            use_mamba,
        )

        self.dec5 = DecoderBlock3D(dims[4], dims[3], dims[3])
        self.dec4 = DecoderBlock3D(dims[3], dims[2], dims[2])
        self.dec3 = DecoderBlock3D(dims[2], dims[1], dims[1])
        self.dec2 = DecoderBlock3D(dims[1], dims[0], dims[0])
        self.dec1 = ResidualConvBlock3D(dims[0], dims[0], res_block=res_block)
        self.bridge = ConvNormAct3D(dims[0], dims[0]) if bridge else nn.Identity()
        self.head = nn.Conv3d(dims[0], out_channels, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "MambaVesselNet expects (B,C,D,H,W); "
                f"received {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            mode = "Salience" if self.use_salience else "baseline"
            raise ValueError(
                f"{mode} MambaVesselNet expects {self.in_channels} input channels, "
                f"received {x.shape[1]}. For Salience use [G_hat_c,G_hat_s,G_hat_t]."
            )

        original_size = x.shape[2:]
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.down1(enc1))
        enc3 = self.enc3(self.down2(enc2))
        enc4 = self.enc4(self.down3(enc3))
        enc5 = self.enc5(self.down4(enc4))

        x = self.enc_mamba(enc5)
        x = self.dec_mamba(x)
        x = self.dec5(x, enc4)
        x = self.dec4(x, enc3)
        x = self.dec3(x, enc2)
        x = self.dec2(x, enc1)
        x = self.bridge(self.dec1(x))
        logits = self.head(x)

        if logits.shape[2:] != original_size:
            logits = F.interpolate(
                logits, size=original_size, mode="trilinear", align_corners=False
            )
        return logits


# Model-factory aliases used by different experiment configurations.
MambaVesselNet3D = MambaVesselNet
MambaVesselNetPP = MambaVesselNet
MambaVesselNetWithSalience = MambaVesselNet
mvnNet = MambaVesselNet
MVNNet = MambaVesselNet


def _smoke_test() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for channels, use_salience in ((1, False), (3, True)):
        model = MambaVesselNet(
            in_channels=channels,
            out_channels=1,
            base_channels=4,
            depths=(1, 1, 1, 1, 1),
            bottleneck_depth=1,
            decoder_depth=1,
            drop_path_rate=0.0,
            use_mamba=True,
            use_salience=use_salience,
        ).to(device).eval()
        with torch.no_grad():
            output = model(torch.randn(1, channels, 16, 32, 32, device=device))
        assert output.shape == (1, 1, 16, 32, 32), output.shape
        params = sum(parameter.numel() for parameter in model.parameters())
        backend = "external-mamba" if any(
            getattr(module, "using_external_mamba", False)
            for module in model.modules()
            if isinstance(module, MambaTokenMixer)
        ) else "torch-fallback"
        print(
            f"channels={channels}, salience={use_salience}: "
            f"output={tuple(output.shape)}, params={params:,}, backend={backend}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MambaVesselNet smoke tests")
    parser.add_argument("--smoke-test", action="store_true", default=True)
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
