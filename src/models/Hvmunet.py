"""H-vmunet style model adapted for the existing 3D segmentation pipeline.

The official H-vmunet implementation is a 2D model.  This file keeps its
main ideas -- high-order feature mixing, four-direction spatial scanning and
the spatial/channel attention bridge -- while exposing a 3D-compatible API:

    input : (B, C, D, H, W)
    output: (B, num_classes, D, H, W) logits

The H-vmunet blocks operate on axial slices after a lightweight depth mixing
layer.  This makes the model usable by the current 3D trainer without the
CUDA-only ``mamba_ssm`` and ``causal_conv1d`` packages used by the official
repository.

Salience is intentionally kept outside this model, just as it is for
UNet3D.  Use ``in_channels=1`` for baseline and ``in_channels=3`` together
with the existing salience preprocessor for the Salience experiment.
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that divides ``channels``."""

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DropPath(nn.Module):
    """Stochastic depth without an external timm dependency."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1).")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class DirectionalMixer2D(nn.Module):
    """Efficient four-direction spatial mixer.

    It approximates the four scan directions of SS2D using shared depthwise
    horizontal/vertical filters in both directions.  The implementation is
    fully PyTorch based and works on CPU as well as CUDA.
    """

    def __init__(self, channels: int, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        padding = kernel_size // 2
        self.pre = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )
        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = F.silu(self.norm(self.pre(x)))

        left_to_right = self.horizontal(x)
        right_to_left = torch.flip(
            self.horizontal(torch.flip(x, dims=(-1,))), dims=(-1,)
        )
        top_to_bottom = self.vertical(x)
        bottom_to_top = torch.flip(
            self.vertical(torch.flip(x, dims=(-2,))), dims=(-2,)
        )
        local = self.local(x)

        scanned = (left_to_right + right_to_left + top_to_bottom + bottom_to_top) * 0.25
        scanned = scanned + local
        return self.out(scanned * torch.sigmoid(self.gate(x)))


class HOrderSS2D(nn.Module):
    """High-order gated feature mixer inspired by H-SS2D."""

    def __init__(
        self,
        channels: int,
        order: int = 4,
        scan_kernel: int = 7,
    ) -> None:
        super().__init__()
        if order < 1 or order > 5:
            raise ValueError("order must be between 1 and 5.")
        divisor = 2 ** (order - 1)
        if channels % divisor != 0:
            raise ValueError(
                f"channels={channels} must be divisible by {divisor} for order={order}."
            )

        dims = [channels // (2 ** i) for i in range(order)]
        dims.reverse()
        self.dims = dims

        # The official high-order split has one value branch and one gate
        # branch for every level.  Their total width is exactly 2*channels.
        self.proj_in = nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True)
        self.mixers = nn.ModuleList(
            DirectionalMixer2D(dim, kernel_size=scan_kernel) for dim in dims
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, bias=False)
            for i in range(order - 1)
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        values_and_gates = self.proj_in(x)
        split_sizes = [self.dims[0]] + self.dims
        chunks = torch.split(values_and_gates, split_sizes, dim=1)

        values = chunks[0]
        gates = chunks[1:]
        values = self.mixers[0](values * torch.sigmoid(gates[0]))

        for index, projection in enumerate(self.projections, start=1):
            values = projection(values)
            values = self.mixers[index](values * torch.sigmoid(gates[index]))

        return self.proj_out(values)


class HBlock(nn.Module):
    """Residual H-vmunet block with high-order spatial mixing and FFN."""

    def __init__(
        self,
        channels: int,
        order: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.spatial = HOrderSS2D(channels, order=order)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, 4 * channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(4 * channels, channels, kernel_size=1, bias=True),
        )
        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(channels))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(channels))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        gamma1 = self.gamma1.view(1, -1, 1, 1)
        gamma2 = self.gamma2.view(1, -1, 1, 1)
        x = x + self.drop_path(gamma1 * self.spatial(self.norm1(x)))
        x = x + self.drop_path(gamma2 * self.ffn(self.norm2(x)))
        return x


class HStage(nn.Module):
    def __init__(
        self,
        channels: int,
        order: int,
        depth: int,
        drop_paths: Iterable[float],
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        paths = list(drop_paths)
        if len(paths) != depth:
            raise ValueError("drop_paths length must equal depth.")
        self.blocks = nn.Sequential(
            *(
                HBlock(
                    channels=channels,
                    order=order,
                    drop_path=paths[i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(depth)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class SCBridge(nn.Module):
    """Spatial/channel attention bridge for the five encoder features."""

    def __init__(self, channels: Sequence[int], reduction: int = 4) -> None:
        super().__init__()
        self.channels = list(channels)
        total = sum(self.channels)
        hidden = max(total // reduction, 8)
        self.spatial = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                    nn.Sigmoid(),
                )
                for _ in self.channels
            ]
        )
        self.channel = nn.Sequential(
            nn.Linear(total, hidden),
            nn.GELU(),
            nn.Linear(hidden, total),
            nn.Sigmoid(),
        )

    def forward(self, features: Sequence[Tensor]) -> List[Tensor]:
        if len(features) != len(self.channels):
            raise ValueError("SCBridge received an unexpected number of features.")

        spatial_features: List[Tensor] = []
        pooled: List[Tensor] = []
        for feature, spatial_gate in zip(features, self.spatial):
            mean_map = feature.mean(dim=1, keepdim=True)
            max_map = feature.amax(dim=1, keepdim=True)
            gate = spatial_gate(torch.cat((mean_map, max_map), dim=1))
            refined = feature + feature * gate
            spatial_features.append(refined)
            pooled.append(F.adaptive_avg_pool2d(refined, output_size=1).flatten(1))

        channel_gate = self.channel(torch.cat(pooled, dim=1))
        split_gates = torch.split(channel_gate, self.channels, dim=1)
        return [
            feature + feature * gate[:, :, None, None]
            for feature, gate in zip(spatial_features, split_gates)
        ]


class UpStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        order: int,
        depth: int,
        drop_paths: Iterable[float],
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.fuse = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.stage = HStage(
            channels=out_channels,
            order=order,
            depth=depth,
            drop_paths=drop_paths,
            layer_scale_init=layer_scale_init,
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat((x, skip), dim=1)
        return self.stage(F.gelu(self.norm(self.fuse(x))))


def _make_channels(
    base_channels: int,
    c_list: Optional[Sequence[int]],
) -> List[int]:
    if c_list is None:
        c_list = [base_channels * (2 ** i) for i in range(5)]
    channels = [int(value) for value in c_list]
    if len(channels) != 5:
        raise ValueError("c_list must contain five channel widths.")
    if any(value <= 0 for value in channels):
        raise ValueError("All channel widths must be positive.")
    return channels


class HVMUNet3D(nn.Module):
    """3D-compatible H-vmunet for voxel-wise binary/multi-class segmentation.

    Parameters use aliases found in common UNet configuration files.  The
    model returns logits, matching the usual ``BCEWithLogitsLoss`` and the
    existing evaluation code that applies ``torch.sigmoid``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        input_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        n_classes: Optional[int] = None,
        base_channels: int = 16,
        c_list: Optional[Sequence[int]] = None,
        depths: Optional[Sequence[int]] = None,
        order: Optional[Sequence[int]] = None,
        drop_path_rate: float = 0.0,
        layer_scale_init: float = 1e-6,
        bridge: bool = True,
        **_: object,
    ) -> None:
        super().__init__()

        if input_channels is not None:
            in_channels = int(input_channels)
        if out_channels is not None:
            num_classes = int(out_channels)
        if n_classes is not None:
            num_classes = int(n_classes)
        if in_channels <= 0 or num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive.")

        channels = _make_channels(base_channels, c_list)
        depths = [1, 1, 1, 1, 1] if depths is None else [int(v) for v in depths]
        if len(depths) != 5 or any(depth <= 0 for depth in depths):
            raise ValueError("depths must contain five positive integers.")
        orders = [1, 2, 3, 4, 5] if order is None else [int(v) for v in order]
        if len(orders) != 5:
            raise ValueError("order must contain five integers.")

        for stage_channels, stage_order in zip(channels, orders):
            divisor = 2 ** (stage_order - 1)
            if stage_order < 1 or stage_order > 5 or stage_channels % divisor != 0:
                raise ValueError(
                    f"Invalid stage configuration: channels={stage_channels}, "
                    f"order={stage_order}."
                )

        total_blocks = sum(depths) + sum(depths[:-1])
        drop_paths = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        encoder_paths: List[List[float]] = []
        offset = 0
        for depth in depths:
            encoder_paths.append(drop_paths[offset : offset + depth])
            offset += depth
        decoder_paths: List[List[float]] = []
        for depth in reversed(depths[:-1]):
            decoder_paths.append(drop_paths[offset : offset + depth])
            offset += depth

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.pad_multiple = 16

        # Lightweight cross-slice context before the 2D H-vmunet stages.
        self.depth_mixer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(_group_count(in_channels), in_channels),
            nn.SiLU(),
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.GELU(),
        )
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index, (stage_channels, stage_order, depth) in enumerate(
            zip(channels, orders, depths)
        ):
            self.encoders.append(
                HStage(
                    channels=stage_channels,
                    order=stage_order,
                    depth=depth,
                    drop_paths=encoder_paths[index],
                    layer_scale_init=layer_scale_init,
                )
            )
            if index < len(channels) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(
                            stage_channels,
                            channels[index + 1],
                            kernel_size=2,
                            stride=2,
                            bias=False,
                        ),
                        nn.GroupNorm(
                            _group_count(channels[index + 1]), channels[index + 1]
                        ),
                        nn.GELU(),
                    )
                )

        self.bridge = SCBridge(channels) if bridge else nn.Identity()
        self.decoders = nn.ModuleList()
        reversed_skips = list(reversed(channels[:-1]))
        reversed_orders = list(reversed(orders[:-1]))
        reversed_depths = list(reversed(depths[:-1]))
        for index, (skip_channels, stage_order, depth) in enumerate(
            zip(reversed_skips, reversed_orders, reversed_depths)
        ):
            in_ch = channels[-1] if index == 0 else reversed_skips[index - 1]
            self.decoders.append(
                UpStage(
                    in_channels=in_ch,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                    order=stage_order,
                    depth=depth,
                    drop_paths=decoder_paths[index],
                    layer_scale_init=layer_scale_init,
                )
            )

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.depth_refine = nn.Conv3d(
            num_classes,
            num_classes,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            groups=num_classes,
            bias=False,
        )
        self._init_weights()
        # Start as an identity residual so the extra depth refinement does
        # not destabilize the first training iterations.
        nn.init.zeros_(self.depth_refine.weight)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _pad_spatial(self, x: Tensor) -> Tuple[Tensor, int, int]:
        height, width = x.shape[-2:]
        pad_h = (self.pad_multiple - height % self.pad_multiple) % self.pad_multiple
        pad_w = (self.pad_multiple - width % self.pad_multiple) % self.pad_multiple
        if pad_h == 0 and pad_w == 0:
            return x, 0, 0
        return F.pad(x, (0, pad_w, 0, pad_h)), pad_h, pad_w

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(
                "HVMUNet3D expects input with shape (B, C, D, H, W), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )

        batch, _, depth, height, width = x.shape
        x = self.depth_mixer(x)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, self.in_channels, height, width)
        x, pad_h, pad_w = self._pad_spatial(x)

        feature = self.stem(x)
        skips: List[Tensor] = []
        for index, encoder in enumerate(self.encoders):
            feature = encoder(feature)
            skips.append(feature)
            if index < len(self.downsamples):
                feature = self.downsamples[index](feature)

        if isinstance(self.bridge, SCBridge):
            skips = self.bridge(skips)

        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            feature = decoder(feature, skip)

        logits = self.head(feature)
        if pad_h or pad_w:
            logits = logits[..., :height, :width]

        logits = logits.reshape(batch, depth, self.num_classes, height, width)
        logits = logits.permute(0, 2, 1, 3, 4).contiguous()
        return logits + self.depth_refine(logits)


# Aliases cover the naming conventions commonly used by model builders.
HVMUNet = HVMUNet3D
Hvmunet = HVMUNet3D
H_vmunet = HVMUNet3D


def build_hvmunet(config: Optional[dict] = None, **kwargs: object) -> HVMUNet3D:
    """Build the model from either a model-config dictionary or kwargs."""

    params = {}
    if config is not None:
        params.update(dict(config))
    params.update(kwargs)
    params.pop("name", None)
    return HVMUNet3D(**params)


def _smoke_test() -> None:
    model = HVMUNet3D(
        in_channels=3,
        num_classes=1,
        base_channels=8,
        depths=[1, 1, 1, 1, 1],
        drop_path_rate=0.05,
    )
    x = torch.randn(1, 3, 5, 65, 67)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 5, 65, 67), y.shape
    print(f"HVMUNet3D smoke test passed: input={tuple(x.shape)}, output={tuple(y.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a small HVMUNet3D smoke test.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the shape test.")
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        parser.print_help()
