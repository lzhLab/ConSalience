#!/usr/bin/env python3
"""Self-contained dynamic 3-D Swin-UNet for medical image segmentation.

The public constructor is compatible with the model factory used by the
existing H-vmunet experiments::

    model = DSwinUNet(in_channels=1, out_channels=1, base_channels=8)
    logits = model(volume)  # (B, C, D, H, W) -> (B, out_channels, D, H, W)

Only PyTorch is required.  The evaluator should keep responsibility for
boundary-aware sliding-window inference, overlap blending, and restoration of
dataset-specific preprocessing.
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Int3 = Tuple[int, int, int]


def _triple(value: Union[int, Sequence[int]]) -> Int3:
    if isinstance(value, int):
        return value, value, value
    value = tuple(int(v) for v in value)
    if len(value) != 3:
        raise ValueError("A 3-D window size must contain exactly three values")
    return value  # type: ignore[return-value]


def _valid_groups(channels: int, requested: int = 8) -> int:
    """Return a GroupNorm group count that divides ``channels``."""
    groups = min(requested, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


def _norm3d(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_valid_groups(channels), channels)


def _safe_avg_pool3d(x: torch.Tensor) -> torch.Tensor:
    """Apply a same-size 3-D average pool, including singleton dimensions.

    ``avg_pool3d(..., kernel_size=3, padding=1)`` still rejects an input whose
    spatial extent is smaller than three.  Deep encoder stages can legitimately
    reach such shapes for small training patches, so replicate-padding is used
    only in those cases and the result is cropped back to the original size.
    """
    depth, height, width = x.shape[-3:]
    pad_d = 1 if depth < 3 else 0
    pad_h = 1 if height < 3 else 0
    pad_w = 1 if width < 3 else 0
    if pad_d or pad_h or pad_w:
        x = F.pad(
            x,
            (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d),
            mode="replicate",
        )
        pooled = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        return pooled[..., pad_d : pad_d + depth, pad_h : pad_h + height, pad_w : pad_w + width]
    return F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)


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
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


def window_partition(x: torch.Tensor, window_size: Int3) -> torch.Tensor:
    """Partition a channels-last tensor ``(B,D,H,W,C)`` into windows."""
    wd, wh, ww = window_size
    b, d, h, w, c = x.shape
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wd * wh * ww, c)


def window_reverse(
    windows: torch.Tensor,
    window_size: Int3,
    batch_size: int,
    depth: int,
    height: int,
    width: int,
) -> torch.Tensor:
    wd, wh, ww = window_size
    channels = windows.shape[-1]
    x = windows.view(
        batch_size,
        depth // wd,
        height // wh,
        width // ww,
        wd,
        wh,
        ww,
        channels,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(batch_size, depth, height, width, channels)


def _relative_position_index(window_size: Int3) -> torch.Tensor:
    wd, wh, ww = window_size
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww), indexing="ij"
        )
    )
    coords = coords.flatten(1)
    relative = coords[:, :, None] - coords[:, None, :]
    relative = relative.permute(1, 2, 0).contiguous()
    relative[:, :, 0] += wd - 1
    relative[:, :, 1] += wh - 1
    relative[:, :, 2] += ww - 1
    relative[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
    relative[:, :, 1] *= 2 * ww - 1
    return relative.sum(-1)


class WindowAttention3D(nn.Module):
    """Window multi-head self-attention with 3-D relative position bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Int3,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads; got "
                f"dim={dim}, num_heads={num_heads}"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        table_size = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        self.register_buffer(
            "relative_position_index",
            _relative_position_index(window_size),
            persistent=False,
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)

        if mask is not None:
            windows_per_sample = mask.shape[0]
            batch_size = batch_windows // windows_per_sample
            attention = attention.view(batch_size, windows_per_sample, self.num_heads, tokens, tokens)
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(batch_windows, self.num_heads, tokens, tokens)

        attention = self.attn_drop(attention.softmax(dim=-1))
        x = (attention @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj_drop(self.proj(x))


def _attention_mask(
    depth: int,
    height: int,
    width: int,
    window_size: Int3,
    shift_size: Int3,
    device: torch.device,
) -> torch.Tensor:
    """Build the standard shifted-window attention mask."""
    wd, wh, ww = window_size
    sd, sh, sw = shift_size
    mask = torch.zeros((1, depth, height, width, 1), device=device)
    d_slices = ((0, -wd), (-wd, -sd), (-sd, None)) if sd else ((0, None),)
    h_slices = ((0, -wh), (-wh, -sh), (-sh, None)) if sh else ((0, None),)
    w_slices = ((0, -ww), (-ww, -sw), (-sw, None)) if sw else ((0, None),)
    counter = 0
    for ds in d_slices:
        for hs in h_slices:
            for ws in w_slices:
                mask[:, ds[0] : ds[1], hs[0] : hs[1], ws[0] : ws[1], :] = counter
                counter += 1
    mask_windows = window_partition(mask, window_size).squeeze(-1)
    attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attention_mask.masked_fill(attention_mask != 0, -100.0).masked_fill(attention_mask == 0, 0.0)


class DynamicSwinBlock3D(nn.Module):
    """Swin block that switches between large and small windows per volume."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Int3,
        small_window_size: Int3,
        shift: bool,
        dropout: float,
        drop_path: float,
        complexity_threshold: float = 0.18,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.small_window_size = small_window_size
        self.shift = shift
        self.complexity_threshold = complexity_threshold
        self.norm1 = nn.LayerNorm(dim)
        self.large_attention = WindowAttention3D(dim, num_heads, window_size, dropout=dropout)
        self.small_attention = WindowAttention3D(dim, num_heads, small_window_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dropout=dropout)
        self.drop_path = DropPath(drop_path)

    @staticmethod
    def _pad_to_window(x: torch.Tensor, window_size: Int3) -> Tuple[torch.Tensor, Int3]:
        _, depth, height, width, _ = x.shape
        pad_d = (window_size[0] - depth % window_size[0]) % window_size[0]
        pad_h = (window_size[1] - height % window_size[1]) % window_size[1]
        pad_w = (window_size[2] - width % window_size[2]) % window_size[2]
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        return x, (depth, height, width)

    def _run_attention(self, x: torch.Tensor, window_size: Int3, shifted: bool) -> torch.Tensor:
        x, original_shape = self._pad_to_window(x, window_size)
        _, depth, height, width, _ = x.shape
        shifts = tuple(size // 2 for size in window_size) if shifted else (0, 0, 0)
        if any(shifts):
            x = torch.roll(x, shifts=tuple(-s for s in shifts), dims=(1, 2, 3))
            mask = _attention_mask(depth, height, width, window_size, shifts, x.device)
        else:
            mask = None
        windows = window_partition(x, window_size)
        if window_size == self.window_size:
            windows = self.large_attention(windows, mask)
        else:
            windows = self.small_attention(windows, mask)
        x = window_reverse(windows, window_size, x.shape[0], depth, height, width)
        if any(shifts):
            x = torch.roll(x, shifts=shifts, dims=(1, 2, 3))
        original_depth, original_height, original_width = original_shape
        return x[:, :original_depth, :original_height, :original_width, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        normalized = self.norm1(x)
        # Local variation is measured before attention.  It provides an
        # input-adaptive proxy for anatomical complexity without using labels.
        normalized_ncdhw = normalized.detach().float().permute(0, 4, 1, 2, 3)
        local = _safe_avg_pool3d(normalized_ncdhw)
        variation = (normalized_ncdhw - local).abs().flatten(1).mean(dim=1)
        magnitude = normalized_ncdhw.abs().flatten(1).mean(dim=1)
        complexity = (variation / (magnitude + 1e-6)).mean().item()
        window = self.small_window_size if complexity > self.complexity_threshold else self.window_size
        attended = self._run_attention(normalized, window, self.shift)
        x = shortcut + self.drop_path(attended)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging3D(nn.Module):
    def __init__(self, dim: int, out_dim: Optional[int] = None) -> None:
        super().__init__()
        out_dim = out_dim or dim * 2
        self.norm = nn.LayerNorm(dim * 8)
        self.reduction = nn.Linear(dim * 8, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, depth, height, width, channels = x.shape
        pad_d, pad_h, pad_w = depth % 2, height % 2, width % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
            depth, height, width = x.shape[1:4]
        x = x.view(batch, depth // 2, 2, height // 2, 2, width // 2, 2, channels)
        x = torch.cat(
            [
                x[:, :, 0, :, 0, :, 0], x[:, :, 1, :, 0, :, 0],
                x[:, :, 0, :, 1, :, 0], x[:, :, 1, :, 1, :, 0],
                x[:, :, 0, :, 0, :, 1], x[:, :, 1, :, 0, :, 1],
                x[:, :, 0, :, 1, :, 1], x[:, :, 1, :, 1, :, 1],
            ],
            dim=-1,
        )
        return self.reduction(self.norm(x))


class DynamicSwinStage3D(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Int3,
        small_window_size: Int3,
        dropout: float,
        drop_path_rates: Sequence[float],
        dynamic_depth: bool = True,
    ) -> None:
        super().__init__()
        self.dynamic_depth = dynamic_depth and depth > 1
        self.blocks = nn.ModuleList(
            [
                DynamicSwinBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    small_window_size=small_window_size,
                    shift=bool(index % 2),
                    dropout=dropout,
                    drop_path=drop_path_rates[index],
                )
                for index in range(depth)
            ]
        )
        if self.dynamic_depth:
            hidden = max(16, dim // 2)
            self.depth_gate = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, depth),
            )
        else:
            self.depth_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depth_gate is not None:
            pooled = x.mean(dim=(1, 2, 3))
            gates = torch.sigmoid(self.depth_gate(pooled))
        else:
            gates = None
        for index, block in enumerate(self.blocks):
            updated = block(x)
            if gates is None:
                x = updated
            else:
                gate = gates[:, index].view(x.shape[0], 1, 1, 1, 1)
                x = x + gate * (updated - x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 4, 1).contiguous()
        return self.norm(x)


class BoundaryAttention3D(nn.Module):
    """Lightweight boundary gate based on local feature discontinuity."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.edge = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.project = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            _norm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = _safe_avg_pool3d(x)
        edge = self.edge(x - local)
        boundary = torch.sigmoid(self.project(torch.cat([x, edge], dim=1)))
        return x * (1.0 + boundary)


class DecoderFusion3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.boundary = BoundaryAttention3D(skip_channels)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            _norm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _norm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        skip = self.boundary(skip)
        return self.fuse(torch.cat([x, skip], dim=1))


class MultiResolutionFusion3D(nn.Module):
    """Fuse the final half-resolution decoder map with the stem feature."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 3, padding=1, bias=False),
            _norm3d(channels),
            nn.GELU(),
        )

    def forward(self, decoder: torch.Tensor, stem: torch.Tensor) -> torch.Tensor:
        if decoder.shape[2:] != stem.shape[2:]:
            decoder = F.interpolate(decoder, size=stem.shape[2:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat([decoder, stem], dim=1))


class DSwinUNet(nn.Module):
    """Dynamic 3-D Swin-UNet.

    Extra keyword arguments are accepted for configuration compatibility with
    the previous ``Hvmunet`` registration.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 8,
        c_list: Optional[Sequence[int]] = None,
        depths: Optional[Sequence[int]] = None,
        num_heads: Optional[Sequence[int]] = None,
        window_size: Union[int, Sequence[int]] = (2, 4, 4),
        small_window_size: Optional[Union[int, Sequence[int]]] = None,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        dynamic_depth: bool = True,
        bridge: bool = True,
        layer_scale_init: Optional[float] = None,
        order: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__()
        del layer_scale_init, order
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")

        channels = tuple(c_list or (base_channels, base_channels * 2, base_channels * 4, base_channels * 8))
        if len(channels) != 4:
            raise ValueError("c_list must contain four channel widths")
        depths = tuple(depths or (2, 2, 2, 2))
        if len(depths) != 4:
            raise ValueError("depths must contain four stage depths")

        windows = _triple(window_size)
        small_windows = _triple(small_window_size or tuple(max(1, value // 2) for value in windows))
        if num_heads is None:
            heads = tuple(max(1, channels_i // 8) for channels_i in channels)
        else:
            heads = tuple(int(value) for value in num_heads)
        if len(heads) != 4:
            raise ValueError("num_heads must contain four values")
        for dim, head in zip(channels, heads):
            if head < 1 or dim % head != 0:
                raise ValueError(f"channel width {dim} is not divisible by num_heads {head}")

        total_blocks = sum(int(value) for value in depths)
        if total_blocks < 1:
            raise ValueError("At least one transformer block is required")
        rates = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        rate_offset = 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.depths = depths
        self.patch_embed = PatchEmbed3D(in_channels, channels[0])
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for index in range(4):
            stage_depth = int(depths[index])
            stage_rates = rates[rate_offset : rate_offset + stage_depth]
            rate_offset += stage_depth
            self.stages.append(
                DynamicSwinStage3D(
                    dim=channels[index],
                    depth=stage_depth,
                    num_heads=heads[index],
                    window_size=windows,
                    small_window_size=small_windows,
                    dropout=dropout,
                    drop_path_rates=stage_rates,
                    dynamic_depth=dynamic_depth,
                )
            )
            if index < 3:
                self.downsamples.append(PatchMerging3D(channels[index], channels[index + 1]))

        self.up3 = DecoderFusion3D(channels[3], channels[2], channels[2])
        self.up2 = DecoderFusion3D(channels[2], channels[1], channels[1])
        self.up1 = DecoderFusion3D(channels[1], channels[0], channels[0])
        self.multi_resolution = MultiResolutionFusion3D(channels[0])
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(channels[0], channels[0], 2, stride=2),
            _norm3d(channels[0]),
            nn.GELU(),
        )
        self.boundary_bridge = BoundaryAttention3D(channels[0]) if bridge else nn.Identity()
        self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "DSwinUNet expects a 5-D tensor (B,C,D,H,W); "
                f"received shape {tuple(x.shape)}"
            )

        original_size = x.shape[2:]

        # Transformer encoder: (B, D, H, W, C)
        stem = self.patch_embed(x)

        features = []
        current = stem

        for index, stage in enumerate(self.stages):
            current = stage(current)
            features.append(current)

            if index < 3:
                current = self.downsamples[index](current)

        # CNN decoder requires: (B, C, D, H, W)
        features_cf = [
            feature.permute(0, 4, 1, 2, 3).contiguous()
            for feature in features
        ]
        stem_cf = stem.permute(0, 4, 1, 2, 3).contiguous()

        decoded = self.up3(features_cf[3], features_cf[2])
        decoded = self.up2(decoded, features_cf[1])
        decoded = self.up1(decoded, features_cf[0])

        decoded = self.multi_resolution(decoded, stem_cf)
        decoded = self.final_up(decoded)
        decoded = self.boundary_bridge(decoded)

        logits = self.head(decoded)

        if logits.shape[2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="trilinear",
                align_corners=False,
            )

        return logits


# Common model-factory aliases.
DSwinUNet3D = DSwinUNet
DynamicSwinUNet = DSwinUNet
D_SwinUNet = DSwinUNet


def _smoke_test() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for channels in (1, 3):
        model = DSwinUNet(
            in_channels=channels,
            out_channels=1,
            base_channels=4,
            c_list=(4, 8, 16, 32),
            depths=(1, 1, 1, 1),
            drop_path_rate=0.0,
        ).to(device).eval()
        with torch.no_grad():
            output = model(torch.randn(1, channels, 16, 32, 32, device=device))
        assert output.shape == (1, 1, 16, 32, 32), output.shape
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"in_channels={channels}: output={tuple(output.shape)}, params={parameters:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DSwinUNet 3-D smoke test")
    parser.add_argument("--smoke-test", action="store_true", default=True)
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()

