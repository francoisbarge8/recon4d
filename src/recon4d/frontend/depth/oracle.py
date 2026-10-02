"""Ground-truth depth with a controllable error model.

This is an analysis tool, not an estimator: it lets experiments dial the quality of the
depth prior from perfect to realistic and measure how the rest of the pipeline degrades.
The error model reproduces the failure modes of per-frame monocular networks:

* an unknown, *per-frame* scale (scale ambiguity and temporal flicker);
* low-frequency multiplicative errors (the layout is right locally, wrong globally);
* blurred depth discontinuities;
* a little pixel noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from recon4d.frontend.depth.base import DepthEstimator


@dataclass(frozen=True)
class DepthNoise:
    """Error model applied to the ground-truth depth.

    Attributes:
        scale_jitter: standard deviation of the per-frame log-scale.
        smooth: standard deviation of the low-frequency multiplicative log-error.
        smooth_cells: number of cells of the low-frequency error along the image height.
        static_fraction: share of the low-frequency error that is identical in all frames
            (a systematic bias) rather than re-drawn for each frame (flicker).
        blur: standard deviation (pixels) of the Gaussian blur applied to the inverse
            depth, which smears occlusion boundaries like a network does.
        pixel: standard deviation of the per-pixel relative noise.
        seed: random seed.
    """

    scale_jitter: float = 0.2
    smooth: float = 0.05
    smooth_cells: int = 3
    static_fraction: float = 0.5
    blur: float = 1.0
    pixel: float = 0.005
    seed: int = 0


def _gaussian_blur(maps: Tensor, sigma: float) -> Tensor:
    radius = max(1, round(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=maps.dtype)
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    x = maps[:, None]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel[None, None, None, :])
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel[None, None, :, None])
    return x[:, 0]


class OracleDepth(DepthEstimator):
    """Returns the ground-truth depth, optionally degraded by a :class:`DepthNoise`."""

    name = "oracle"
    kind = "scale"

    def __init__(self, depth: Tensor, noise: DepthNoise | None = None) -> None:
        self.depth = depth
        self.noise = noise
        if noise is not None:
            self.name = "oracle+noise"

    def predict(self, images: Tensor) -> Tensor:
        if images.shape[:3] != self.depth.shape:
            raise ValueError("the oracle depth does not match the video")
        noise = self.noise
        if noise is None:
            return self.depth.clone()
        generator = torch.Generator().manual_seed(noise.seed)
        n_frames, height, width = self.depth.shape
        depth = self.depth
        if noise.blur > 0:
            depth = 1.0 / _gaussian_blur(1.0 / depth, noise.blur)
        log_error = torch.zeros_like(depth)
        if noise.smooth > 0:
            rows = noise.smooth_cells + 1
            cols = max(2, round(noise.smooth_cells * width / height) + 1)
            static = torch.randn(1, 1, rows, cols, generator=generator)
            varying = torch.randn(n_frames, 1, rows, cols, generator=generator)
            mix = noise.static_fraction
            field = noise.smooth * (mix**0.5 * static + (1.0 - mix) ** 0.5 * varying)
            log_error = F.interpolate(
                field, size=(height, width), mode="bicubic", align_corners=True
            )[:, 0]
        if noise.scale_jitter > 0:
            log_error = log_error + noise.scale_jitter * torch.randn(
                n_frames, 1, 1, generator=generator
            )
        if noise.pixel > 0:
            log_error = log_error + noise.pixel * torch.randn(depth.shape, generator=generator)
        return depth * torch.exp(log_error)
