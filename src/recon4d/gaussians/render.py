"""Colour / depth / feature rendering on top of the rasterizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.gaussians.rasterizer import Projection, RasterSettings, project_gaussians, rasterize
from recon4d.geometry.camera import camera_centers


@dataclass(frozen=True)
class Camera:
    """A pinhole view: intrinsics ``K (3, 3)``, pose ``w2c (4, 4)`` and image size."""

    K: Tensor
    w2c: Tensor
    width: int
    height: int

    @property
    def center(self) -> Tensor:
        return camera_centers(self.w2c)


@dataclass
class RenderOutput:
    color: Tensor
    """``(H, W, 3)`` image composited over the background."""
    depth: Tensor
    """``(H, W)`` expected z-depth of the composited surface (0 where nothing was hit)."""
    alpha: Tensor
    """``(H, W)`` accumulated opacity."""
    extras: Tensor | None
    """``(H, W, E)`` alpha-composited extra features, when requested."""
    projection: Projection
    """Screen-space Gaussians; ``projection.means2d.grad`` drives densification."""


def render_gaussians(
    means: Tensor,
    rotations: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    camera: Camera,
    settings: RasterSettings | None = None,
    background: Tensor | None = None,
    extras: Tensor | None = None,
) -> RenderOutput:
    """Render Gaussians to colour, depth and optional per-Gaussian features.

    Depth and extras are rendered in the same pass as colour by compositing extra feature
    channels. Depth is normalised by the accumulated opacity, i.e. it is the expected depth
    of the surface actually hit rather than a blend with an arbitrary background depth.

    Args:
        rotations: ``(N, 4)`` quaternions or ``(N, 3, 3)`` matrices.
        colors: ``(N, 3)`` RGB.
        background: ``(3,)`` colour placed behind the Gaussians (default black).
        extras: ``(N, E)`` additional features to composite.
    """
    settings = settings or RasterSettings()
    projection = project_gaussians(
        means,
        rotations,
        scales,
        camera.K,
        camera.w2c,
        camera.width,
        camera.height,
        settings,
        opacities,
    )
    if projection.means2d.requires_grad:
        projection.means2d.retain_grad()
    features = [colors, projection.depths[:, None]]
    if extras is not None:
        features.append(extras)
    out = rasterize(
        projection, opacities, torch.cat(features, dim=1), camera.width, camera.height, settings
    )
    color = out.image[..., :3]
    if background is not None:
        color = color + (1.0 - out.alpha)[..., None] * background
    hit = out.alpha > 1e-4
    depth = torch.where(
        hit, out.image[..., 3] / out.alpha.clamp_min(1e-4), torch.zeros_like(out.alpha)
    )
    return RenderOutput(
        color=color,
        depth=depth,
        alpha=out.alpha,
        extras=out.image[..., 4:] if extras is not None else None,
        projection=projection,
    )
