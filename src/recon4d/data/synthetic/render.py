"""Ray-traced rendering of a :class:`~recon4d.data.synthetic.scene.Scene`."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.data.synthetic.primitives import DTYPE
from recon4d.data.synthetic.scene import Scene
from recon4d.geometry.camera import pixel_centers, pixel_rays


@dataclass(frozen=True)
class RenderedFrame:
    """One ray-traced frame with its exact per-pixel ground truth."""

    rgb: Tensor
    """``(H, W, 3)`` float32 in ``[0, 1]``, box-filtered over the sub-pixel samples."""
    depth: Tensor
    """``(H, W)`` float32 z-depth of the surface seen through the pixel centre."""
    obj: Tensor
    """``(H, W)`` int64 object index at the pixel centre."""
    local: Tensor
    """``(H, W, 3)`` float64 object-local coordinates of the pixel-centre surface point."""


def render_frame(
    scene: Scene, K: Tensor, c2w: Tensor, t: float, height: int, width: int, spp: int = 3
) -> RenderedFrame:
    """Render ``scene`` at time ``t`` from the camera ``(K, c2w)``.

    Colour is supersampled on an ``spp x spp`` grid per pixel. Geometry buffers are taken
    at the pixel centre only: averaging depth across an occlusion boundary would invent
    surfaces that do not exist.
    """
    if spp < 1:
        raise ValueError("spp must be >= 1")
    K = K.to(DTYPE)
    c2w = c2w.to(DTYPE)
    n_pixels = height * width
    centers = pixel_centers(height, width, dtype=DTYPE).reshape(-1, 2)

    # Sample 0 is the pixel centre (geometry buffers), followed by the rest of the
    # supersampling grid. All rays are traced in one batch: per-call overhead dominates
    # on small tensors.
    sub = (torch.arange(spp, dtype=DTYPE) + 0.5) / spp - 0.5
    grid = torch.stack(torch.meshgrid(sub, sub, indexing="xy"), dim=-1).reshape(-1, 2)
    is_center = (grid == 0).all(dim=1)
    offsets = torch.cat([torch.zeros(1, 2, dtype=DTYPE), grid[~is_center]])
    uv = (centers[None] + offsets[:, None]).reshape(-1, 2)

    origins, dirs = pixel_rays(K, c2w, uv)
    hits = scene.intersect(origins, dirs, t)
    colors = scene.shade(hits, origins, dirs, t).reshape(-1, n_pixels, 3)
    # With an even grid the centre is not a grid sample: use it for geometry only.
    rgb = colors.mean(dim=0) if is_center.any() else colors[1:].mean(dim=0)

    return RenderedFrame(
        rgb=rgb.reshape(height, width, 3),
        depth=hits.t[:n_pixels].to(torch.float32).reshape(height, width),
        obj=hits.obj[:n_pixels].reshape(height, width),
        local=hits.local[:n_pixels].reshape(height, width, 3),
    )
