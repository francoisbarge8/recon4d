"""Point clouds from depth maps: back-projection, filtering and down-sampling."""

from __future__ import annotations

import torch
from torch import Tensor

from recon4d.geometry.camera import (
    backproject_depth,
    in_image,
    invert_se3,
    project,
    sample_bilinear,
    transform_points,
)


def backproject_frames(
    depth: Tensor,
    K: Tensor,
    c2w: Tensor,
    images: Tensor | None = None,
    mask: Tensor | None = None,
    stride: int = 1,
) -> tuple[Tensor, Tensor | None]:
    """Lift depth maps to one world-space point cloud.

    Args:
        depth: ``(F, H, W)`` z-depth maps.
        K: ``(3, 3)`` intrinsics.
        c2w: ``(F, 4, 4)`` camera-to-world poses.
        images: optional ``(F, H, W, 3)`` colours.
        mask: optional ``(F, H, W)`` selection of pixels to keep.
        stride: sub-sampling of the pixel grid.

    Returns:
        ``points (M, 3)`` and ``colors (M, 3)`` (None without ``images``).
    """
    offset = stride // 2
    points, colors = [], []
    for f in range(depth.shape[0]):
        world = backproject_depth(K, c2w[f], depth[f])[offset::stride, offset::stride]
        keep = depth[f][offset::stride, offset::stride] > 0
        if mask is not None:
            keep = keep & mask[f][offset::stride, offset::stride]
        points.append(world[keep])
        if images is not None:
            colors.append(images[f][offset::stride, offset::stride][keep])
    return torch.cat(points), (torch.cat(colors) if images is not None else None)


def voxel_downsample(
    points: Tensor, voxel_size: float, colors: Tensor | None = None
) -> tuple[Tensor, Tensor | None]:
    """Replace all points falling in the same cubic voxel by their centroid."""
    if points.shape[0] == 0:
        return points, colors
    keys = torch.floor(points / voxel_size).to(torch.int64)
    _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True, return_counts=True)
    n = counts.shape[0]
    weight = counts.to(points.dtype)[:, None]
    centroids = torch.zeros(n, 3, dtype=points.dtype).index_add_(0, inverse, points) / weight
    if colors is None:
        return centroids, None
    mean_colors = torch.zeros(n, colors.shape[1], dtype=colors.dtype).index_add_(0, inverse, colors)
    return centroids, mean_colors / weight.to(colors.dtype)


def downsample_to(
    points: Tensor, target: int, colors: Tensor | None = None, iterations: int = 12
) -> tuple[Tensor, Tensor | None, float]:
    """Voxel down-sample to roughly ``target`` points (bisection on the voxel size).

    Returns the points, colours and the voxel size that was used.
    """
    if points.shape[0] <= target:
        return points, colors, 0.0
    extent = float((points.amax(dim=0) - points.amin(dim=0)).max())
    lo, hi = extent * 1e-4, extent
    best = (points, colors, 0.0)
    for _ in range(iterations):
        mid = (lo * hi) ** 0.5
        p, c = voxel_downsample(points, mid, colors)
        if p.shape[0] > target:
            lo = mid
        else:
            hi = mid
            best = (p, c, mid)
    return best


def multiview_consistency(
    depth: Tensor,
    K: Tensor,
    w2c: Tensor,
    frame: int,
    neighbors: list[int],
    rel_tolerance: float = 0.03,
    valid: Tensor | None = None,
) -> Tensor:
    """Count, for every pixel of ``frame``, the neighbouring views that agree on its depth.

    A pixel is lifted to 3D with its depth and projected into each neighbour; the views
    agree when the neighbour's depth map, sampled at that projection, matches the depth
    of the point in the neighbour's camera within ``rel_tolerance``. This is the geometric
    consistency filter of multi-view stereo; it removes depth outliers, occlusion-boundary
    artefacts and (in static reconstruction) moving objects.

    Args:
        depth: ``(T, H, W)`` depth maps expressed in a common metric frame.
        w2c: ``(T, 4, 4)``.
        valid: optional ``(T, H, W)`` mask of usable depths.

    Returns:
        ``(H, W)`` integer count of consistent neighbours.
    """
    height, width = depth.shape[-2:]
    world = backproject_depth(K, invert_se3(w2c[frame]), depth[frame]).reshape(-1, 3)
    count = torch.zeros(height * width, dtype=torch.int64)
    for other in neighbors:
        uv, z = project(K, transform_points(w2c[other], world))
        inside = (z > 1e-6) & in_image(uv, height, width)
        sampled = sample_bilinear(depth[other][None], uv, mode="nearest")[:, 0]
        agree = inside & ((sampled - z).abs() < rel_tolerance * z)
        if valid is not None:
            ok = sample_bilinear(valid[other][None].to(depth.dtype), uv, mode="nearest")[:, 0]
            agree = agree & (ok > 0.5)
        count += agree.to(torch.int64)
    return count.reshape(height, width)
