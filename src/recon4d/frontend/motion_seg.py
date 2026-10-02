"""Motion segmentation: which pixels and tracks belong to independently moving objects?

No semantic model is involved. Once the camera poses and a depth map are known, the image
motion of every *static* pixel is predicted by geometry alone (the "rigid flow"). Pixels
whose observed optical flow disagrees with that prediction are moving on their own.

Tracks are labelled with the same principle, at track level: a static track is a single
3D point re-observed over time, so it must have a small reprojection error for the
estimated cameras; a track that cannot be explained by any static point moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor

from recon4d.frontend.flow import FlowEstimator, forward_backward_error
from recon4d.geometry.camera import (
    in_image,
    invert_se3,
    pixel_centers,
    project,
    transform_points,
    unproject,
)
from recon4d.types import Tracks


@dataclass
class MotionSegConfig:
    """Parameters of motion segmentation.

    Attributes:
        steps: frame offsets at which observed and rigid flow are compared; each pixel
            takes the median residual over the available offsets (forward and backward).
        residual_threshold: flow residual (pixels) above which a pixel is moving.
        relative_threshold: additional margin, as a fraction of the rigid flow magnitude,
            tolerating the residual caused by depth errors (which scales with parallax).
        fb_threshold: maximum forward-backward flow error (pixels) of a usable flow vector.
        min_area: connected components smaller than this (pixels) are discarded.
        morph_radius: radius of the morphological closing / opening used to clean masks.
        track_threshold: mean reprojection error (pixels) above which a track is dynamic.
        track_mask_fraction: a track is dynamic when it lies inside the dynamic mask for
            more than this fraction of its visible frames.
    """

    steps: tuple[int, ...] = (2, 4)
    residual_threshold: float = 1.0
    relative_threshold: float = 0.15
    fb_threshold: float = 1.0
    min_area: int = 30
    morph_radius: int = 2
    track_threshold: float = 3.0
    track_mask_fraction: float = 0.5


def rigid_flow(depth: Tensor, K: Tensor, w2c_src: Tensor, w2c_dst: Tensor) -> tuple[Tensor, Tensor]:
    """Flow induced on a static scene by the camera motion from ``src`` to ``dst``.

    Args:
        depth: ``(H, W)`` depth of the source frame, in the frame of the poses.

    Returns:
        ``flow (H, W, 2)`` and ``valid (H, W)`` (depth known, point in front of and inside
        the destination image).
    """
    height, width = depth.shape
    centers = pixel_centers(height, width, dtype=depth.dtype).reshape(-1, 2)
    world = transform_points(invert_se3(w2c_src), unproject(K, centers, depth.reshape(-1)))
    uv, z = project(K, transform_points(w2c_dst, world))
    valid = (depth.reshape(-1) > 0) & (z > 1e-6) & in_image(uv, height, width)
    return (uv - centers).reshape(height, width, 2), valid.reshape(height, width)


def _clean_mask(mask: np.ndarray, cfg: MotionSegConfig) -> np.ndarray:
    """Morphological clean-up and removal of small connected components."""
    mask = mask.astype(np.uint8)
    if cfg.morph_radius > 0:
        size = 2 * cfg.morph_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.min_area
    return keep[labels]


def flow_residuals(
    images: Tensor,
    depth: Tensor,
    K: Tensor,
    w2c: Tensor,
    flow: FlowEstimator,
    cfg: MotionSegConfig | None = None,
) -> Tensor:
    """Per-pixel disagreement between observed and rigid flow, ``(T, H, W)`` in pixels.

    The residual is normalised: the part explainable by a relative depth error (a fraction
    ``relative_threshold`` of the rigid flow magnitude) is subtracted first. Each frame is
    compared with its neighbours at ``+-step`` for every step, and the median over the
    valid comparisons is returned (NaN where no comparison is available).
    """
    cfg = cfg or MotionSegConfig()
    n_frames = images.shape[0]
    stacks: list[list[Tensor]] = [[] for _ in range(n_frames)]
    for step in cfg.steps:
        for t in range(n_frames):
            for other in (t - step, t + step):
                if not 0 <= other < n_frames:
                    continue
                observed = flow.between(images, t, other)
                reverse = flow.between(images, other, t)
                reliable = forward_backward_error(observed, reverse) < cfg.fb_threshold
                rigid, valid = rigid_flow(depth[t], K, w2c[t], w2c[other])
                residual = (observed - rigid).norm(dim=-1) - cfg.relative_threshold * rigid.norm(
                    dim=-1
                )
                stacks[t].append(
                    torch.where(reliable & valid, residual.clamp_min(0.0), float("nan"))
                )
    return torch.stack([torch.nanmedian(torch.stack(s), dim=0).values for s in stacks])


def motion_masks(residuals: Tensor, cfg: MotionSegConfig | None = None) -> Tensor:
    """Threshold and clean flow residuals into dynamic masks ``(T, H, W)``."""
    cfg = cfg or MotionSegConfig()
    raw = torch.nan_to_num(residuals, nan=0.0) > cfg.residual_threshold
    cleaned = [_clean_mask(frame.numpy(), cfg) for frame in raw]
    return torch.from_numpy(np.stack(cleaned))


def label_tracks(
    tracks: Tracks,
    reproj_error: Tensor,
    static: Tensor,
    masks: Tensor | None = None,
    cfg: MotionSegConfig | None = None,
) -> Tensor:
    """Dynamic / static label ``(N,)`` of every track.

    A track reconstructed as a static point (``static``) is never dynamic. Otherwise it is
    dynamic when no static 3D point explains it (large ``reproj_error``) and, if dense
    ``masks`` are available, when it also lies on the dynamic mask most of the time (which
    separates real motion from plain tracking failures).
    """
    cfg = cfg or MotionSegConfig()
    dynamic = ~static & (reproj_error > cfg.track_threshold)
    if masks is not None:
        n_frames, height, width = masks.shape
        col = tracks.uv[..., 0].floor().clamp(0, width - 1).to(torch.int64)
        row = tracks.uv[..., 1].floor().clamp(0, height - 1).to(torch.int64)
        frame = torch.arange(n_frames)[None, :].expand_as(col)
        on_mask = masks[frame, row, col] & tracks.visible
        fraction = on_mask.sum(dim=1) / tracks.visible.sum(dim=1).clamp_min(1)
        dynamic = dynamic & (fraction > cfg.track_mask_fraction)
    return dynamic
