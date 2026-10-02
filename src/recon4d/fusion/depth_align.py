"""Bring monocular depth maps into the metric frame of the camera poses.

A monocular network predicts each frame independently and only up to an unknown scale (and
shift, for affine-invariant models), so its maps neither agree with each other nor with the
camera trajectory. Structure-from-motion, on the other hand, gives few but multi-view
consistent 3D points. This module fuses the two: for every frame, the dense prediction is
warped so that it passes through the sparse SfM depths.

Two models are fitted per frame, robustly (iteratively re-weighted least squares):

* a **global** alignment: a scale (``z = s z_pred``) for scale-invariant depth, or a
  scale and shift in inverse depth (``1/z = a d_pred + b``) for affine-invariant disparity;
* optionally a **smooth correction field**: a coarse grid of log-scale offsets, bilinearly
  interpolated over the image, that absorbs the low-frequency errors a global model cannot
  (monocular depth is typically right locally and wrong at large scale). The fit is a
  small linear least-squares problem regularised towards a flat field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.geometry.camera import pixel_centers, sample_depth, transform_points
from recon4d.types import Tracks

DTYPE = torch.float64


@dataclass
class DepthAlignConfig:
    """Parameters of the depth alignment.

    Attributes:
        kind: ``"scale"`` when the prediction is a depth up to scale, ``"disparity"`` when
            it is an affine-invariant inverse depth.
        grid: resolution ``(rows, cols)`` of the smooth correction field; ``None`` keeps
            the global alignment only.
        grid_smoothness: weight of the penalty on differences between neighbouring nodes,
            relative to the data weight of an average node.
        grid_prior: weight of the penalty pulling every node to zero correction, relative
            to the data weight of an average node.
        robust_scale: smallest scale of the Cauchy loss of the robust fits, as a relative
            depth error (the scale adapts to the spread of the residuals above this
            floor); residuals well above the scale are treated as outliers.
        min_points: frames with fewer sparse depths reuse the parameters of their nearest
            well-constrained frame.
        irls_iterations: number of re-weighting iterations.
    """

    kind: str = "scale"
    grid: tuple[int, int] | None = (4, 5)
    grid_smoothness: float = 0.01
    grid_prior: float = 0.001
    robust_scale: float = 0.05
    min_points: int = 12
    irls_iterations: int = 8


@dataclass
class AlignedDepth:
    depth: Tensor
    """``(T, H, W)`` depth maps in the frame of the camera poses."""
    valid: Tensor
    """``(T, H, W)`` True where the aligned depth is finite and positive."""
    residual: Tensor
    """``(T,)`` median relative error on the sparse depths after alignment."""
    num_points: Tensor
    """``(T,)`` number of sparse depths used per frame."""


def _robust_weights(residual: Tensor, floor: float) -> Tensor:
    """IRLS weights of the Cauchy loss, ``1 / (1 + (r / c)^2)``, with an adaptive scale.

    Unlike Huber's, the influence of a residual vanishes as it grows, so gross outliers
    (mis-triangulated points, tracks sitting on an occlusion boundary) do not bias the fit.

    The scale ``c`` follows the spread of the residuals (2.385 robust standard deviations,
    the 95%-efficiency tuning of the Cauchy loss) and never drops below ``floor``. A fixed
    small scale would be wrong here: when the model cannot represent the data exactly, as a
    single scale facing a spatially varying error, the inliers themselves have large
    residuals, and treating them as outliers makes the fit collapse.
    """
    sigma = 1.4826 * (residual - residual.median()).abs().median()
    scale = torch.clamp(2.385 * sigma, min=floor)
    return 1.0 / (1.0 + (residual / scale) ** 2)


def fit_scale(pred: Tensor, target: Tensor, scale: float, iterations: int) -> Tensor:
    """Robust ``s`` minimising the relative error of ``s * pred`` w.r.t. ``target``.

    The fit is done in log space, where the model is a pure offset: start from the median
    and refine by iteratively re-weighted least squares.
    """
    log_ratio = torch.log(target) - torch.log(pred)
    offset = log_ratio.median()
    for _ in range(iterations):
        w = _robust_weights(log_ratio - offset, scale)
        offset = (w * log_ratio).sum() / w.sum()
    return torch.exp(offset)


def fit_scale_shift(
    pred: Tensor, target: Tensor, scale: float, iterations: int
) -> tuple[Tensor, Tensor]:
    """Robust ``(a, b)`` such that ``a * pred + b`` matches ``target``.

    The fit starts from a Theil-Sen estimate (median of the slopes between random pairs of
    samples, then median intercept), which tolerates a large fraction of outliers, and is
    refined by iteratively re-weighted least squares. Residuals are normalised by the
    median of ``target`` so that ``scale`` is a relative error.
    """
    norm = target.median().abs().clamp_min(1e-12)
    n = pred.shape[0]
    generator = torch.Generator().manual_seed(0)
    i = torch.randint(n, (2000,), generator=generator)
    j = torch.randint(n, (2000,), generator=generator)
    dx = pred[i] - pred[j]
    usable = dx.abs() > 1e-12
    if usable.sum() < 2:
        return (target.median() / pred.median().clamp_min(1e-12)).to(pred.dtype), pred.new_zeros(())
    a = ((target[i] - target[j])[usable] / dx[usable]).median()
    b = (target - a * pred).median()
    for _ in range(iterations):
        w = _robust_weights((a * pred + b - target) / norm, scale)
        sw = w.sum()
        sx, sy = (w * pred).sum(), (w * target).sum()
        sxx, sxy = (w * pred * pred).sum(), (w * pred * target).sum()
        det = sw * sxx - sx * sx
        if det.abs() < 1e-18:
            break
        a = (sw * sxy - sx * sy) / det
        b = (sy - a * sx) / sw
    return a, b


def _bilinear_weights(uv: Tensor, height: int, width: int, grid: tuple[int, int]) -> Tensor:
    """``(N, rows * cols)`` interpolation weights of image positions on a coarse grid.

    Grid nodes are spread uniformly, the outermost ones lying on the image border.
    """
    rows, cols = grid
    gx = (uv[:, 0] / width).clamp(0.0, 1.0) * (cols - 1)
    gy = (uv[:, 1] / height).clamp(0.0, 1.0) * (rows - 1)
    x0 = torch.floor(gx).clamp(0, cols - 2).to(torch.int64)
    y0 = torch.floor(gy).clamp(0, rows - 2).to(torch.int64)
    wx = gx - x0
    wy = gy - y0
    weights = torch.zeros(uv.shape[0], rows * cols, dtype=uv.dtype)
    index = torch.arange(uv.shape[0])
    for dy, dx, w in (
        (0, 0, (1 - wx) * (1 - wy)),
        (0, 1, wx * (1 - wy)),
        (1, 0, (1 - wx) * wy),
        (1, 1, wx * wy),
    ):
        weights[index, (y0 + dy) * cols + x0 + dx] += w
    return weights


def _smoothness_matrix(grid: tuple[int, int]) -> Tensor:
    """Finite-difference operator whose rows are differences between neighbouring nodes."""
    rows, cols = grid
    index = torch.arange(rows * cols).reshape(rows, cols)
    pairs = torch.cat(
        [
            torch.stack([index[:, :-1].reshape(-1), index[:, 1:].reshape(-1)], dim=1),
            torch.stack([index[:-1].reshape(-1), index[1:].reshape(-1)], dim=1),
        ]
    )
    D = torch.zeros(pairs.shape[0], rows * cols, dtype=DTYPE)
    D[torch.arange(pairs.shape[0]), pairs[:, 0]] = -1.0
    D[torch.arange(pairs.shape[0]), pairs[:, 1]] = 1.0
    return D


def fit_correction_field(
    uv: Tensor, log_error: Tensor, height: int, width: int, cfg: DepthAlignConfig
) -> Tensor:
    """Smooth log-scale field ``g`` with ``g(uv_i) ~ log_error_i``; returns ``(rows, cols)``.

    Solves ``min_g sum_i w_i (B_i g - e_i)^2 + m (l_s |D g|^2 + l_p |g|^2)`` by IRLS, where
    ``B`` holds bilinear interpolation weights and ``D`` neighbour differences. ``m`` is the
    number of samples per grid node, so that the regularisation is expressed relative to
    the data term and does not depend on how many tracks happen to be visible.
    """
    grid = cfg.grid
    B = _bilinear_weights(uv.to(DTYPE), height, width, grid)
    e = log_error.to(DTYPE)
    D = _smoothness_matrix(grid)
    per_node = uv.shape[0] / B.shape[1]
    regulariser = per_node * (
        cfg.grid_smoothness * D.T @ D + cfg.grid_prior * torch.eye(B.shape[1], dtype=DTYPE)
    )
    w = torch.ones_like(e)
    g = torch.zeros(B.shape[1], dtype=DTYPE)
    for _ in range(cfg.irls_iterations):
        A = B.T @ (B * w[:, None]) + regulariser
        g = torch.linalg.solve(A, B.T @ (w * e))
        w = _robust_weights(B @ g - e, cfg.robust_scale)
    return g.reshape(grid)


def sparse_depths(
    tracks: Tracks, points: Tensor, inlier: Tensor, w2c: Tensor, frame: int
) -> tuple[Tensor, Tensor]:
    """Image positions and z-depths of the SfM points observed in ``frame``."""
    seen = inlier[:, frame]
    cam = transform_points(w2c[frame], points[seen].to(w2c.dtype))
    return tracks.uv[seen, frame], cam[:, 2]


def align_depth_maps(
    pred: Tensor,
    tracks: Tracks,
    points: Tensor,
    inlier: Tensor,
    w2c: Tensor,
    cfg: DepthAlignConfig | None = None,
) -> AlignedDepth:
    """Align per-frame depth predictions to sparse multi-view consistent depths.

    Args:
        pred: ``(T, H, W)`` predicted depth (``kind="scale"``) or inverse depth
            (``kind="disparity"``).
        tracks: the 2D tracks that were reconstructed.
        points: ``(N, 3)`` their 3D points.
        inlier: ``(N, T)`` observations that are consistent with a static 3D point.
        w2c: ``(T, 4, 4)`` camera poses.
    """
    cfg = cfg or DepthAlignConfig()
    if cfg.kind not in ("scale", "disparity"):
        raise ValueError(f"unknown depth kind {cfg.kind!r}")
    n_frames, height, width = pred.shape
    pred64 = pred.to(DTYPE)
    # Work on a depth-like map so that sampling interpolates inverse depth in both cases.
    depth_like = pred64 if cfg.kind == "scale" else 1.0 / pred64.clamp_min(1e-9)

    samples = []
    for t in range(n_frames):
        uv, z = sparse_depths(tracks, points, inlier, w2c.to(DTYPE), t)
        value, ok = sample_depth(depth_like[t], uv.to(DTYPE))
        ok = ok & (z > 0)
        samples.append((uv[ok].to(DTYPE), value[ok], z[ok]))
    counts = torch.tensor([s[0].shape[0] for s in samples])
    constrained = torch.nonzero(counts >= cfg.min_points)[:, 0]
    if constrained.numel() == 0:
        raise RuntimeError("no frame has enough sparse depths to align the depth maps")

    # Global alignment per frame; under-constrained frames borrow from the nearest one.
    globally_aligned = torch.zeros_like(pred64)
    params: dict[int, tuple[Tensor, Tensor]] = {}
    for t in constrained.tolist():
        _, value, z = samples[t]
        if cfg.kind == "scale":
            params[t] = (
                fit_scale(value, z, cfg.robust_scale, cfg.irls_iterations),
                torch.zeros(()),
            )
        else:
            params[t] = fit_scale_shift(1.0 / value, 1.0 / z, cfg.robust_scale, cfg.irls_iterations)
    for t in range(n_frames):
        source = int(constrained[(constrained - t).abs().argmin()])
        a, b = params[source]
        if cfg.kind == "scale":
            globally_aligned[t] = a * pred64[t]
        else:
            globally_aligned[t] = 1.0 / (a * pred64[t] + b).clamp_min(1e-9)

    aligned = globally_aligned.clone()
    if cfg.grid is not None:
        # Interpolation weights of every pixel centre on the coarse grid (shared by frames).
        centers = pixel_centers(height, width, dtype=DTYPE).reshape(-1, 2)
        node_weights = _bilinear_weights(centers, height, width, cfg.grid)
    residual = torch.full((n_frames,), float("nan"), dtype=DTYPE)
    for t in constrained.tolist():
        uv, _, z = samples[t]
        current, ok = sample_depth(globally_aligned[t], uv)
        if cfg.grid is not None and ok.sum() >= cfg.min_points:
            log_error = torch.log(z[ok]) - torch.log(current[ok])
            field = fit_correction_field(uv[ok], log_error, height, width, cfg)
            dense = (node_weights @ field.reshape(-1)).reshape(height, width)
            aligned[t] = globally_aligned[t] * torch.exp(dense)
        final, ok = sample_depth(aligned[t], uv)
        if ok.any():
            residual[t] = ((final[ok] - z[ok]).abs() / z[ok]).median()

    valid = torch.isfinite(aligned) & (aligned > 0)
    aligned = torch.where(valid, aligned, torch.zeros_like(aligned))
    return AlignedDepth(aligned.to(pred.dtype), valid, residual.to(pred.dtype), counts)
