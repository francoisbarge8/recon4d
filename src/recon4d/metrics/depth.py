"""Depth map metrics and the alignment that monocular predictions require.

Monocular depth is only defined up to an unknown scale (and, for affine-invariant models
such as MiDaS or Depth Anything, an unknown shift in inverse-depth space), so predictions
are aligned to the ground truth before the standard metrics are computed.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _least_squares_scale_shift(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Closed-form ``argmin_{s, t} sum (s x + t - y)^2``."""
    x64, y64 = x.to(torch.float64), y.to(torch.float64)
    n = x64.numel()
    sx, sy = x64.sum(), y64.sum()
    sxx, sxy = (x64 * x64).sum(), (x64 * y64).sum()
    det = n * sxx - sx * sx
    if det.abs() < 1e-12:
        return torch.ones((), dtype=x.dtype), (y64.mean() - x64.mean()).to(x.dtype)
    scale = (n * sxy - sx * sy) / det
    shift = (sy - scale * sx) / n
    return scale.to(x.dtype), shift.to(x.dtype)


def align_depth(
    pred: Tensor, gt: Tensor, mask: Tensor | None = None, mode: str = "scale"
) -> Tensor:
    """Align a predicted depth map to the ground truth.

    Args:
        pred, gt: depth maps of identical shape.
        mask: pixels used to estimate the alignment (default: where ``gt > 0``).
        mode: ``"none"``; ``"scale"`` (ratio of medians, for scale-invariant depth);
            ``"scale_shift"`` (least squares in depth space); or ``"disparity"`` (least
            squares scale and shift in inverse-depth space, the protocol for
            affine-invariant models, where ``pred`` is interpreted as an inverse depth).

    Returns:
        The aligned prediction, as a depth map.
    """
    if mask is None:
        mask = gt > 0
    if mode == "none":
        return pred
    if mode == "scale":
        return pred * (gt[mask].median() / pred[mask].median().clamp_min(1e-12))
    if mode == "scale_shift":
        scale, shift = _least_squares_scale_shift(pred[mask], gt[mask])
        return scale * pred + shift
    if mode == "disparity":
        scale, shift = _least_squares_scale_shift(pred[mask], 1.0 / gt[mask])
        return 1.0 / (scale * pred + shift).clamp_min(1e-6)
    raise ValueError(f"unknown alignment mode {mode!r}")


def depth_metrics(pred: Tensor, gt: Tensor, mask: Tensor | None = None) -> dict[str, float]:
    """Standard monocular depth metrics (Eigen et al., 2014) over the masked pixels.

    ``abs_rel``, ``sq_rel``, ``rmse``, ``rmse_log`` are errors (lower is better);
    ``delta1/2/3`` are the fractions of pixels with ``max(pred/gt, gt/pred) < 1.25^k``.
    """
    if mask is None:
        mask = gt > 0
    mask = mask & (pred > 0)
    p, g = pred[mask].to(torch.float64), gt[mask].to(torch.float64)
    if p.numel() == 0:
        raise ValueError("no valid pixel to evaluate")
    ratio = torch.maximum(p / g, g / p)
    return {
        "abs_rel": float(((p - g).abs() / g).mean()),
        "sq_rel": float((((p - g) ** 2) / g).mean()),
        "rmse": float(torch.sqrt(((p - g) ** 2).mean())),
        "rmse_log": float(torch.sqrt(((torch.log(p) - torch.log(g)) ** 2).mean())),
        "delta1": float((ratio < 1.25).double().mean()),
        "delta2": float((ratio < 1.25**2).double().mean()),
        "delta3": float((ratio < 1.25**3).double().mean()),
    }
