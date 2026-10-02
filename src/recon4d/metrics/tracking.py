"""Point tracking metrics (TAP-Vid protocol) and 3D trajectory errors."""

from __future__ import annotations

import torch
from torch import Tensor

TAPVID_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)
TAPVID_RESOLUTION = 256.0


def tapvid_metrics(
    pred_uv: Tensor,
    pred_visible: Tensor,
    gt_uv: Tensor,
    gt_visible: Tensor,
    query_frames: Tensor,
    width: int,
    height: int,
    thresholds: tuple[float, ...] = TAPVID_THRESHOLDS,
) -> dict[str, float]:
    """Metrics of *TAP-Vid* (Doersch et al., NeurIPS 2022).

    Coordinates are rescaled to a 256 x 256 image, as in the benchmark, so that the pixel
    thresholds have the same meaning at any resolution. The query frame of every track is
    excluded from the evaluation.

    Args:
        pred_uv, gt_uv: ``(N, T, 2)``.
        pred_visible, gt_visible: ``(N, T)``.
        query_frames: ``(N,)`` frame in which each track was queried.

    Returns:
        ``occlusion_accuracy``; ``pts_within_k`` (fraction of visible points tracked
        within ``k`` pixels) and their mean ``delta_avg``; ``jaccard_k`` and their mean
        ``average_jaccard`` (position and visibility jointly); ``epe`` (mean error in
        *original* pixels over the visible points); ``epe_tracked`` (the same, restricted
        to the points the tracker itself reports as visible, i.e. the accuracy of the
        tracks it actually delivers) and ``track_length`` (mean number of frames a track
        is reported visible).
    """
    n_frames = gt_visible.shape[1]
    evaluate = torch.arange(n_frames)[None, :] != query_frames[:, None]
    scale = torch.tensor([TAPVID_RESOLUTION / width, TAPVID_RESOLUTION / height], dtype=gt_uv.dtype)
    distance = ((pred_uv - gt_uv) * scale).norm(dim=-1)
    gt_vis = gt_visible & evaluate
    pred_vis = pred_visible & evaluate

    metrics: dict[str, float] = {}
    n_eval = evaluate.sum().clamp_min(1)
    metrics["occlusion_accuracy"] = float(((pred_visible == gt_visible) & evaluate).sum() / n_eval)
    n_gt = gt_vis.sum().clamp_min(1)
    within_all, jaccard_all = [], []
    for threshold in thresholds:
        within = distance < threshold
        correct = within & gt_vis
        frac = float(correct.sum() / n_gt)
        true_positive = correct & pred_vis
        false_positive = pred_vis & ~correct
        jaccard = float(true_positive.sum() / (gt_vis.sum() + false_positive.sum()).clamp_min(1))
        metrics[f"pts_within_{threshold:g}"] = frac
        metrics[f"jaccard_{threshold:g}"] = jaccard
        within_all.append(frac)
        jaccard_all.append(jaccard)
    metrics["delta_avg"] = sum(within_all) / len(within_all)
    metrics["average_jaccard"] = sum(jaccard_all) / len(jaccard_all)
    raw = (pred_uv - gt_uv).norm(dim=-1)
    metrics["epe"] = float((raw * gt_vis).sum() / n_gt)
    both = gt_vis & pred_vis
    metrics["epe_tracked"] = float((raw * both).sum() / both.sum().clamp_min(1))
    metrics["track_length"] = float(pred_visible.sum(dim=1).to(torch.float32).mean())
    return metrics


def trajectory_error_3d(
    pred_xyz: Tensor, gt_xyz: Tensor, valid: Tensor, thresholds: tuple[float, ...] = (0.05, 0.10)
) -> dict[str, float]:
    """3D end-point error of trajectories expressed in the ground-truth metric frame.

    Args:
        pred_xyz, gt_xyz: ``(N, T, 3)``.
        valid: ``(N, T)`` entries to evaluate.

    Returns:
        ``epe_3d`` (mean Euclidean error) and ``delta_3d_k``, the fraction of points
        within ``k`` (same unit as the inputs), as in the 3D tracking evaluation of
        *Shape of Motion* (5 cm and 10 cm).
    """
    error = (pred_xyz - gt_xyz).norm(dim=-1)
    n = valid.sum().clamp_min(1)
    metrics = {"epe_3d": float((error * valid).sum() / n)}
    for threshold in thresholds:
        metrics[f"delta_3d_{threshold:g}"] = float(((error < threshold) & valid).sum() / n)
    return metrics
