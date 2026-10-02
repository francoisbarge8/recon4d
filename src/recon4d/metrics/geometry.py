"""Point-cloud metrics: Chamfer distance, accuracy / completeness and F-score."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor


def nearest_distances(source: Tensor, target: Tensor) -> Tensor:
    """Euclidean distance from every ``source`` point to its nearest ``target`` point.

    Uses a k-d tree, i.e. exact nearest neighbours in ``O((N + M) log M)``.
    """
    if target.shape[0] == 0:
        return torch.full((source.shape[0],), float("inf"), dtype=source.dtype)
    tree = cKDTree(target.detach().cpu().numpy().astype(np.float64))
    distances, _ = tree.query(source.detach().cpu().numpy().astype(np.float64))
    return torch.as_tensor(distances, dtype=source.dtype)


def nearest_distances_torch(source: Tensor, target: Tensor, chunk: int = 4096) -> Tensor:
    """Differentiable brute-force variant of :func:`nearest_distances` (chunked ``cdist``)."""
    return torch.cat([torch.cdist(part, target).amin(dim=1) for part in torch.split(source, chunk)])


def chamfer_distance(pred: Tensor, gt: Tensor, differentiable: bool = False) -> dict[str, Tensor]:
    """Symmetric Chamfer distance between two point sets.

    Returns a dictionary with

    * ``accuracy``: mean distance from predicted points to the ground truth (how far the
      reconstruction is from the true surface);
    * ``completeness``: mean distance from ground-truth points to the prediction (how much
      of the true surface is missing);
    * ``chamfer``: their average, the usual (L1) Chamfer distance.

    Distances are un-squared, in the units of the inputs.
    """
    nearest = nearest_distances_torch if differentiable else nearest_distances
    accuracy = nearest(pred, gt).mean()
    completeness = nearest(gt, pred).mean()
    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "chamfer": 0.5 * (accuracy + completeness),
    }


def f_score(pred: Tensor, gt: Tensor, threshold: float) -> dict[str, Tensor]:
    """Precision, recall and F-score at a distance ``threshold`` (Knapitsch et al., 2017).

    Precision is the fraction of predicted points closer than ``threshold`` to the ground
    truth, recall the fraction of ground-truth points closer than ``threshold`` to the
    prediction, and the F-score their harmonic mean.
    """
    precision = (nearest_distances(pred, gt) < threshold).to(torch.float32).mean()
    recall = (nearest_distances(gt, pred) < threshold).to(torch.float32).mean()
    denominator = precision + recall
    f = torch.where(denominator > 0, 2.0 * precision * recall / denominator.clamp_min(1e-12), 0.0)
    return {"precision": precision, "recall": recall, "fscore": f}


def geometry_metrics(
    pred: Tensor, gt: Tensor, thresholds: tuple[float, ...] = (0.02, 0.05)
) -> dict[str, float]:
    """Chamfer distance and F-scores of a reconstructed point cloud, as plain floats.

    Both point sets must be expressed in the same metric frame (align the reconstruction
    to the ground truth first, see :func:`recon4d.metrics.pose.align_trajectory`).
    """
    if pred.shape[0] == 0:
        raise ValueError("the predicted point cloud is empty")
    to_gt = nearest_distances(pred, gt)
    to_pred = nearest_distances(gt, pred)
    metrics = {
        "accuracy": float(to_gt.mean()),
        "completeness": float(to_pred.mean()),
        "chamfer": float(0.5 * (to_gt.mean() + to_pred.mean())),
    }
    for threshold in thresholds:
        precision = float((to_gt < threshold).float().mean())
        recall = float((to_pred < threshold).float().mean())
        total = precision + recall
        metrics[f"fscore@{threshold:g}"] = 2.0 * precision * recall / total if total > 0 else 0.0
        metrics[f"precision@{threshold:g}"] = precision
        metrics[f"recall@{threshold:g}"] = recall
    return metrics
