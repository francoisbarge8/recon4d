"""Multi-view linear triangulation."""

from __future__ import annotations

import torch
from torch import Tensor

from recon4d.geometry.camera import camera_centers, project, transform_points


def triangulate_dlt(K: Tensor, w2c: Tensor, uv: Tensor, visible: Tensor) -> tuple[Tensor, Tensor]:
    """Triangulate points from an arbitrary number of calibrated views.

    Solves the homogeneous DLT system in normalised image coordinates, restricted to the
    views where each point is observed.

    Args:
        K: ``(3, 3)`` shared intrinsics.
        w2c: ``(V, 4, 4)`` world-to-camera matrices.
        uv: ``(N, V, 2)`` pixel observations.
        visible: ``(N, V)`` boolean observation mask.

    Returns:
        ``points (N, 3)`` and ``valid (N,)``; a point is valid when it is observed at least
        twice and its homogeneous coordinate is not degenerate.
    """
    dtype = torch.float64
    K64, w2c64, uv64 = K.to(dtype), w2c.to(dtype), uv.to(dtype)
    x = (uv64[..., 0] - K64[0, 2]) / K64[0, 0]
    y = (uv64[..., 1] - K64[1, 2]) / K64[1, 1]
    P = w2c64[:, :3, :]  # (V, 3, 4) projection in normalised coordinates
    rows_x = x[..., None] * P[None, :, 2, :] - P[None, :, 0, :]
    rows_y = y[..., None] * P[None, :, 2, :] - P[None, :, 1, :]
    A = torch.stack([rows_x, rows_y], dim=2)  # (N, V, 2, 4)
    A = A * visible[..., None, None].to(dtype)
    A = A.reshape(A.shape[0], -1, 4)
    # The solution is the eigenvector of A^T A with the smallest eigenvalue.
    _, eigvecs = torch.linalg.eigh(A.transpose(1, 2) @ A)
    X_h = eigvecs[:, :, 0]
    w = X_h[:, 3]
    valid = (visible.sum(dim=1) >= 2) & (w.abs() > 1e-12)
    w_safe = torch.where(valid, w, torch.ones_like(w))
    points = X_h[:, :3] / w_safe[:, None]
    return points.to(uv.dtype), valid


def reprojection_errors(
    K: Tensor, w2c: Tensor, points: Tensor, uv: Tensor
) -> tuple[Tensor, Tensor]:
    """Pixel reprojection error of ``points (N, 3)`` in every view.

    Returns ``errors (N, V)`` and ``depths (N, V)`` (z-depth in each camera).
    """
    cam = transform_points(w2c[None], points[:, None, None, :])[:, :, 0]  # (N, V, 3)
    proj, z = project(K, cam)
    return (proj - uv).norm(dim=-1), z


def triangulation_angles(w2c: Tensor, points: Tensor, visible: Tensor) -> Tensor:
    """Largest angle (radians) between viewing rays of each point over its observing views."""
    centers = camera_centers(w2c)  # (V, 3)
    rays = torch.nn.functional.normalize(points[:, None] - centers[None], dim=-1)  # (N, V, 3)
    cos = rays @ rays.transpose(1, 2)  # (N, V, V)
    pair = visible[:, :, None] & visible[:, None, :]
    cos = torch.where(pair, cos, torch.ones_like(cos))
    return torch.acos(cos.amin(dim=(1, 2)).clamp(-1.0, 1.0))
