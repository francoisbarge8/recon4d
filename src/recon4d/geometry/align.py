"""Closed-form point-set alignment (Kabsch / Umeyama) and Sim(3) helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.geometry.camera import make_se3


@dataclass(frozen=True)
class Sim3:
    """Similarity transform ``x -> scale * R @ x + t``."""

    scale: Tensor
    R: Tensor
    t: Tensor

    def apply(self, points: Tensor) -> Tensor:
        return self.scale * (points @ self.R.transpose(-1, -2)) + self.t

    def inverse(self) -> Sim3:
        R_inv = self.R.transpose(-1, -2)
        return Sim3(1.0 / self.scale, R_inv, -(R_inv @ self.t) / self.scale)

    def apply_to_c2w(self, c2w: Tensor) -> Tensor:
        """Move camera-to-world poses ``(..., 4, 4)`` into the target frame.

        Camera centres are mapped by the similarity and orientations by its rotation, so
        the result is again a rigid camera-to-world matrix.
        """
        R = self.R @ c2w[..., :3, :3]
        centers = self.apply(c2w[..., :3, 3])
        return make_se3(R, centers)

    def to(self, dtype: torch.dtype) -> Sim3:
        return Sim3(self.scale.to(dtype), self.R.to(dtype), self.t.to(dtype))


def umeyama(
    src: Tensor, dst: Tensor, weights: Tensor | None = None, with_scale: bool = True
) -> Sim3:
    """Least-squares similarity aligning ``src`` to ``dst`` (Umeyama, TPAMI 1991).

    Minimises ``sum_i w_i || dst_i - (s R src_i + t) ||^2`` over ``s > 0``, ``R`` in SO(3)
    and ``t``. With ``with_scale=False`` this is the Kabsch rigid alignment.

    Args:
        src, dst: ``(N, 3)`` corresponding points.
        weights: optional non-negative ``(N,)`` weights.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[-1] != 3:
        raise ValueError(f"expected two (N, 3) point sets, got {src.shape} and {dst.shape}")
    if src.shape[0] < 3:
        raise ValueError("at least 3 correspondences are required")
    if weights is None:
        weights = torch.ones(src.shape[0], dtype=src.dtype, device=src.device)
    w = (weights / weights.sum().clamp_min(1e-12))[:, None]
    mu_src = (w * src).sum(dim=0)
    mu_dst = (w * dst).sum(dim=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (w * dst_c).transpose(0, 1) @ src_c
    U, S, Vh = torch.linalg.svd(cov)
    # Reflection guard: force det(R) = +1.
    d = torch.ones(3, dtype=src.dtype, device=src.device)
    if torch.det(U) * torch.det(Vh) < 0:
        d[2] = -1.0
    R = U @ torch.diag(d) @ Vh
    if with_scale:
        var_src = (w[:, 0] * (src_c * src_c).sum(dim=-1)).sum()
        scale = (S * d).sum() / var_src.clamp_min(1e-12)
    else:
        scale = torch.ones((), dtype=src.dtype, device=src.device)
    t = mu_dst - scale * (R @ mu_src)
    return Sim3(scale, R, t)


def kabsch(src: Tensor, dst: Tensor, weights: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Rigid alignment ``dst ~ R @ src + t``; returns ``(R, t)``."""
    sim = umeyama(src, dst, weights=weights, with_scale=False)
    return sim.R, sim.t


def batched_kabsch(src: Tensor, dst: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Weighted rigid alignment for a batch of point sets.

    Args:
        src, dst: ``(B, N, 3)``.
        weights: ``(B, N)`` non-negative weights (zero for missing correspondences).

    Returns:
        ``R (B, 3, 3)`` and ``t (B, 3)`` with ``dst ~ R @ src + t``. Batches whose total
        weight is zero return the identity.
    """
    total = weights.sum(dim=-1, keepdim=True)
    w = (weights / total.clamp_min(1e-12))[..., None]
    mu_src = (w * src).sum(dim=1)
    mu_dst = (w * dst).sum(dim=1)
    src_c = src - mu_src[:, None]
    dst_c = dst - mu_dst[:, None]
    cov = (w * dst_c).transpose(1, 2) @ src_c
    # A tiny diagonal keeps the SVD well defined for degenerate (empty / collinear) sets.
    cov = cov + 1e-12 * torch.eye(3, dtype=src.dtype, device=src.device)
    U, _, Vh = torch.linalg.svd(cov)
    det = torch.det(U) * torch.det(Vh)
    d = torch.ones(src.shape[0], 3, dtype=src.dtype, device=src.device)
    d[:, 2] = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))
    R = U @ torch.diag_embed(d) @ Vh
    t = mu_dst - (R @ mu_src[..., None])[..., 0]
    empty = total[:, 0] <= 0
    if empty.any():
        R = torch.where(empty[:, None, None], torch.eye(3, dtype=R.dtype, device=R.device), R)
        t = torch.where(empty[:, None], torch.zeros_like(t), t)
    return R, t
