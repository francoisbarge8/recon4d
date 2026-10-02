"""Adaptive density control (clone / split / prune) of a Gaussian cloud."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.gaussians.model import CloudOptimizer, GaussianCloud, inverse_sigmoid
from recon4d.geometry.rotations import quat_to_rotmat


@dataclass(frozen=True)
class DensifyConfig:
    """Schedule and thresholds of the adaptive density control of 3DGS.

    Attributes:
        start, stop: iteration range in which Gaussians are added and pruned.
        interval: number of iterations between two density updates.
        grad_threshold: mean screen-space positional gradient (in NDC units) above which a
            Gaussian is densified: it marks under-reconstructed regions.
        size_threshold: Gaussians larger than this fraction of the scene extent are
            *split* into two smaller ones; smaller ones are *cloned*.
        min_opacity: Gaussians more transparent than this are pruned.
        max_scale: Gaussians larger than this fraction of the scene extent are pruned.
        max_gaussians: hard budget; when it would be exceeded only the candidates with the
            largest gradients are densified.
        max_growth: at most this fraction of the current Gaussians is densified per step
            (again the ones with the largest gradients), which keeps the growth gradual
            whatever the scale of the loss.
        split_shrink: children of a split are this many times smaller than their parent.
        opacity_reset_interval: every this many iterations all opacities are clamped to
            ``opacity_reset_value`` to let the optimisation remove floaters (0 disables).
    """

    start: int = 200
    stop: int = 1500
    interval: int = 100
    grad_threshold: float = 2e-4
    size_threshold: float = 0.01
    min_opacity: float = 0.005
    max_scale: float = 0.2
    max_gaussians: int = 20000
    max_growth: float = 0.1
    split_shrink: float = 1.6
    opacity_reset_interval: int = 0
    opacity_reset_value: float = 0.01

    def is_due(self, iteration: int) -> bool:
        return self.start <= iteration < self.stop and iteration % self.interval == 0

    def reset_is_due(self, iteration: int) -> bool:
        return (
            self.opacity_reset_interval > 0
            and 0 < iteration < self.stop
            and iteration % self.opacity_reset_interval == 0
        )


class DensificationStats:
    """Running average of the screen-space positional gradient of every Gaussian."""

    def __init__(self, n: int) -> None:
        self.grad_sum = torch.zeros(n)
        self.count = torch.zeros(n)

    def update(self, means2d_grad: Tensor, visible: Tensor, width: int, height: int) -> None:
        """Accumulate ``|dL/d(mean2d)|`` of the Gaussians visible in the last render.

        Gradients are converted from pixel to NDC units so that the densification threshold
        does not depend on the image resolution.
        """
        scale = torch.tensor([0.5 * width, 0.5 * height], dtype=means2d_grad.dtype)
        norm = (means2d_grad * scale).norm(dim=-1)
        self.grad_sum += torch.where(visible, norm, torch.zeros_like(norm))
        self.count += visible.to(self.count.dtype)

    def mean(self) -> Tensor:
        return self.grad_sum / self.count.clamp_min(1.0)

    def reset(self, n: int) -> None:
        self.grad_sum = torch.zeros(n)
        self.count = torch.zeros(n)


def _select(params: dict[str, Tensor], mask: Tensor) -> dict[str, Tensor]:
    return {name: value[mask] for name, value in params.items()}


@torch.no_grad()
def densify_and_prune(
    cloud: GaussianCloud,
    optimizer: CloudOptimizer,
    stats: DensificationStats,
    cfg: DensifyConfig,
    extent: float,
    generator: torch.Generator | None = None,
) -> dict[str, int]:
    """One step of adaptive density control. Returns the number of edits of each kind.

    Under-reconstructed regions are detected through a large average view-space positional
    gradient. There, small Gaussians are duplicated (the geometry is under-covered) and
    large ones are replaced by two samples drawn from them (the geometry is over-covered by
    one blob). Transparent and oversized Gaussians are removed.
    """
    n = len(cloud)
    grads = stats.mean()
    scales = cloud.scales.amax(dim=1)
    candidates = grads >= cfg.grad_threshold

    budget = min(max(cfg.max_gaussians - n, 0), int(cfg.max_growth * n))
    if int(candidates.sum()) > budget:
        # Not enough room for everybody: keep the most under-reconstructed ones.
        keep = torch.zeros_like(candidates)
        if budget > 0:
            top = torch.topk(torch.where(candidates, grads, torch.zeros_like(grads)), budget)
            keep[top.indices] = True
        candidates = candidates & keep

    clone = candidates & (scales <= cfg.size_threshold * extent)
    split = candidates & ~clone
    params = cloud.detached()

    new: list[dict[str, Tensor]] = []
    if clone.any():
        new.append(_select(params, clone))
    if split.any():
        parents = _select(params, split)
        std = torch.exp(parents["log_scales"])
        rotation = quat_to_rotmat(parents["quats"])
        for _ in range(2):
            noise = torch.randn(std.shape, generator=generator, dtype=std.dtype) * std
            child = dict(parents)
            child["means"] = parents["means"] + (rotation @ noise[..., None])[..., 0]
            child["log_scales"] = parents["log_scales"] - torch.log(torch.tensor(cfg.split_shrink))
            new.append(child)

    # Prune: split parents, transparent and oversized Gaussians.
    remove = split | (cloud.opacities < cfg.min_opacity) | (scales > cfg.max_scale * extent)
    n_pruned = int((remove & ~split).sum())
    optimizer.prune(~remove)
    if new:
        optimizer.append({name: torch.cat([part[name] for part in new]) for name in params})
    stats.reset(len(cloud))
    return {
        "cloned": int(clone.sum()),
        "split": int(split.sum()),
        "pruned": n_pruned,
        "total": len(cloud),
    }


@torch.no_grad()
def reset_opacity(cloud: GaussianCloud, optimizer: CloudOptimizer, value: float) -> None:
    """Clamp all opacities to at most ``value`` and clear their optimiser state."""
    ceiling = float(inverse_sigmoid(value))
    optimizer.reset("opacity_logits", cloud.params["opacity_logits"].detach().clamp_max(ceiling))
