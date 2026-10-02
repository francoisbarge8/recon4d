"""Procedural solid textures.

Textures are functions of a 3D point in *object-local* coordinates, so they stay attached
to a surface as the object moves. All randomness comes from an integer lattice hash, which
makes the generated images bit-reproducible across platforms and library versions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

Color = tuple[float, float, float]

_MASK32 = 0xFFFFFFFF


def lattice_hash(ix: Tensor, iy: Tensor, iz: Tensor, seed: int | Tensor) -> Tensor:
    """Hash integer lattice coordinates to floats in ``[0, 1)`` (32-bit arithmetic in int64)."""
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 1274126177) & _MASK32
    h = ((h ^ (h >> 13)) * 1274126177) & _MASK32
    h = (h ^ (h >> 16)) & _MASK32
    return (h & 0xFFFFFF).to(torch.float32) / float(1 << 24)


_CORNERS = torch.tensor(
    [[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], dtype=torch.int64
)


def value_noise(p: Tensor, seed: int | Tensor) -> Tensor:
    """Trilinearly interpolated lattice noise in ``[0, 1]`` at points ``(..., 3)``.

    ``seed`` may be a tensor broadcastable to ``p.shape[:-1]`` (one seed per octave).
    """
    base = torch.floor(p)
    frac = (p - base).to(torch.float32)
    frac = frac * frac * (3.0 - 2.0 * frac)  # smoothstep fade
    corners = _CORNERS.to(p.device)
    lattice = base.to(torch.int64)[..., None, :] + corners  # (..., 8, 3)
    if isinstance(seed, Tensor):
        seed = seed[..., None]
    values = lattice_hash(lattice[..., 0], lattice[..., 1], lattice[..., 2], seed)
    weights = torch.where(corners.bool(), frac[..., None, :], 1.0 - frac[..., None, :]).prod(dim=-1)
    return (weights * values).sum(dim=-1)


def fbm(p: Tensor, octaves: int, seed: int, lacunarity: float = 2.0, gain: float = 0.5) -> Tensor:
    """Fractal Brownian motion: a normalised sum of ``octaves`` noise layers, in ``[0, 1]``."""
    index = torch.arange(octaves, device=p.device)
    frequency = (lacunarity**index).to(p.dtype)
    amplitude = (gain**index).to(torch.float32)
    layers = value_noise(p[:, None, :] * frequency[None, :, None], seed + 101 * index)
    return (layers * amplitude).sum(dim=-1) / amplitude.sum()


def _smoothstep(edge0: float, edge1: float, x: Tensor) -> Tensor:
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class Texture:
    """Two-colour procedural pattern with fractal brightness detail.

    Attributes:
        pattern: ``"checker"``, ``"stripes"``, ``"dots"`` or ``"noise"``.
        color_a, color_b: the two albedo colours blended by the pattern mask.
        frequency: pattern cells per unit of local length.
        detail: amplitude of the multiplicative fBm modulation (0 disables it). The
            modulation gives every surface trackable fine-scale structure.
        detail_frequency: base frequency of the modulation.
        seed: lattice-hash seed.
    """

    pattern: str
    color_a: Color
    color_b: Color
    frequency: float = 2.0
    detail: float = 0.25
    detail_frequency: float = 7.0
    seed: int = 0

    def mask(self, p: Tensor) -> Tensor:
        q = p * self.frequency
        if self.pattern == "checker":
            cell = torch.floor(q).to(torch.int64).sum(dim=-1)
            return (cell % 2).to(torch.float32)
        if self.pattern == "stripes":
            axis = torch.tensor([0.8, 0.5, 0.33], dtype=p.dtype, device=p.device)
            phase = (q * axis).sum(dim=-1) + 0.6 * value_noise(q * 0.5, self.seed + 7)
            return _smoothstep(-0.25, 0.25, torch.sin(2.0 * torch.pi * phase).to(torch.float32))
        if self.pattern == "dots":
            cell = torch.floor(q)
            i = cell.to(torch.int64)
            ix, iy, iz = i.unbind(-1)
            jitter = torch.stack(
                [lattice_hash(ix, iy, iz, self.seed + k) for k in (11, 12, 13)], dim=-1
            )
            center = cell + 0.3 + 0.4 * jitter
            dist = (q - center).to(torch.float32).norm(dim=-1)
            return 1.0 - _smoothstep(0.26, 0.32, dist)
        if self.pattern == "noise":
            return _smoothstep(0.42, 0.58, fbm(q, octaves=2, seed=self.seed + 3))
        raise ValueError(f"unknown texture pattern {self.pattern!r}")

    def albedo(self, p: Tensor) -> Tensor:
        """Albedo ``(N, 3)`` in ``[0, 1]`` at local points ``p (N, 3)``."""
        m = self.mask(p)[:, None]
        a = torch.tensor(self.color_a, dtype=torch.float32, device=p.device)
        b = torch.tensor(self.color_b, dtype=torch.float32, device=p.device)
        color = (1.0 - m) * a + m * b
        if self.detail > 0:
            n = fbm(p * self.detail_frequency, octaves=2, seed=self.seed + 29)
            color = color * (1.0 + self.detail * (2.0 * n[:, None] - 1.0))
        return color.clamp(0.0, 1.0)
