"""Camera trajectories for the synthetic benchmark."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from recon4d.data.synthetic.primitives import DTYPE
from recon4d.geometry.camera import look_at


def _smooth_noise(n: int, rng: np.random.RandomState, components: int = 3) -> np.ndarray:
    """Band-limited noise in roughly ``[-1, 1]``: a sum of a few low-frequency sinusoids."""
    s = np.linspace(0.0, 1.0, n)
    out = np.zeros(n)
    for k in range(1, components + 1):
        out += rng.uniform(0.3, 1.0) / k * np.sin(2.0 * np.pi * (k * 0.7 * s + rng.uniform()))
    return out / np.abs(out).max().clip(1e-9)


def orbit_trajectory(
    n_frames: int,
    target: tuple[float, float, float],
    radius: float,
    height: float,
    angle_start_deg: float,
    angle_end_deg: float,
    radius_end: float | None = None,
    height_end: float | None = None,
    shake: float = 0.0,
    seed: int = 0,
) -> Tensor:
    """Camera-to-world poses ``(T, 4, 4)`` of a camera orbiting ``target``.

    The camera moves on an arc while always looking near ``target``; radius and height can
    drift linearly. ``shake`` adds smooth hand-held perturbations (metres) to the camera
    position and to the look-at point.
    """
    rng = np.random.RandomState(seed)  # noqa: NPY002 - frozen legacy stream for reproducibility
    s = np.linspace(0.0, 1.0, n_frames)
    angle = np.deg2rad(angle_start_deg + (angle_end_deg - angle_start_deg) * s)
    r = radius + ((radius_end if radius_end is not None else radius) - radius) * s
    h = height + ((height_end if height_end is not None else height) - height) * s
    eye = np.stack([target[0] + r * np.cos(angle), h, target[2] + r * np.sin(angle)], axis=1)
    aim = np.tile(np.array(target, dtype=np.float64), (n_frames, 1))
    if shake > 0:
        eye += shake * np.stack([_smooth_noise(n_frames, rng) for _ in range(3)], axis=1)
        aim += 0.5 * shake * np.stack([_smooth_noise(n_frames, rng) for _ in range(3)], axis=1)
    eye_t = torch.as_tensor(eye, dtype=DTYPE)
    aim_t = torch.as_tensor(aim, dtype=DTYPE)
    return torch.stack([look_at(eye_t[i], aim_t[i]) for i in range(n_frames)])


def fixed_camera(
    target: tuple[float, float, float], radius: float, height: float, angle_deg: float
) -> Tensor:
    """Camera-to-world pose ``(4, 4)`` of a static camera looking at ``target``."""
    angle = math.radians(angle_deg)
    eye = torch.tensor(
        [target[0] + radius * math.cos(angle), height, target[2] + radius * math.sin(angle)],
        dtype=DTYPE,
    )
    return look_at(eye, torch.tensor(target, dtype=DTYPE))
