"""Pinhole camera model and rigid transforms.

Conventions
-----------
* Camera frame is OpenCV / COLMAP: ``+x`` right, ``+y`` down, ``+z`` forward.
* ``w2c`` is a 4x4 world-to-camera matrix, ``x_cam = R @ x_world + t``; ``c2w`` is its inverse.
* Image coordinates are continuous with **pixel centres at half-integers**: pixel
  ``(col, row)`` covers ``[col, col + 1) x [row, row + 1)`` and its centre is
  ``(col + 0.5, row + 0.5)``. This is COLMAP's convention and matches
  ``grid_sample(..., align_corners=False)``. OpenCV functions use integer centres;
  the 0.5 shift is applied where those are called.
* Depth always means z-depth (distance along the optical axis), not ray length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of an image of size ``width x height``."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_x_deg: float) -> Intrinsics:
        f = 0.5 * width / torch.tan(torch.deg2rad(torch.tensor(fov_x_deg / 2.0))).item()
        return cls(f, f, width / 2.0, height / 2.0, width, height)

    @classmethod
    def from_matrix(cls, K: Tensor, width: int, height: int) -> Intrinsics:
        return cls(
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), width, height
        )

    def matrix(self, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=dtype,
            device=device,
        )

    def scaled(self, factor: float) -> Intrinsics:
        """Intrinsics of the same camera after resizing the image by ``factor``."""
        return Intrinsics(
            self.fx * factor,
            self.fy * factor,
            self.cx * factor,
            self.cy * factor,
            round(self.width * factor),
            round(self.height * factor),
        )


def pixel_centers(height: int, width: int, device=None, dtype=torch.float32) -> Tensor:
    """``(H, W, 2)`` grid of pixel-centre coordinates ``(x, y)``."""
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)


def project(K: Tensor, points_cam: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Project camera-frame points ``(..., 3)`` to pixels ``(..., 2)``; also returns z-depth."""
    z = points_cam[..., 2]
    z_safe = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
    u = K[..., 0, 0] * points_cam[..., 0] / z_safe + K[..., 0, 2]
    v = K[..., 1, 1] * points_cam[..., 1] / z_safe + K[..., 1, 2]
    return torch.stack([u, v], dim=-1), z


def unproject(K: Tensor, uv: Tensor, depth: Tensor) -> Tensor:
    """Lift pixels ``(..., 2)`` with z-depth ``(...)`` to camera-frame points ``(..., 3)``."""
    x = (uv[..., 0] - K[..., 0, 2]) / K[..., 0, 0] * depth
    y = (uv[..., 1] - K[..., 1, 2]) / K[..., 1, 1] * depth
    return torch.stack([x, y, depth], dim=-1)


def make_se3(R: Tensor, t: Tensor) -> Tensor:
    """Assemble ``(..., 4, 4)`` homogeneous transforms from ``R (..., 3, 3)`` and ``t (..., 3)``."""
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def invert_se3(T: Tensor) -> Tensor:
    """Closed-form inverse of rigid transforms ``(..., 4, 4)``."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    return make_se3(R_inv, -(R_inv @ t[..., None])[..., 0])


def transform_points(T: Tensor, points: Tensor) -> Tensor:
    """Apply ``T (..., 4, 4)`` to points ``(..., N, 3)``."""
    return points @ T[..., :3, :3].transpose(-1, -2) + T[..., None, :3, 3]


def camera_centers(w2c: Tensor) -> Tensor:
    """Camera positions in world coordinates from world-to-camera matrices."""
    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3]
    return -(R.transpose(-1, -2) @ t[..., None])[..., 0]


def look_at(eye: Tensor, target: Tensor, up: Tensor | None = None) -> Tensor:
    """Camera-to-world matrix of a camera at ``eye`` looking at ``target``.

    ``up`` is the world up direction (default ``+y``); the camera's ``+y`` axis points down.
    """
    if up is None:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=eye.dtype, device=eye.device)
    forward = F.normalize(target - eye, dim=-1)
    right = F.normalize(torch.cross(forward, up.expand_as(forward), dim=-1), dim=-1)
    down = torch.cross(forward, right, dim=-1)
    R = torch.stack([right, down, forward], dim=-1)
    return make_se3(R, eye)


def pixel_rays(K: Tensor, c2w: Tensor, uv: Tensor) -> tuple[Tensor, Tensor]:
    """World-space rays through pixels ``uv (..., 2)``.

    Returns ``(origins, directions)``. Directions are scaled so that their component
    along the optical axis is 1, hence ``origin + depth * direction`` is the 3D point
    at z-depth ``depth``.
    """
    dirs_cam = unproject(K, uv, torch.ones_like(uv[..., 0]))
    dirs = dirs_cam @ c2w[..., :3, :3].transpose(-1, -2)
    origins = c2w[..., :3, 3].expand_as(dirs)
    return origins, dirs


def backproject_depth(K: Tensor, c2w: Tensor, depth: Tensor) -> Tensor:
    """Back-project a depth map ``(H, W)`` to world-space points ``(H, W, 3)``."""
    height, width = depth.shape[-2:]
    uv = pixel_centers(height, width, device=depth.device, dtype=depth.dtype)
    return transform_points(c2w, unproject(K, uv, depth).reshape(-1, 3)).reshape(height, width, 3)


def sample_bilinear(image: Tensor, uv: Tensor, mode: str = "bilinear") -> Tensor:
    """Sample ``image (C, H, W)`` at continuous pixel coordinates ``uv (N, 2)``.

    Returns ``(N, C)``. Coordinates outside the image are clamped to the border.
    """
    _, height, width = image.shape
    size = torch.tensor([width, height], dtype=uv.dtype, device=uv.device)
    grid = (uv / size * 2.0 - 1.0).reshape(1, 1, -1, 2)
    out = F.grid_sample(
        image[None], grid.to(image.dtype), mode=mode, padding_mode="border", align_corners=False
    )
    return out[0, :, 0].transpose(0, 1)


def in_image(uv: Tensor, height: int, width: int, margin: float = 0.0) -> Tensor:
    """Boolean mask of pixel coordinates lying inside the image (shrunk by ``margin``)."""
    return (
        (uv[..., 0] >= margin)
        & (uv[..., 0] <= width - margin)
        & (uv[..., 1] >= margin)
        & (uv[..., 1] <= height - margin)
    )
