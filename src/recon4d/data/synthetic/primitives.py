"""Analytic primitives, their ray intersections, surface sampling and motion models.

Every primitive lives in a canonical local frame (unit sphere, cube ``[-1, 1]^3``, unit
cylinder along ``y``) and is placed in the world by a time-dependent :class:`Pose`
``x_world = R @ (scale * x_local) + center``. A point of the surface is identified by its
local coordinates, which is what makes exact ground-truth point tracks trivial: transport
is just re-evaluating the pose at another time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from recon4d.geometry.rotations import so3_exp

DTYPE = torch.float64
_INF = float("inf")
_T_MIN = 1e-6

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Pose:
    """Placement of a primitive: ``x_world = R @ (scale * x_local) + center``."""

    R: Tensor
    center: Tensor
    scale: Tensor

    def to_world(self, local: Tensor) -> Tensor:
        return (local * self.scale) @ self.R.transpose(0, 1) + self.center

    def to_local(self, world: Tensor) -> Tensor:
        return ((world - self.center) @ self.R) / self.scale

    def dir_to_local(self, direction: Tensor) -> Tensor:
        return (direction @ self.R) / self.scale

    def normal_to_world(self, normal_local: Tensor) -> Tensor:
        # Normals transform with the inverse transpose of the linear map R @ diag(scale).
        n = (normal_local / self.scale) @ self.R.transpose(0, 1)
        return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _vec(v: Vec3 | Tensor) -> Tensor:
    return torch.as_tensor(v, dtype=DTYPE)


def _rot_y(angle: float) -> Tensor:
    return so3_exp(torch.tensor([0.0, angle, 0.0], dtype=DTYPE))


def _ease(t: float) -> float:
    """Smooth 0 -> 1 ramp with zero velocity at both ends."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))


# ----------------------------------------------------------------------------- motions


class Motion:
    """Time-dependent pose of a primitive; ``t`` is normalised time in ``[0, 1]``."""

    is_dynamic: bool = True

    def pose(self, t: float) -> Pose:
        raise NotImplementedError


@dataclass(frozen=True)
class Static(Motion):
    center: Vec3
    scale: Vec3
    rotvec: Vec3 = (0.0, 0.0, 0.0)
    is_dynamic = False

    def pose(self, t: float) -> Pose:
        return Pose(so3_exp(_vec(self.rotvec)), _vec(self.center), _vec(self.scale))


@dataclass(frozen=True)
class RollOnCircle(Motion):
    """A ball rolling without slipping along a circular path on the floor.

    The ball of radius ``radius`` has its centre at height ``radius`` and travels from
    polar angle ``angle_start`` to ``angle_end`` (radians) on a circle of radius
    ``path_radius`` around ``path_center`` (x, z). ``ease`` selects a smooth start/stop.
    """

    path_center: tuple[float, float]
    path_radius: float
    radius: float
    angle_start: float
    angle_end: float
    ease: bool = False

    def angle(self, t: float) -> float:
        s = _ease(t) if self.ease else t
        return self.angle_start + (self.angle_end - self.angle_start) * s

    def pose(self, t: float) -> Pose:
        phi = self.angle(t)
        delta = phi - self.angle_start
        center = torch.tensor(
            [
                self.path_center[0] + self.path_radius * math.cos(phi),
                self.radius,
                self.path_center[1] + self.path_radius * math.sin(phi),
            ],
            dtype=DTYPE,
        )
        # Rolling without slipping: the angular velocity is (path_radius / radius) along the
        # radial direction e_r(phi), which itself turns about the vertical. In the frame
        # co-rotating with e_r the angular velocity is constant, giving the closed form
        #   R(phi) = Rot_y(-delta) @ exp(delta * [ (path_radius / radius) e_r(phi_0) + e_y ]_x).
        e_r0 = torch.tensor(
            [math.cos(self.angle_start), 0.0, math.sin(self.angle_start)], dtype=DTYPE
        )
        axis = (self.path_radius / self.radius) * e_r0 + torch.tensor([0.0, 1.0, 0.0], dtype=DTYPE)
        R = _rot_y(-delta) @ so3_exp(delta * axis)
        return Pose(R, center, torch.full((3,), self.radius, dtype=DTYPE))


@dataclass(frozen=True)
class SlideSpin(Motion):
    """Back-and-forth slide between two floor points combined with a spin about the vertical."""

    start: Vec3
    end: Vec3
    scale: Vec3
    cycles: float = 0.5
    yaw_start: float = 0.0
    yaw_total: float = math.pi

    def pose(self, t: float) -> Pose:
        s = 0.5 - 0.5 * math.cos(2.0 * math.pi * self.cycles * t)
        center = _vec(self.start) + s * (_vec(self.end) - _vec(self.start))
        return Pose(_rot_y(self.yaw_start + self.yaw_total * t), center, _vec(self.scale))


@dataclass(frozen=True)
class Bounce(Motion):
    """Ballistic hops along a straight ground track, with a constant tumbling rate."""

    start: tuple[float, float]
    end: tuple[float, float]
    radius: float
    height: float
    bounces: float = 2.0
    tumble: Vec3 = (0.0, 0.0, 0.0)
    """Total rotation (axis * angle) accumulated over the sequence."""

    def pose(self, t: float) -> Pose:
        x = self.start[0] + t * (self.end[0] - self.start[0])
        z = self.start[1] + t * (self.end[1] - self.start[1])
        phase = (self.bounces * t) % 1.0
        y = self.radius + self.height * 4.0 * phase * (1.0 - phase)  # parabola per hop
        center = torch.tensor([x, y, z], dtype=DTYPE)
        return Pose(
            so3_exp(t * _vec(self.tumble)), center, torch.full((3,), self.radius, dtype=DTYPE)
        )


@dataclass(frozen=True)
class Squash(Motion):
    """Volume-preserving squash-and-stretch of an object resting on the floor (non-rigid)."""

    base_center: tuple[float, float]
    scale: Vec3
    amplitude: float = 0.3
    cycles: float = 1.5
    sway: float = 0.0
    """Peak horizontal displacement along x, to combine deformation with translation."""

    def pose(self, t: float) -> Pose:
        stretch = 1.0 + self.amplitude * math.sin(2.0 * math.pi * self.cycles * t)
        lateral = 1.0 / math.sqrt(stretch)
        sx, sy, sz = self.scale
        scale = torch.tensor([sx * lateral, sy * stretch, sz * lateral], dtype=DTYPE)
        x = self.base_center[0] + self.sway * math.sin(2.0 * math.pi * 0.5 * self.cycles * t)
        center = torch.tensor([x, sy * stretch, self.base_center[1]], dtype=DTYPE)
        return Pose(torch.eye(3, dtype=DTYPE), center, scale)


# ----------------------------------------------------------------------- intersections


def _safe_div(num: Tensor, den: Tensor) -> Tensor:
    den = torch.where(den.abs() < 1e-14, torch.full_like(den, 1e-14), den)
    return num / den


def intersect_sphere(o: Tensor, d: Tensor) -> tuple[Tensor, Tensor]:
    """Nearest hit of rays ``o + t d`` with the unit sphere. Returns ``(t, local_normal)``."""
    a = (d * d).sum(dim=-1)
    b = 2.0 * (o * d).sum(dim=-1)
    c = (o * o).sum(dim=-1) - 1.0
    disc = b * b - 4.0 * a * c
    root = torch.sqrt(disc.clamp_min(0.0))
    t_near = _safe_div(-b - root, 2.0 * a)
    t_far = _safe_div(-b + root, 2.0 * a)
    t = torch.where(t_near > _T_MIN, t_near, t_far)
    hit = (disc >= 0.0) & (t > _T_MIN)
    t = torch.where(hit, t, torch.full_like(t, _INF))
    normal = o + torch.where(hit, t, torch.zeros_like(t))[:, None] * d
    return t, normal


def intersect_box(o: Tensor, d: Tensor) -> tuple[Tensor, Tensor]:
    """Nearest hit of rays with the cube ``[-1, 1]^3`` (slab method)."""
    t1 = _safe_div(-1.0 - o, d)
    t2 = _safe_div(1.0 - o, d)
    t_lo = torch.minimum(t1, t2)
    t_hi = torch.maximum(t1, t2)
    t_enter, axis_enter = t_lo.max(dim=-1)
    t_exit, axis_exit = t_hi.min(dim=-1)
    outside = t_enter > _T_MIN
    hit = (t_exit >= t_enter) & (t_exit > _T_MIN)
    t = torch.where(outside, t_enter, t_exit)
    axis = torch.where(outside, axis_enter, axis_exit)
    t = torch.where(hit, t, torch.full_like(t, _INF))
    # Entering faces oppose the ray direction; exiting faces (ray origin inside) follow it.
    sign = -torch.sign(torch.gather(d, 1, axis[:, None]))[:, 0]
    sign = torch.where(outside, sign, -sign)
    normal = torch.zeros_like(o)
    normal.scatter_(1, axis[:, None], sign[:, None])
    return t, normal


def intersect_cylinder(o: Tensor, d: Tensor) -> tuple[Tensor, Tensor]:
    """Nearest hit of rays with the capped cylinder ``x^2 + z^2 <= 1, |y| <= 1``."""
    ox, oy, oz = o.unbind(-1)
    dx, dy, dz = d.unbind(-1)
    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - 1.0
    disc = b * b - 4.0 * a * c
    root = torch.sqrt(disc.clamp_min(0.0))
    best_t = torch.full_like(ox, _INF)
    best_n = torch.zeros_like(o)
    for t_side in (_safe_div(-b - root, 2.0 * a), _safe_div(-b + root, 2.0 * a)):
        y = oy + t_side * dy
        ok = (disc >= 0.0) & (a > 1e-14) & (t_side > _T_MIN) & (y.abs() <= 1.0) & (t_side < best_t)
        n = torch.stack([ox + t_side * dx, torch.zeros_like(ox), oz + t_side * dz], dim=-1)
        best_t = torch.where(ok, t_side, best_t)
        best_n = torch.where(ok[:, None], n, best_n)
    for cap in (-1.0, 1.0):
        t_cap = _safe_div(cap - oy, dy)
        x = ox + t_cap * dx
        z = oz + t_cap * dz
        ok = (t_cap > _T_MIN) & (x * x + z * z <= 1.0) & (t_cap < best_t)
        n = torch.zeros_like(o)
        n[:, 1] = cap
        best_t = torch.where(ok, t_cap, best_t)
        best_n = torch.where(ok[:, None], n, best_n)
    return best_t, best_n


INTERSECT = {"sphere": intersect_sphere, "box": intersect_box, "cylinder": intersect_cylinder}


# -------------------------------------------------------------------- surface sampling


def _sample_local(kind: str, n: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Points and unit normals distributed uniformly w.r.t. *local* surface area."""
    if kind == "sphere":
        p = rng.standard_normal((n, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        return p, p.copy()
    if kind == "box":
        axis = rng.randint(0, 3, size=n)
        sign = rng.choice([-1.0, 1.0], size=n)
        p = rng.uniform(-1.0, 1.0, size=(n, 3))
        normal = np.zeros((n, 3))
        p[np.arange(n), axis] = sign
        normal[np.arange(n), axis] = sign
        return p, normal
    if kind == "cylinder":
        # Local areas: side 2*pi*1*2 = 4*pi, each cap pi.
        region = rng.choice(3, size=n, p=[4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        radius = np.sqrt(rng.uniform(0.0, 1.0, size=n))
        height = rng.uniform(-1.0, 1.0, size=n)
        side = region == 0
        cap_sign = np.where(region == 1, -1.0, 1.0)
        r = np.where(side, 1.0, radius)
        p = np.stack([r * np.cos(theta), np.where(side, height, cap_sign), r * np.sin(theta)], 1)
        normal = np.stack(
            [
                np.where(side, np.cos(theta), 0.0),
                np.where(side, 0.0, cap_sign),
                np.where(side, np.sin(theta), 0.0),
            ],
            axis=1,
        )
        return p, normal
    raise ValueError(f"unknown primitive kind {kind!r}")


def sample_surface(
    kind: str, scale: Tensor, n: int, rng: np.random.RandomState
) -> tuple[Tensor, Tensor]:
    """Sample ``n`` surface points uniformly w.r.t. *world* area for a scaled primitive.

    Rejection sampling: candidates drawn uniformly in the local frame are accepted with
    probability proportional to the local-to-world area ratio ``det(S) * |S^-1 n|``.

    Returns local coordinates and local unit normals, both ``(n, 3)``.
    """
    s = scale.detach().cpu().numpy()
    stretch_max = np.prod(s) / s.min()
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    count = 0
    while count < n:
        candidates, candidate_normals = _sample_local(kind, max(2 * (n - count), 256), rng)
        stretch = np.prod(s) * np.linalg.norm(candidate_normals / s, axis=1)
        keep = rng.uniform(size=stretch.shape[0]) * stretch_max <= stretch
        points.append(candidates[keep])
        normals.append(candidate_normals[keep])
        count += int(keep.sum())
    return (
        torch.as_tensor(np.concatenate(points)[:n], dtype=DTYPE),
        torch.as_tensor(np.concatenate(normals)[:n], dtype=DTYPE),
    )


def surface_area(kind: str, scale: Tensor, n_estimate: int = 20000) -> float:
    """World-space surface area of a scaled primitive (exact for boxes, Monte Carlo otherwise)."""
    s = scale.detach().cpu().numpy()
    if kind == "box":
        return float(8.0 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2]))
    local_area = {"sphere": 4.0 * np.pi, "cylinder": 6.0 * np.pi}[kind]
    rng = np.random.RandomState(0)  # noqa: NPY002 - frozen legacy stream for reproducibility
    _, normals = _sample_local(kind, n_estimate, rng)
    stretch = np.prod(s) * np.linalg.norm(normals / s, axis=1)
    return float(local_area * stretch.mean())
