"""Rotation parametrisations.

Conventions
-----------
* Quaternions are ``(w, x, y, z)``, real part first (the 3DGS convention).
* Rotation matrices act on column vectors: ``x' = R @ x``.
* so(3) vectors are ``axis * angle`` in radians.

Every function broadcasts over leading dimensions and is differentiable,
including at the identity (small-angle branches use Taylor expansions).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_SMALL_ANGLE = 1e-4


def skew(v: Tensor) -> Tensor:
    """Cross-product matrix ``[v]_x`` such that ``skew(v) @ u == cross(v, u)``."""
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).reshape(*v.shape[:-1], 3, 3)


def normalize_quat(q: Tensor, eps: float = 1e-12) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Unit (or un-normalised) quaternion ``(..., 4)`` to rotation matrix ``(..., 3, 3)``."""
    w, x, y, z = normalize_quat(q).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def rotmat_to_quat(R: Tensor) -> Tensor:
    """Rotation matrix ``(..., 3, 3)`` to unit quaternion ``(..., 4)`` with ``w >= 0``.

    Uses the numerically stable four-branch construction: each branch divides by the
    largest of ``|w|, |x|, |y|, |z|`` so no branch ever divides by a small number.
    """
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    q_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    )
    # sqrt with a zero (instead of infinite) gradient at 0: the unused branches would
    # otherwise poison the backward pass with 0 * inf = nan.
    positive = q_abs > 0
    q_abs = torch.where(
        positive, torch.sqrt(torch.where(positive, q_abs, torch.ones_like(q_abs))), 0.0
    )
    candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., None].clamp_min(0.1))
    best = q_abs.argmax(dim=-1)
    index = best[..., None, None].expand(*best.shape, 1, 4)
    q = torch.gather(candidates, -2, index).squeeze(-2)
    q = torch.where(q[..., :1] < 0, -q, q)
    return normalize_quat(q)


def quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product ``a * b`` (the rotation ``b`` followed by ``a``)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def so3_exp(omega: Tensor) -> Tensor:
    """Rodrigues' formula: axis-angle ``(..., 3)`` to rotation matrix ``(..., 3, 3)``."""
    theta2 = (omega * omega).sum(dim=-1)
    small = theta2 < _SMALL_ANGLE**2
    theta = torch.sqrt(torch.where(small, torch.ones_like(theta2), theta2))
    a = torch.where(small, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / (theta * theta))
    K = skew(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(K)
    return eye + a[..., None, None] * K + b[..., None, None] * (K @ K)


def so3_log(R: Tensor) -> Tensor:
    """Rotation matrix ``(..., 3, 3)`` to axis-angle ``(..., 3)`` with angle in ``[0, pi]``."""
    q = rotmat_to_quat(R)
    w, v = q[..., 0], q[..., 1:]
    n2 = (v * v).sum(dim=-1)
    small = n2 < _SMALL_ANGLE**2
    n = torch.sqrt(torch.where(small, torch.ones_like(n2), n2))
    # theta / |v|, with theta = 2 atan2(|v|, w); the limit at |v| -> 0 is 2 / w.
    factor = torch.where(small, 2.0 / w.clamp_min(1e-6), 2.0 * torch.atan2(n, w) / n)
    return factor[..., None] * v


def rot6d_to_rotmat(d6: Tensor) -> Tensor:
    """Continuous 6D rotation representation (Zhou et al., CVPR 2019) to a rotation matrix.

    ``d6`` holds the first two *columns* of the matrix; Gram-Schmidt recovers the third.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotmat_to_rot6d(R: Tensor) -> Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rotation_angle(R: Tensor) -> Tensor:
    """Geodesic angle (radians) of a rotation matrix, stable near 0 and pi."""
    # |R - R^T|_F = 2 sqrt(2) sin(theta) and trace = 1 + 2 cos(theta): atan2 uses both.
    sin = (R - R.transpose(-1, -2)).flatten(-2).norm(dim=-1) / (2.0 * 2.0**0.5)
    cos = (R.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
    return torch.atan2(sin, cos)
