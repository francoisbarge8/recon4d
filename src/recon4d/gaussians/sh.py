"""Real spherical harmonics for view-dependent colour (degrees 0 to 3).

The basis and sign conventions are those of the reference 3D Gaussian Splatting code, so
coefficients are interchangeable with ``.ply`` files produced by other implementations.
"""

from __future__ import annotations

import torch
from torch import Tensor

SH_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

MAX_SH_DEGREE = 3


def num_sh_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def sh_basis(degree: int, dirs: Tensor) -> Tensor:
    """Evaluate the real SH basis at unit directions ``(N, 3)``; returns ``(N, (degree+1)^2)``."""
    if not 0 <= degree <= MAX_SH_DEGREE:
        raise ValueError(f"SH degree must be in [0, {MAX_SH_DEGREE}], got {degree}")
    x, y, z = dirs.unbind(-1)
    basis = [torch.full_like(x, SH_C0)]
    if degree >= 1:
        basis += [-_C1 * y, _C1 * z, -_C1 * x]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        basis += [
            _C2[0] * xy,
            _C2[1] * yz,
            _C2[2] * (2.0 * zz - xx - yy),
            _C2[3] * xz,
            _C2[4] * (xx - yy),
        ]
    if degree >= 3:
        basis += [
            _C3[0] * y * (3.0 * xx - yy),
            _C3[1] * xy * z,
            _C3[2] * y * (4.0 * zz - xx - yy),
            _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            _C3[4] * x * (4.0 * zz - xx - yy),
            _C3[5] * z * (xx - yy),
            _C3[6] * x * (xx - 3.0 * yy),
        ]
    return torch.stack(basis, dim=-1)


def eval_sh(degree: int, coeffs: Tensor, dirs: Tensor) -> Tensor:
    """Colours ``(N, 3)`` from SH coefficients ``(N, K, 3)`` seen along ``dirs (N, 3)``.

    Follows the 3DGS convention: the SH expansion models ``colour - 0.5`` and the result is
    clamped at zero. Only the first ``(degree + 1)^2`` coefficients are used, which lets
    the active degree grow during optimisation.
    """
    k = num_sh_coeffs(degree)
    basis = sh_basis(degree, dirs)
    color = (basis[..., None] * coeffs[:, :k]).sum(dim=1)
    return (color + 0.5).clamp_min(0.0)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Degree-0 coefficient reproducing a view-independent colour."""
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh_dc: Tensor) -> Tensor:
    return sh_dc * SH_C0 + 0.5
