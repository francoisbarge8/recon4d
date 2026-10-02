"""Differentiable Gaussian splatting: rasterizer, scene models and optimisation."""

from recon4d.gaussians.rasterizer import (
    Projection,
    RasterOutput,
    RasterSettings,
    covariance_3d,
    project_gaussians,
    rasterize,
)
from recon4d.gaussians.sh import eval_sh, num_sh_coeffs, rgb_to_sh, sh_basis, sh_to_rgb

__all__ = [
    "Projection",
    "RasterOutput",
    "RasterSettings",
    "covariance_3d",
    "eval_sh",
    "num_sh_coeffs",
    "project_gaussians",
    "rasterize",
    "rgb_to_sh",
    "sh_basis",
    "sh_to_rgb",
]
