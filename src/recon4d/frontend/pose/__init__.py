"""Camera pose estimation: track-based structure-from-motion and bundle adjustment."""

from recon4d.frontend.pose.bundle_adjustment import (
    BAProblem,
    BAResult,
    bundle_adjust,
    projection_jacobians,
    reprojection_residuals,
)
from recon4d.frontend.pose.sfm import (
    SfMConfig,
    SfMResult,
    baseline_ratio,
    reconstruct,
    triangulate_with_poses,
)

__all__ = [
    "BAProblem",
    "BAResult",
    "SfMConfig",
    "SfMResult",
    "baseline_ratio",
    "bundle_adjust",
    "projection_jacobians",
    "reconstruct",
    "reprojection_residuals",
    "triangulate_with_poses",
]
