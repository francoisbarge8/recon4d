"""Evaluation metrics for images, geometry, camera poses, tracks and temporal stability."""

from recon4d.metrics.depth import align_depth, depth_metrics
from recon4d.metrics.geometry import (
    chamfer_distance,
    f_score,
    geometry_metrics,
    nearest_distances,
    nearest_distances_torch,
)
from recon4d.metrics.image import LPIPS, masked_ssim_map, psnr, ssim, ssim_map
from recon4d.metrics.pose import (
    absolute_trajectory_error,
    align_trajectory,
    pose_metrics,
    relative_pose_error,
)
from recon4d.metrics.temporal import (
    depth_temporal_error,
    temporal_difference_psnr,
    warp_backward,
    warping_error,
)
from recon4d.metrics.tracking import tapvid_metrics, trajectory_error_3d

__all__ = [
    "LPIPS",
    "absolute_trajectory_error",
    "align_depth",
    "align_trajectory",
    "chamfer_distance",
    "depth_metrics",
    "depth_temporal_error",
    "f_score",
    "geometry_metrics",
    "masked_ssim_map",
    "nearest_distances",
    "nearest_distances_torch",
    "pose_metrics",
    "psnr",
    "relative_pose_error",
    "ssim",
    "ssim_map",
    "tapvid_metrics",
    "temporal_difference_psnr",
    "trajectory_error_3d",
    "warp_backward",
    "warping_error",
]
