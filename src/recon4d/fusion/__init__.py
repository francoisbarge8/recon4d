"""Fusion of per-frame depth into consistent 3D geometry."""

from recon4d.fusion.pointcloud import (
    backproject_frames,
    downsample_to,
    multiview_consistency,
    voxel_downsample,
)

__all__ = ["backproject_frames", "downsample_to", "multiview_consistency", "voxel_downsample"]
