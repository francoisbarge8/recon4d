"""Fusion of per-frame depth into consistent 3D geometry."""

from recon4d.fusion.pointcloud import (
    backproject_frames,
    downsample_to,
    multiview_consistency,
    voxel_downsample,
)
from recon4d.fusion.tsdf import TSDFConfig, TSDFVolume, fuse_depth_maps, volume_bounds

__all__ = [
    "TSDFConfig",
    "TSDFVolume",
    "backproject_frames",
    "downsample_to",
    "fuse_depth_maps",
    "multiview_consistency",
    "volume_bounds",
    "voxel_downsample",
]
