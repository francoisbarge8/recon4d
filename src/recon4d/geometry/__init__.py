"""Geometry primitives: rotations, the pinhole camera model and alignment.

See :mod:`recon4d.geometry.camera` for the coordinate conventions used throughout.
"""

from recon4d.geometry.align import Sim3, batched_kabsch, kabsch, umeyama
from recon4d.geometry.camera import (
    Intrinsics,
    backproject_depth,
    camera_centers,
    in_image,
    invert_se3,
    look_at,
    make_se3,
    pixel_centers,
    pixel_rays,
    project,
    sample_bilinear,
    sample_depth,
    transform_points,
    unproject,
)
from recon4d.geometry.rotations import (
    normalize_quat,
    quat_multiply,
    quat_to_rotmat,
    rot6d_to_rotmat,
    rotation_angle,
    rotmat_to_quat,
    rotmat_to_rot6d,
    skew,
    so3_exp,
    so3_log,
)
from recon4d.geometry.triangulation import (
    reprojection_errors,
    triangulate_dlt,
    triangulation_angles,
)

__all__ = [
    "Intrinsics",
    "Sim3",
    "backproject_depth",
    "batched_kabsch",
    "camera_centers",
    "in_image",
    "invert_se3",
    "kabsch",
    "look_at",
    "make_se3",
    "normalize_quat",
    "pixel_centers",
    "pixel_rays",
    "project",
    "quat_multiply",
    "quat_to_rotmat",
    "reprojection_errors",
    "rot6d_to_rotmat",
    "rotation_angle",
    "rotmat_to_quat",
    "rotmat_to_rot6d",
    "sample_bilinear",
    "sample_depth",
    "skew",
    "so3_exp",
    "so3_log",
    "transform_points",
    "triangulate_dlt",
    "triangulation_angles",
    "umeyama",
    "unproject",
]
