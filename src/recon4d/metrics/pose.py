"""Camera trajectory metrics: ATE, RPE and rotation error.

A monocular reconstruction lives in an arbitrary similarity frame, so trajectories are
first aligned to the ground truth with the closed-form Sim(3) of Umeyama on the camera
centres (the convention of the TUM RGB-D benchmark and of the ``evo`` toolkit).
"""

from __future__ import annotations

import torch
from torch import Tensor

from recon4d.geometry.align import Sim3, umeyama
from recon4d.geometry.camera import invert_se3
from recon4d.geometry.rotations import rotation_angle


def align_trajectory(est_c2w: Tensor, gt_c2w: Tensor, with_scale: bool = True) -> Sim3:
    """Similarity mapping the estimated world frame onto the ground-truth one.

    Args:
        est_c2w, gt_c2w: ``(T, 4, 4)`` camera-to-world poses of the same frames.
    """
    sim = umeyama(
        est_c2w[:, :3, 3].to(torch.float64),
        gt_c2w[:, :3, 3].to(torch.float64),
        with_scale=with_scale,
    )
    return sim.to(est_c2w.dtype)


def absolute_trajectory_error(
    est_c2w: Tensor, gt_c2w: Tensor, with_scale: bool = True
) -> tuple[Tensor, Tensor]:
    """ATE: RMSE of the camera positions after alignment.

    Returns the RMSE and the aligned estimated poses ``(T, 4, 4)``.
    """
    sim = align_trajectory(est_c2w, gt_c2w, with_scale)
    aligned = sim.apply_to_c2w(est_c2w)
    errors = (aligned[:, :3, 3] - gt_c2w[:, :3, 3]).norm(dim=-1)
    return torch.sqrt((errors**2).mean()), aligned


def relative_pose_error(
    est_c2w: Tensor, gt_c2w: Tensor, delta: int = 1, scale: float | Tensor = 1.0
) -> tuple[Tensor, Tensor]:
    """RPE over a fixed frame offset ``delta`` (local drift, independent of alignment).

    For each ``i`` the relative motion ``inv(P_i) P_{i+delta}`` of the estimate is compared
    with the ground truth's. ``scale`` rescales the estimated translations first (pass the
    Sim(3) scale for monocular trajectories).

    Returns the RMSE of the translational part and of the rotational part (degrees).
    """
    est = est_c2w.clone()
    est[:, :3, 3] = est[:, :3, 3] * scale
    rel_est = invert_se3(est[:-delta]) @ est[delta:]
    rel_gt = invert_se3(gt_c2w[:-delta]) @ gt_c2w[delta:]
    error = invert_se3(rel_gt) @ rel_est
    trans = error[:, :3, 3].norm(dim=-1)
    rot = torch.rad2deg(rotation_angle(error[:, :3, :3]))
    return torch.sqrt((trans**2).mean()), torch.sqrt((rot**2).mean())


def pose_metrics(est_c2w: Tensor, gt_c2w: Tensor, with_scale: bool = True) -> dict[str, float]:
    """All trajectory metrics as floats.

    * ``ate``: absolute trajectory error (RMSE, ground-truth units);
    * ``rpe_trans``, ``rpe_rot_deg``: frame-to-frame relative pose error;
    * ``rot_deg``: mean absolute orientation error after alignment (degrees);
    * ``scale``: the alignment scale (ground-truth units per estimated unit).
    """
    est = est_c2w.to(torch.float64)
    gt = gt_c2w.to(torch.float64)
    sim = align_trajectory(est, gt, with_scale)
    aligned = sim.apply_to_c2w(est)
    position_error = (aligned[:, :3, 3] - gt[:, :3, 3]).norm(dim=-1)
    rotation_error = rotation_angle(gt[:, :3, :3].transpose(1, 2) @ aligned[:, :3, :3])
    rpe_trans, rpe_rot = relative_pose_error(est, gt, 1, sim.scale)
    return {
        "ate": float(torch.sqrt((position_error**2).mean())),
        "rpe_trans": float(rpe_trans),
        "rpe_rot_deg": float(rpe_rot),
        "rot_deg": float(torch.rad2deg(rotation_error).mean()),
        "scale": float(sim.scale),
    }
