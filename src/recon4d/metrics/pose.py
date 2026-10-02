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


def align_frames(est_c2w: Tensor, gt_c2w: Tensor, lever: float | None = None) -> Sim3:
    """Similarity between two world frames estimated from full camera poses.

    :func:`align_trajectory` only uses the camera *positions*. For the nearly straight
    camera paths of casual videos this leaves the rotation about the direction of travel
    poorly determined, and a tilt of one degree displaces a scene five metres away by
    almost ten centimetres. Whenever the alignment is used to compare geometry or to place
    a held-out camera, orientations must constrain it too.

    Each camera therefore contributes four points: its centre and three points at distance
    ``lever`` along its axes. The scale is initialised from the centres and refined in a
    few fixed-point iterations (the lever must be expressed in each frame's own unit).

    Args:
        lever: lever arm in ground-truth units (default: the spatial extent of the
            ground-truth trajectory, at least 1).
    """
    est = est_c2w.to(torch.float64)
    gt = gt_c2w.to(torch.float64)
    centers_est, centers_gt = est[:, :3, 3], gt[:, :3, 3]
    sim = umeyama(centers_est, centers_gt)
    if lever is None:
        extent = (centers_gt.amax(dim=0) - centers_gt.amin(dim=0)).norm()
        lever = float(extent.clamp_min(1.0))
    # Columns of a camera-to-world rotation are the camera axes in world coordinates.
    axes_est = est[:, :3, :3].transpose(1, 2)  # (T, 3 axes, 3)
    axes_gt = gt[:, :3, :3].transpose(1, 2)
    target = torch.cat([centers_gt, (centers_gt[:, None] + lever * axes_gt).reshape(-1, 3)])
    for _ in range(5):
        arm = lever / sim.scale
        source = torch.cat([centers_est, (centers_est[:, None] + arm * axes_est).reshape(-1, 3)])
        sim = umeyama(source, target)
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


def orientation_errors(est_c2w: Tensor, gt_c2w: Tensor) -> Tensor:
    """Per-frame orientation error (radians) after the best global rotation alignment.

    The rotation between the two world frames is estimated from the orientations
    themselves, as the chordal mean of ``R_gt R_est^T``. Deriving it from the camera
    *positions* instead (as the ATE alignment does) is ill-conditioned for the nearly
    straight camera paths of casual videos, where a rotation about the direction of travel
    barely moves the positions.
    """
    R_est, R_gt = est_c2w[:, :3, :3], gt_c2w[:, :3, :3]
    U, _, Vh = torch.linalg.svd((R_gt @ R_est.transpose(1, 2)).sum(dim=0))
    d = torch.ones(3, dtype=R_est.dtype)
    d[2] = torch.sign(torch.det(U @ Vh))
    alignment = U @ torch.diag(d) @ Vh
    return rotation_angle(R_gt.transpose(1, 2) @ (alignment @ R_est))


def pose_metrics(est_c2w: Tensor, gt_c2w: Tensor, with_scale: bool = True) -> dict[str, float]:
    """All trajectory metrics as floats.

    * ``ate``: absolute trajectory error (RMSE, ground-truth units);
    * ``rpe_trans``, ``rpe_rot_deg``: frame-to-frame relative pose error;
    * ``rot_deg``: mean absolute orientation error (degrees), see
      :func:`orientation_errors`;
    * ``scale``: the alignment scale (ground-truth units per estimated unit).
    """
    est = est_c2w.to(torch.float64)
    gt = gt_c2w.to(torch.float64)
    sim = align_trajectory(est, gt, with_scale)
    aligned = sim.apply_to_c2w(est)
    position_error = (aligned[:, :3, 3] - gt[:, :3, 3]).norm(dim=-1)
    rotation_error = orientation_errors(est, gt)
    rpe_trans, rpe_rot = relative_pose_error(est, gt, 1, sim.scale)
    return {
        "ate": float(torch.sqrt((position_error**2).mean())),
        "rpe_trans": float(rpe_trans),
        "rpe_rot_deg": float(rpe_rot),
        "rot_deg": float(torch.rad2deg(rotation_error).mean()),
        "scale": float(sim.scale),
    }
