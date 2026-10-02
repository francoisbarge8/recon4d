"""Structure-from-motion from point tracks.

Point tracks give correspondences across the whole video for free, so no feature matching
is needed. The reconstruction is incremental:

1. **Two-view initialisation.** Among frame pairs with enough parallax, the essential
   matrix is estimated with RANSAC and decomposed; the pair with the largest number of
   inliers triangulated at a healthy angle starts the map.
2. **Registration.** Each remaining frame, taken in order of temporal distance to the
   seed, is localised with PnP + RANSAC against the current 3D points, new points are
   triangulated, and a local bundle adjustment keeps the drift in check.
3. **Global refinement.** Bundle adjustment over all cameras and points (optionally the
   focal length), alternated with outlier removal and re-triangulation.

RANSAC and the Huber loss make the estimate robust to tracking errors and to independently
moving objects, whose tracks violate the epipolar geometry of the static scene. Those
tracks are returned as outliers and are the seed of motion segmentation.

The minimal solvers (5-point essential matrix, P3P) are OpenCV's; bundle adjustment is the
PyTorch implementation of :mod:`recon4d.frontend.pose.bundle_adjustment`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor

from recon4d.frontend.pose.bundle_adjustment import BAProblem, bundle_adjust
from recon4d.geometry.camera import camera_centers, invert_se3
from recon4d.geometry.triangulation import (
    reprojection_errors,
    triangulate_dlt,
    triangulation_angles,
)
from recon4d.types import Tracks
from recon4d.utils import get_logger

logger = get_logger(__name__)
DTYPE = torch.float64


@dataclass
class SfMConfig:
    """Parameters of the track-based structure-from-motion.

    Attributes:
        ransac_threshold: inlier threshold (pixels) of the essential-matrix and PnP RANSAC.
        reproj_threshold: maximum mean reprojection error (pixels) of a valid 3D point.
        min_angle_deg: minimum triangulation angle of a valid 3D point.
        min_track_length: tracks observed in fewer frames are ignored.
        min_parallax: minimum median displacement (fraction of the image diagonal) between
            the two frames of the initial pair.
        local_ba_interval: a local bundle adjustment runs every this many registrations.
        max_ba_points: number of points used in bundle adjustment (the longest tracks are
            preferred); the others are re-triangulated afterwards.
        refine_focal: also optimise a shared focal length in the final bundle adjustment.
        huber_delta: Huber threshold (pixels) of bundle adjustment.
        seed: RANSAC seed (OpenCV's RNG).
    """

    ransac_threshold: float = 1.0
    reproj_threshold: float = 1.5
    min_angle_deg: float = 1.5
    min_track_length: int = 4
    min_parallax: float = 0.03
    local_ba_interval: int = 6
    max_ba_points: int = 2500
    refine_focal: bool = False
    huber_delta: float = 1.0
    seed: int = 0


@dataclass
class SfMResult:
    """Output of :func:`reconstruct`.

    The world frame is that of the first registered camera, with a scale fixed by
    normalising the median depth of the points to 1.
    """

    w2c: Tensor
    """``(T, 4, 4)`` world-to-camera poses (float64)."""
    registered: Tensor
    """``(T,)`` True for frames that were localised."""
    K: Tensor
    """``(3, 3)`` intrinsics (refined when requested)."""
    points: Tensor
    """``(N, 3)`` one 3D point per track (meaningful where ``valid``)."""
    valid: Tensor
    """``(N,)`` tracks successfully reconstructed as static points."""
    inlier: Tensor
    """``(N, T)`` observations consistent with their static point."""
    reproj_error: Tensor
    """``(N,)`` mean reprojection error over each track's visible frames, in pixels."""

    @property
    def c2w(self) -> Tensor:
        return invert_se3(self.w2c)


def _cv_points(uv: Tensor) -> np.ndarray:
    # OpenCV pixel centres are integers, ours are half-integers.
    return (uv - 0.5).cpu().numpy().astype(np.float64)


def _cv_K(K: Tensor) -> np.ndarray:
    K_cv = K.cpu().numpy().astype(np.float64).copy()
    K_cv[0, 2] -= 0.5
    K_cv[1, 2] -= 0.5
    return K_cv


def _two_view(
    K: Tensor, uv_a: Tensor, uv_b: Tensor, cfg: SfMConfig
) -> tuple[Tensor, Tensor] | None:
    """Relative pose ``(R, t)`` of camera b w.r.t. camera a and the RANSAC inlier mask."""
    if uv_a.shape[0] < 8:
        return None
    K_cv = _cv_K(K)
    pa, pb = _cv_points(uv_a), _cv_points(uv_b)
    E, mask = cv2.findEssentialMat(
        pa, pb, K_cv, method=cv2.RANSAC, prob=0.9999, threshold=cfg.ransac_threshold
    )
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, pose_mask = cv2.recoverPose(E, pa, pb, K_cv, mask=mask.copy())
    T = torch.eye(4, dtype=DTYPE)
    T[:3, :3] = torch.from_numpy(R)
    T[:3, 3] = torch.from_numpy(t[:, 0])
    return T, torch.from_numpy(pose_mask.reshape(-1) > 0)


def _select_initial_pair(
    tracks: Tracks, K: Tensor, cfg: SfMConfig
) -> tuple[int, int, Tensor, Tensor]:
    """Find the frame pair that best constrains the initial reconstruction.

    Returns ``(a, b, w2c_b, inliers)`` with camera ``a`` at the origin.
    """
    n_frames = tracks.num_frames
    diagonal = None
    best: tuple[float, int, int, Tensor, Tensor] | None = None
    # Candidate pairs: a few anchors, paired with frames at increasing temporal distance.
    anchors = sorted({0, n_frames // 3, n_frames // 2, 2 * n_frames // 3})
    gaps = sorted(
        {max(2, n_frames // 8), n_frames // 4, n_frames // 3, n_frames // 2, n_frames - 1}
    )
    for a in anchors:
        for gap in gaps:
            b = a + gap
            if b >= n_frames:
                continue
            common = tracks.visible[:, a] & tracks.visible[:, b]
            if common.sum() < 30:
                continue
            uv_a, uv_b = tracks.uv[common, a], tracks.uv[common, b]
            if diagonal is None:
                diagonal = float(torch.linalg.norm(2.0 * K[:2, 2]))
            parallax = float((uv_b - uv_a).norm(dim=-1).median()) / diagonal
            if parallax < cfg.min_parallax:
                continue
            result = _two_view(K, uv_a, uv_b, cfg)
            if result is None:
                continue
            w2c_b, inlier = result
            if inlier.sum() < 20:
                continue
            w2c = torch.stack([torch.eye(4, dtype=DTYPE), w2c_b])
            uv = torch.stack([uv_a[inlier], uv_b[inlier]], dim=1).to(DTYPE)
            both = torch.ones(uv.shape[0], 2, dtype=torch.bool)
            points, ok = triangulate_dlt(K.to(DTYPE), w2c, uv, both)
            angles = torch.rad2deg(triangulation_angles(w2c, points, both))
            good = ok & (angles > cfg.min_angle_deg)
            # Favour many well-conditioned points: count them, weighted by median angle.
            score = float(good.sum()) * min(
                float(angles[good].median()) if good.any() else 0.0, 10.0
            )
            if best is None or score > best[0]:
                full = torch.zeros(len(tracks), dtype=torch.bool)
                full[torch.nonzero(common)[:, 0][inlier]] = True
                best = (score, a, b, w2c_b, full)
    if best is None:
        raise RuntimeError(
            "structure-from-motion could not find an initial frame pair: "
            "not enough parallax or too few tracks"
        )
    return best[1], best[2], best[3], best[4]


def _triangulate_tracks(
    tracks: Tracks, K: Tensor, w2c: Tensor, registered: Tensor, cfg: SfMConfig, candidates: Tensor
) -> tuple[Tensor, Tensor]:
    """Triangulate ``candidates`` from the registered frames; returns points and validity."""
    frames = torch.nonzero(registered)[:, 0]
    index = torch.nonzero(candidates)[:, 0]
    points = torch.zeros(len(tracks), 3, dtype=DTYPE)
    valid = torch.zeros(len(tracks), dtype=torch.bool)
    if index.numel() == 0 or frames.numel() < 2:
        return points, valid
    uv = tracks.uv[index][:, frames].to(DTYPE)
    vis = tracks.visible[index][:, frames]
    cams = w2c[frames]
    Kd = K.to(DTYPE)
    xyz, ok = triangulate_dlt(Kd, cams, uv, vis)
    errors, depths = reprojection_errors(Kd, cams, xyz, uv)
    n_vis = vis.sum(dim=1).clamp_min(1)
    mean_error = (errors * vis).sum(dim=1) / n_vis
    in_front = ((depths > 0) | ~vis).all(dim=1)
    angles = torch.rad2deg(triangulation_angles(cams, xyz, vis))
    good = (
        ok
        & in_front
        & (mean_error < cfg.reproj_threshold)
        & (angles > cfg.min_angle_deg)
        & (vis.sum(dim=1) >= 2)
    )
    points[index] = xyz
    valid[index] = good
    return points, valid


def _register_frame(
    tracks: Tracks, K: Tensor, points: Tensor, valid: Tensor, frame: int, cfg: SfMConfig
) -> Tensor | None:
    """Localise ``frame`` with PnP + RANSAC against the valid 3D points."""
    seen = valid & tracks.visible[:, frame]
    if seen.sum() < 12:
        return None
    object_points = points[seen].cpu().numpy().astype(np.float64)
    image_points = _cv_points(tracks.uv[seen, frame])
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points,
        image_points,
        _cv_K(K),
        None,
        reprojectionError=2.0 * cfg.ransac_threshold,
        confidence=0.9999,
        iterationsCount=300,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 10:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points[inliers[:, 0]], image_points[inliers[:, 0]], _cv_K(K), None, rvec, tvec
    )
    R, _ = cv2.Rodrigues(rvec)
    T = torch.eye(4, dtype=DTYPE)
    T[:3, :3] = torch.from_numpy(R)
    T[:3, 3] = torch.from_numpy(tvec.reshape(3))
    return T


def _run_ba(
    tracks: Tracks,
    K: Tensor,
    w2c: Tensor,
    registered: Tensor,
    points: Tensor,
    valid: Tensor,
    cfg: SfMConfig,
    fixed_frame: int,
    optimize_focal: bool = False,
    max_iterations: int = 20,
) -> tuple[Tensor, Tensor, Tensor]:
    """Bundle-adjust the registered cameras and (a subset of) the valid points."""
    frames = torch.nonzero(registered)[:, 0]
    point_ids = torch.nonzero(valid)[:, 0]
    if point_ids.numel() > cfg.max_ba_points:
        # Keep the longest tracks: they tie the most cameras together.
        length = (tracks.visible[point_ids][:, frames]).sum(dim=1)
        point_ids = point_ids[torch.argsort(length, descending=True)[: cfg.max_ba_points]]
    vis = tracks.visible[point_ids][:, frames]
    p_local, c_local = torch.nonzero(vis, as_tuple=True)
    problem = BAProblem(
        w2c=w2c[frames],
        points=points[point_ids],
        cam_index=c_local,
        point_index=p_local,
        uv=tracks.uv[point_ids][:, frames][p_local, c_local].to(DTYPE),
        K=K.to(DTYPE),
        fixed_cameras=frames == fixed_frame,
    )
    result = bundle_adjust(
        problem,
        max_iterations=max_iterations,
        huber_delta=cfg.huber_delta,
        optimize_focal=optimize_focal,
    )
    w2c = w2c.clone()
    points = points.clone()
    w2c[frames] = result.w2c
    points[point_ids] = result.points
    return w2c, points, result.K


def reconstruct(tracks: Tracks, K: Tensor, cfg: SfMConfig | None = None) -> SfMResult:
    """Estimate camera poses and sparse structure from point tracks.

    Args:
        tracks: 2D tracks over the video.
        K: ``(3, 3)`` intrinsics (an initial guess when ``cfg.refine_focal`` is set).
    """
    cfg = cfg or SfMConfig()
    cv2.setRNGSeed(cfg.seed)
    n_frames = tracks.num_frames
    K = K.to(DTYPE).clone()
    long_enough = tracks.visible.sum(dim=1) >= cfg.min_track_length

    a, b, w2c_b, seed_inliers = _select_initial_pair(tracks, K, cfg)
    logger.info("SfM: initial pair (%d, %d) with %d inliers", a, b, int(seed_inliers.sum()))
    w2c = torch.eye(4, dtype=DTYPE).repeat(n_frames, 1, 1)
    w2c[b] = w2c_b
    registered = torch.zeros(n_frames, dtype=torch.bool)
    registered[a] = registered[b] = True
    points, valid = _triangulate_tracks(tracks, K, w2c, registered, cfg, seed_inliers & long_enough)

    # Register the other frames, the ones between / around the seed pair first.
    middle = 0.5 * (a + b)
    order = sorted((f for f in range(n_frames) if not registered[f]), key=lambda f: abs(f - middle))
    since_ba = 0
    for frame in order:
        pose = _register_frame(tracks, K, points, valid, frame, cfg)
        if pose is None:
            logger.warning("SfM: frame %d could not be registered", frame)
            continue
        w2c[frame] = pose
        registered[frame] = True
        new_points, new_valid = _triangulate_tracks(
            tracks, K, w2c, registered, cfg, long_enough & ~valid
        )
        points = torch.where(new_valid[:, None], new_points, points)
        valid = valid | new_valid
        since_ba += 1
        if since_ba >= cfg.local_ba_interval:
            w2c, points, _ = _run_ba(
                tracks, K, w2c, registered, points, valid, cfg, a, max_iterations=8
            )
            since_ba = 0

    # Global refinement: bundle adjustment, then re-triangulation of every track with the
    # refined cameras (this also re-classifies inliers and outliers).
    for round_index in range(3):
        focal = cfg.refine_focal and round_index > 0
        w2c, points, K = _run_ba(tracks, K, w2c, registered, points, valid, cfg, a, focal, 30)
        points, valid = _triangulate_tracks(tracks, K, w2c, registered, cfg, long_enough)

    # Fix the gauge: world = first registered camera's frame is kept; scale = median depth 1.
    frames = torch.nonzero(registered)[:, 0]
    errors, depths = reprojection_errors(K, w2c, points, tracks.uv.to(DTYPE))
    observed = tracks.visible & registered[None, :]
    scale = depths[valid][observed[valid]].median()
    points = points / scale
    w2c[:, :3, 3] = w2c[:, :3, 3] / scale

    n_observed = observed.sum(dim=1).clamp_min(1)
    mean_error = (errors * observed).sum(dim=1) / n_observed
    inlier = observed & (errors < 2.0 * cfg.reproj_threshold) & valid[:, None]
    logger.info(
        "SfM: %d/%d frames registered, %d/%d tracks reconstructed, median error %.2f px",
        int(registered.sum()),
        n_frames,
        int(valid.sum()),
        len(tracks),
        float(mean_error[valid].median()) if valid.any() else float("nan"),
    )
    del frames
    return SfMResult(w2c, registered, K, points, valid, inlier, mean_error)


def baseline_ratio(w2c: Tensor, points: Tensor) -> float:
    """Camera baseline relative to the scene depth, a measure of how well posed SfM is."""
    centers = camera_centers(w2c)
    extent = (centers.amax(dim=0) - centers.amin(dim=0)).norm()
    depth = (points - centers.mean(dim=0)).norm(dim=-1).median()
    return float(extent / depth.clamp_min(1e-12))
