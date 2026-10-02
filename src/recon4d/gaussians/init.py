"""Initialisation of a Gaussian scene from the outputs of the front-end.

Static Gaussians are seeded on the back-projected depth of a few keyframes, keeping only
pixels that are static and whose depth is confirmed by neighbouring views. Dynamic
Gaussians are seeded on the moving pixels and attached to motion bases fitted to the 3D
trajectories of the dynamic point tracks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.spatial import cKDTree
from torch import Tensor

from recon4d.fusion.pointcloud import (
    backproject_frames,
    downsample_to,
    multiview_consistency,
    voxel_downsample,
)
from recon4d.gaussians.model import GaussianCloud
from recon4d.gaussians.motion import MotionBases, init_motion_bases
from recon4d.gaussians.scene import MOTION_LOGITS, GaussianScene
from recon4d.geometry.camera import (
    backproject_depth,
    invert_se3,
    sample_depth,
    transform_points,
    unproject,
)
from recon4d.types import TrainingData
from recon4d.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class InitConfig:
    """Parameters of the scene initialisation.

    Attributes:
        n_static: target number of static Gaussians.
        n_dynamic: target number of dynamic Gaussians.
        keyframes: number of frames whose depth seeds the Gaussians.
        stride: pixel stride of the back-projection.
        consistency_views: a static seed needs its depth confirmed by this many of its
            neighbouring frames (0 disables the check).
        consistency_tolerance: relative depth tolerance of that check.
        scale_factor: initial size of a Gaussian, as a fraction of the distance to its
            neighbours. Values below 1 keep the per-pixel overdraw (and hence the cost of
            CPU rasterisation) low; the optimisation grows what needs to grow.
        opacity: initial opacity.
        sh_degree: degree of the spherical harmonics.
        num_bases: number of rigid motion bases.
        min_track_frames: a dynamic track needs this many valid 3D positions to be used.
        basis_fit_iterations: Adam steps refining the bases on the 3D tracks.
        knn: number of neighbouring tracks blended to give a dynamic Gaussian its motion.
    """

    n_static: int = 6000
    n_dynamic: int = 1500
    keyframes: int = 6
    stride: int = 2
    consistency_views: int = 2
    consistency_tolerance: float = 0.05
    scale_factor: float = 0.5
    opacity: float = 0.5
    sh_degree: int = 0
    num_bases: int = 6
    min_track_frames: int = 4
    basis_fit_iterations: int = 300
    knn: int = 3


def _keyframes(frames: list[int], count: int) -> list[int]:
    if len(frames) <= count:
        return list(frames)
    positions = torch.linspace(0, len(frames) - 1, count).round().to(torch.int64).tolist()
    return [frames[i] for i in sorted(set(positions))]


def scene_extent(data: TrainingData) -> float:
    """Characteristic scene size: the median depth seen by the training frames."""
    if data.depth is None:
        raise ValueError("a depth prior is needed to measure the scene extent")
    depth = data.depth[data.train_frames]
    valid = depth > 0
    if data.depth_valid is not None:
        valid = valid & data.depth_valid[data.train_frames]
    return float(depth[valid].median())


def init_static_cloud(data: TrainingData, cfg: InitConfig) -> GaussianCloud:
    """Seed static Gaussians on multi-view consistent, non-moving depth pixels."""
    if data.depth is None:
        raise ValueError("static initialisation needs a depth prior")
    frames = _keyframes(data.train_frames, cfg.keyframes)
    c2w = invert_se3(data.w2c)
    valid = data.depth > 0
    if data.depth_valid is not None:
        valid = valid & data.depth_valid
    masks = []
    train = data.train_frames
    for f in frames:
        keep = valid[f].clone()
        if data.dynamic_mask is not None:
            keep &= ~data.dynamic_mask[f]
        if cfg.consistency_views > 0 and len(train) > 2:
            # Compare with the nearest training frames on both sides.
            position = train.index(f)
            neighbours = [
                train[i]
                for i in (position - 2, position - 1, position + 1, position + 2)
                if 0 <= i < len(train)
            ]
            agree = multiview_consistency(
                data.depth, data.K, data.w2c, f, neighbours, cfg.consistency_tolerance, valid
            )
            keep &= agree >= min(cfg.consistency_views, len(neighbours))
        masks.append(keep)
    points, colors = backproject_frames(
        data.depth[frames], data.K, c2w[frames], data.images[frames], torch.stack(masks), cfg.stride
    )
    if points.shape[0] == 0:
        raise RuntimeError("no static seed point survived the consistency checks")
    points, colors, _ = downsample_to(points, cfg.n_static, colors)
    logger.info("static initialisation: %d Gaussians from frames %s", points.shape[0], frames)
    return GaussianCloud.from_points(
        points, colors, cfg.sh_degree, cfg.opacity, scale_factor=cfg.scale_factor
    )


def lift_tracks(data: TrainingData, select: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Lift 2D tracks to world space with the depth prior.

    Returns ``xyz (N, T, 3)`` and ``ok (N, T)``; a position is valid when the track is
    visible and a reliable depth can be sampled under it (not on a depth discontinuity).
    """
    tracks = data.tracks if select is None else data.tracks.subset(select)
    c2w = invert_se3(data.w2c)
    n = len(tracks)
    xyz = torch.zeros(n, data.num_frames, 3)
    ok = torch.zeros(n, data.num_frames, dtype=torch.bool)
    for t in range(data.num_frames):
        valid = None if data.depth_valid is None else data.depth_valid[t]
        z, sampled = sample_depth(data.depth[t], tracks.uv[:, t], valid=valid)
        ok[:, t] = sampled & tracks.visible[:, t]
        xyz[:, t] = transform_points(c2w[t], unproject(data.K, tracks.uv[:, t], z))
    return xyz, ok


def fit_motion_bases(
    bases: MotionBases,
    logits: Tensor,
    canonical: Tensor,
    xyz: Tensor,
    ok: Tensor,
    iterations: int,
    extent: float,
    smoothness: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Refine motion bases, coefficients and canonical points on 3D tracks (L1 fit).

    Returns the refined ``(logits, canonical)``; ``bases`` is updated in place.
    """
    logits = logits.clone().requires_grad_(True)
    canonical = canonical.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": [bases.rot6d], "lr": 3e-3},
            {"params": [bases.trans], "lr": 3e-3 * extent},
            {"params": [logits], "lr": 3e-2},
            {"params": [canonical], "lr": 1e-3 * extent},
        ]
    )
    weight = ok.to(xyz.dtype)[..., None] / (ok.sum().clamp_min(1) * extent)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        predicted = bases.trajectories(logits, canonical)
        loss = ((predicted - xyz).abs() * weight).sum() + smoothness * bases.smoothness(extent)
        loss.backward()
        optimizer.step()
    return logits.detach(), canonical.detach()


def init_dynamic(
    data: TrainingData, cfg: InitConfig, extent: float
) -> tuple[GaussianCloud, MotionBases] | None:
    """Seed dynamic Gaussians and their motion bases; ``None`` when nothing moves."""
    tracks = data.tracks
    if tracks is None or tracks.dynamic is None or data.dynamic_mask is None or data.depth is None:
        return None
    xyz, ok = lift_tracks(data, tracks.dynamic)
    keep = ok.sum(dim=1) >= cfg.min_track_frames
    xyz, ok = xyz[keep], ok[keep]
    if xyz.shape[0] < 4 or not data.dynamic_mask.any():
        logger.info("dynamic initialisation: no usable dynamic tracks, the scene is static")
        return None

    # A basis needs enough tracks to be estimated at all.
    num_bases = max(1, min(cfg.num_bases, xyz.shape[0] // 12))
    bases, logits, canonical, canonical_frame = init_motion_bases(xyz, ok, num_bases)
    logits, canonical = fit_motion_bases(
        bases, logits, canonical, xyz, ok, cfg.basis_fit_iterations, extent
    )
    with torch.no_grad():
        residual = (bases.trajectories(logits, canonical) - xyz).norm(dim=-1)[ok].median()
    logger.info(
        "dynamic initialisation: %d tracks, %d bases, canonical frame %d, fit residual %.4f",
        xyz.shape[0],
        bases.num_bases,
        canonical_frame,
        float(residual),
    )

    # Seed Gaussians on moving pixels of a few keyframes and carry them to canonical space
    # with the motion of the nearest tracks.
    c2w = invert_se3(data.w2c)
    weights_tracks = torch.softmax(logits, dim=-1)
    frames = _keyframes(data.train_frames, cfg.keyframes)
    seeds, seed_colors, seed_weights = [], [], []
    for f in frames:
        mask = data.dynamic_mask[f] & (data.depth[f] > 0)
        if data.depth_valid is not None:
            mask &= data.depth_valid[f]
        visible = ok[:, f]
        if not mask.any() or visible.sum() < 1:
            continue
        world = backproject_depth(data.K, c2w[f], data.depth[f])[mask]
        k = min(cfg.knn, int(visible.sum()))
        distance, index = cKDTree(xyz[visible, f].numpy()).query(world.numpy(), k=k)
        distance = torch.as_tensor(distance, dtype=world.dtype).reshape(-1, k)
        index = torch.as_tensor(index, dtype=torch.int64).reshape(-1, k)
        blend = 1.0 / (distance + 1e-3 * extent)
        blend = blend / blend.sum(dim=1, keepdim=True)
        w = (blend[..., None] * weights_tracks[visible][index]).sum(dim=1)  # (M, B)
        with torch.no_grad():
            rotation, translation = bases.blend(torch.log(w.clamp_min(1e-6)), f)
        seeds.append((rotation.transpose(1, 2) @ (world - translation)[..., None])[..., 0])
        seed_colors.append(data.images[f][mask])
        seed_weights.append(w)
    if not seeds:
        return None
    points = torch.cat(seeds)
    features = torch.cat([torch.cat(seed_colors), torch.cat(seed_weights)], dim=1)
    if points.shape[0] > cfg.n_dynamic:
        points, features, _ = downsample_to(points, cfg.n_dynamic, features)
    else:
        points, features = voxel_downsample(points, 1e-4 * extent, features)
    colors, w = features[:, :3], features[:, 3:]
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-9)
    cloud = GaussianCloud.from_points(
        points,
        colors,
        cfg.sh_degree,
        cfg.opacity,
        scale_factor=cfg.scale_factor,
        extras={MOTION_LOGITS: torch.log(w.clamp_min(1e-4))},
    )
    logger.info("dynamic initialisation: %d Gaussians", len(cloud))
    return cloud, bases


def init_scene(
    data: TrainingData, cfg: InitConfig, dynamic: bool = True
) -> tuple[GaussianScene, float]:
    """Build the initial scene and return it with the scene extent."""
    extent = scene_extent(data)
    static = init_static_cloud(data, cfg)
    moving = init_dynamic(data, cfg, extent) if dynamic else None
    if moving is None:
        return GaussianScene(static), extent
    return GaussianScene(static, moving[0], moving[1]), extent
