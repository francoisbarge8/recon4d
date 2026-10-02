"""The end-to-end pipeline: video in, (dynamic) Gaussian scene out.

Front-end (CPU):

1. **Depth prediction** - one depth map per frame, up to an unknown scale.
2. **Point tracking** - 2D tracks through the video.
3. **Camera poses** - structure-from-motion on the tracks (robust to moving objects).
4. **Depth alignment** - the depth maps are brought into the frame of the poses.
5. **Motion segmentation** - pixels and tracks that move independently of the camera.

Back-end (CPU or GPU):

6. **Initialisation** - static Gaussians on consistent depth, dynamic Gaussians and their
   motion bases on the moving tracks.
7. **Optimisation** - photometric, depth, mask, track and rigidity losses.

Every stage is a swappable back-end selected by name in :class:`PipelineConfig`; the
``oracle`` back-ends replace a stage by its ground truth, which is how the benchmark
attributes the final error to each component.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch import Tensor

from recon4d.config import save_yaml
from recon4d.data.sequence import VideoSequence
from recon4d.frontend.depth import DepthEstimator, DepthNoise, OracleDepth
from recon4d.frontend.flow import DISFlow, FlowEstimator, OracleFlow, RAFTFlow
from recon4d.frontend.motion_seg import (
    MotionSegConfig,
    dynamic_queries,
    flow_residuals,
    label_tracks,
    mask_fraction,
    motion_masks,
)
from recon4d.frontend.pose import SfMConfig, SfMResult, reconstruct, triangulate_with_poses
from recon4d.frontend.tracking import KLTConfig, KLTTracker, PointTracker, grid_queries
from recon4d.frontend.tracking.cotracker import CoTrackerConfig
from recon4d.frontend.tracking.flow_chain import FlowChainConfig, FlowChainTracker
from recon4d.fusion.depth_align import DepthAlignConfig, align_depth_maps
from recon4d.gaussians.init import InitConfig, init_scene
from recon4d.gaussians.scene import GaussianScene
from recon4d.gaussians.trainer import SceneTrainer, TrainConfig
from recon4d.geometry.camera import Intrinsics, invert_se3
from recon4d.types import Tracks, TrainingData
from recon4d.utils import get_logger, save_json, seed_everything, timed

logger = get_logger(__name__)

DEPTH_BACKENDS = ("oracle", "oracle-noisy", "depth-anything", "learned")
TRACKER_BACKENDS = ("klt", "flow-chain", "cotracker", "oracle")
FLOW_BACKENDS = ("dis", "raft", "raft-large", "oracle")
POSE_BACKENDS = ("sfm", "gt")


@dataclass
class PipelineConfig:
    """Configuration of the whole pipeline.

    Attributes:
        depth: depth back-end, one of :data:`DEPTH_BACKENDS`.
        pose_tracker: tracker (one of :data:`TRACKER_BACKENDS`) whose tracks estimate the
            camera poses and anchor the depth alignment. Accuracy matters more than
            density here: a few hundred well-localised corners are ideal.
        tracker: tracker giving a *dense* coverage of the scene, moving objects included;
            its tracks supervise the motion of the dynamic Gaussians. They are added to
            the pose tracks; ``"none"`` (or the same back-end as ``pose_tracker``) keeps
            the pose tracks only.
        flow: optical flow back-end, one of :data:`FLOW_BACKENDS`; used by motion
            segmentation and by the flow-chain tracker.
        poses: ``"sfm"`` estimates the poses, ``"gt"`` uses the ground truth.
        dynamic: reconstruct moving objects (False: plain static 3DGS).
        device: device of the scene optimisation.
        seed: seed of every random number generator.
        depth_checkpoint: model id (``depth-anything``) or weights file (``learned``).
        focal_guess: initial focal length, as a fraction of the larger image side, used
            when the video has no known intrinsics (the focal is then refined by SfM).
    """

    depth: str = "oracle-noisy"
    pose_tracker: str = "klt"
    tracker: str = "flow-chain"
    flow: str = "dis"
    poses: str = "sfm"
    dynamic: bool = True
    device: str = "cpu"
    seed: int = 0
    depth_checkpoint: str = ""
    focal_guess: float = 1.2
    depth_noise: DepthNoise = field(default_factory=DepthNoise)
    klt: KLTConfig = field(default_factory=KLTConfig)
    flow_chain: FlowChainConfig = field(default_factory=FlowChainConfig)
    cotracker: CoTrackerConfig = field(default_factory=CoTrackerConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    align: DepthAlignConfig = field(default_factory=DepthAlignConfig)
    motion: MotionSegConfig = field(default_factory=MotionSegConfig)
    init: InitConfig = field(default_factory=InitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class FrontendResult:
    """Everything the front-end estimates from the video."""

    K: Tensor
    """``(3, 3)`` intrinsics (refined by SfM when they were unknown)."""
    w2c: Tensor
    """``(T, 4, 4)`` world-to-camera poses."""
    registered: Tensor
    """``(T,)`` frames whose pose was actually estimated."""
    tracks: Tracks
    """Point tracks with their dynamic / static labels (pose tracks first)."""
    n_pose_tracks: int
    """Number of leading tracks that come from the pose tracker."""
    raw_depth: Tensor
    """``(T, H, W)`` depth prediction as returned by the estimator."""
    depth_kind: str
    """``"scale"`` or ``"disparity"`` (see :class:`~recon4d.frontend.depth.DepthEstimator`)."""
    depth: Tensor
    """``(T, H, W)`` depth aligned to the frame of the poses."""
    depth_valid: Tensor
    """``(T, H, W)`` validity of the aligned depth."""
    dynamic_mask: Tensor
    """``(T, H, W)`` pixels moving independently of the camera."""
    flow_residual: Tensor | None
    """``(T, H, W)`` disagreement between observed and rigid flow (None if not computed)."""
    sfm: SfMResult
    """Sparse reconstruction (poses, points, inliers)."""
    timings: dict[str, float]
    """Wall-clock seconds spent in each stage."""


@dataclass
class PipelineResult:
    scene: GaussianScene
    frontend: FrontendResult
    data: TrainingData
    extent: float
    history: list[dict[str, float]]
    timings: dict[str, float]


def _require_gt(seq: VideoSequence, what: str):
    if seq.gt is None:
        raise ValueError(f"the {what} back-end needs a sequence with ground truth")
    return seq.gt


def make_depth_estimator(cfg: PipelineConfig, seq: VideoSequence) -> DepthEstimator:
    if cfg.depth == "oracle":
        return OracleDepth(_require_gt(seq, "oracle depth").depth)
    if cfg.depth == "oracle-noisy":
        noise = replace(cfg.depth_noise, seed=cfg.depth_noise.seed + cfg.seed)
        return OracleDepth(_require_gt(seq, "oracle depth").depth, noise)
    if cfg.depth == "depth-anything":
        from recon4d.frontend.depth.depth_anything import (
            DEFAULT_CHECKPOINT,
            DepthAnythingV2,
        )

        return DepthAnythingV2(cfg.depth_checkpoint or DEFAULT_CHECKPOINT, cfg.device)
    if cfg.depth == "learned":
        from recon4d.frontend.depth.tiny_net import LearnedDepth

        return LearnedDepth(cfg.depth_checkpoint, cfg.device)
    raise ValueError(f"unknown depth back-end {cfg.depth!r}; choose among {DEPTH_BACKENDS}")


def make_flow_estimator(cfg: PipelineConfig, seq: VideoSequence) -> FlowEstimator:
    if cfg.flow == "dis":
        return DISFlow()
    if cfg.flow == "raft":
        return RAFTFlow(small=True, device=cfg.device)
    if cfg.flow == "raft-large":
        return RAFTFlow(small=False, device=cfg.device)
    if cfg.flow == "oracle":
        return OracleFlow(_require_gt(seq, "oracle flow").oracle)
    raise ValueError(f"unknown flow back-end {cfg.flow!r}; choose among {FLOW_BACKENDS}")


class _OracleTracker(PointTracker):
    """Exact tracks of a pixel grid (an analysis tool, not an estimator)."""

    name = "oracle"

    def __init__(self, oracle, stride: int, keyframe_interval: int) -> None:
        self.oracle = oracle
        self.stride = stride
        self.keyframe_interval = keyframe_interval

    def track_queries(self, images: Tensor, query_frames: Tensor, query_uv: Tensor) -> Tracks:
        truth = self.oracle.tracks(query_frames, query_uv)
        return Tracks(truth.uv, truth.visible, query_frame=query_frames.clone())

    def sample_queries(self, images: Tensor) -> tuple[Tensor, Tensor]:
        n_frames, height, width, _ = images.shape
        grid = grid_queries(height, width, self.stride, margin=2)
        frames = torch.arange(0, n_frames, self.keyframe_interval)
        return frames.repeat_interleave(len(grid)), grid.repeat(len(frames), 1)


def make_tracker(
    name: str, cfg: PipelineConfig, seq: VideoSequence, flow: FlowEstimator
) -> PointTracker:
    if name == "klt":
        return KLTTracker(cfg.klt)
    if name == "flow-chain":
        return FlowChainTracker(flow, cfg.flow_chain)
    if name == "cotracker":
        from recon4d.frontend.tracking.cotracker import CoTracker

        return CoTracker(cfg.cotracker, cfg.device)
    if name == "oracle":
        oracle = _require_gt(seq, "oracle tracker").oracle
        return _OracleTracker(oracle, cfg.flow_chain.grid_stride, cfg.flow_chain.keyframe_interval)
    raise ValueError(f"unknown tracker back-end {name!r}; choose among {TRACKER_BACKENDS}")


def run_frontend(seq: VideoSequence, cfg: PipelineConfig) -> FrontendResult:
    """Estimate depth, tracks, camera poses and motion masks for a video."""
    timings: dict[str, float] = {}
    images = seq.images
    flow = make_flow_estimator(cfg, seq)

    with timed("depth", timings, logger):
        estimator = make_depth_estimator(cfg, seq)
        raw_depth = estimator.predict(images)

    with timed("tracking", timings, logger):
        pose_tracks = make_tracker(cfg.pose_tracker, cfg, seq, flow).track(images)
    logger.info("tracking: %d pose tracks from %s", len(pose_tracks), cfg.pose_tracker)

    with timed("poses", timings, logger):
        if cfg.poses == "sfm":
            if seq.intrinsics is not None:
                K, sfm_cfg = seq.K(torch.float64), cfg.sfm
            else:
                focal = cfg.focal_guess * max(seq.width, seq.height)
                K = Intrinsics(
                    focal, focal, seq.width / 2, seq.height / 2, seq.width, seq.height
                ).matrix(torch.float64)
                sfm_cfg = replace(cfg.sfm, refine_focal=True)
            sfm = reconstruct(
                pose_tracks, K, replace(sfm_cfg, seed=cfg.seed), raw_depth, estimator.kind
            )
        elif cfg.poses == "gt":
            gt = _require_gt(seq, "ground-truth pose")
            sfm = triangulate_with_poses(pose_tracks, seq.K(), invert_se3(gt.c2w), cfg.sfm)
        else:
            raise ValueError(f"unknown pose back-end {cfg.poses!r}; choose among {POSE_BACKENDS}")
    K = sfm.K.to(torch.float32)
    w2c = sfm.w2c.to(torch.float32)

    with timed("depth_alignment", timings, logger):
        aligned = align_depth_maps(
            raw_depth,
            pose_tracks,
            sfm.points.to(torch.float32),
            sfm.inlier,
            w2c,
            replace(cfg.align, kind=estimator.kind),
        )

    # Dense tracks: triangulated with the poses just estimated, which tells which of them
    # are consistent with a static point.
    tracks, reproj_error, static = pose_tracks, sfm.reproj_error, sfm.valid
    dense_tracker = None
    if cfg.tracker not in ("none", "", cfg.pose_tracker):
        with timed("dense_tracking", timings, logger):
            dense_tracker = make_tracker(cfg.tracker, cfg, seq, flow)
            dense = dense_tracker.track(images)
            dense_map = triangulate_with_poses(dense, sfm.K, sfm.w2c, cfg.sfm)
        logger.info("tracking: %d dense tracks from %s", len(dense), cfg.tracker)
        tracks = Tracks.concatenate([pose_tracks, dense])
        reproj_error = torch.cat([reproj_error, dense_map.reproj_error])
        static = torch.cat([static, dense_map.valid])

    residual = None
    if cfg.dynamic:
        with timed("motion_segmentation", timings, logger):
            residual = flow_residuals(images, aligned.depth, K, w2c, flow, cfg.motion)
            masks = motion_masks(residual, cfg.motion)
            dynamic = label_tracks(
                tracks, reproj_error.to(torch.float32), static, masks, cfg.motion
            )
            if cfg.motion.dynamic_track_stride > 0 and masks.any():
                # Moving objects are small: cover them with many more tracks than the
                # scene-wide grids provide.
                tracker = dense_tracker or make_tracker(cfg.pose_tracker, cfg, seq, flow)
                frames, uv = dynamic_queries(
                    masks, cfg.motion.dynamic_track_stride, cfg.motion.dynamic_track_interval
                )
                extra = tracker.track_queries(images, frames, uv)
                extra = extra.subset(mask_fraction(extra, masks) > cfg.motion.track_mask_fraction)
                tracks = Tracks.concatenate([tracks, extra])
                dynamic = torch.cat([dynamic, torch.ones(len(extra), dtype=torch.bool)])
        logger.info(
            "motion segmentation: %.1f%% of the pixels and %d tracks are dynamic",
            100.0 * masks.float().mean(),
            int(dynamic.sum()),
        )
    else:
        masks = torch.zeros(images.shape[:3], dtype=torch.bool)
        dynamic = torch.zeros(len(tracks), dtype=torch.bool)

    return FrontendResult(
        K=K,
        w2c=w2c,
        registered=sfm.registered,
        tracks=tracks.with_labels(dynamic),
        n_pose_tracks=len(pose_tracks),
        raw_depth=raw_depth,
        depth_kind=estimator.kind,
        depth=aligned.depth,
        depth_valid=aligned.valid,
        dynamic_mask=masks,
        flow_residual=residual,
        sfm=sfm,
        timings=timings,
    )


def training_data(seq: VideoSequence, frontend: FrontendResult) -> TrainingData:
    """Assemble what the scene optimisation consumes.

    Frames whose pose could not be estimated are dropped from the training set.
    """
    registered = set(torch.nonzero(frontend.registered)[:, 0].tolist())
    train_frames = [f for f in seq.train_indices if f in registered]
    return TrainingData(
        images=seq.images,
        K=frontend.K,
        w2c=frontend.w2c,
        train_frames=train_frames,
        depth=frontend.depth,
        depth_valid=frontend.depth_valid,
        dynamic_mask=frontend.dynamic_mask,
        tracks=frontend.tracks,
    )


def reconstruct_scene(
    data: TrainingData, cfg: PipelineConfig
) -> tuple[GaussianScene, float, list[dict[str, float]]]:
    """Initialise and optimise the Gaussian scene; returns ``(scene, extent, history)``."""
    scene, extent = init_scene(data, cfg.init, cfg.dynamic)
    device = torch.device(cfg.device)
    scene.to(device)
    trainer = SceneTrainer(scene, data.to(device), replace(cfg.train, seed=cfg.seed), extent)
    history = trainer.fit()
    return scene, extent, history


def run_pipeline(
    seq: VideoSequence, cfg: PipelineConfig, out_dir: str | Path | None = None
) -> PipelineResult:
    """Run the whole pipeline on a video.

    With ``out_dir`` set, the configuration, the training history and the timings are
    written there (evaluation and rendering are separate steps, see
    :mod:`recon4d.evaluation`).
    """
    seed_everything(cfg.seed)
    frontend = run_frontend(seq, cfg)
    data = training_data(seq, frontend)
    timings = dict(frontend.timings)
    with timed("optimisation", timings, logger):
        scene, extent, history = reconstruct_scene(data, cfg)
    if out_dir is not None:
        out_dir = Path(out_dir)
        save_yaml(out_dir / "config.yaml", cfg)
        save_json(out_dir / "history.json", history)
        save_json(out_dir / "timings.json", timings)
    return PipelineResult(scene, frontend, data, extent, history, timings)
