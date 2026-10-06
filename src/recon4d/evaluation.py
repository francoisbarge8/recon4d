"""Evaluation of the front-end and of the reconstructed scene against ground truth.

A monocular reconstruction lives in its own similarity frame. All comparisons are made
after mapping it onto the ground-truth frame with one Sim(3), estimated from the camera
poses (see :func:`recon4d.metrics.pose.align_frames`); no ground-truth geometry is used to
compute that alignment.

Reported groups (all values are floats; lengths are in ground-truth units):

``pose``            ATE, RPE and orientation error of the camera trajectory
``depth_raw``       per-frame depth prediction with the best per-frame scale (an upper
                    bound on what per-frame alignment can achieve)
``depth_aligned``   depth after alignment to the poses
``depth_rendered``  depth rendered from the optimised scene (held-out frames)
``tracking``        TAP-Vid metrics of the point tracks
``motion_mask``     IoU / precision / recall of the dynamic masks (scenes with moving
                    objects), fraction of the static pixels labelled as moving
``nvs_train/test``  PSNR / SSIM / LPIPS on training and held-out frames
``nvs_val``         the same on held-out *cameras* at the training timestamps, restricted
                    to co-visible pixels, plus the moving objects alone
``geometry_*``      Chamfer distance and F-score of static and dynamic geometry
``tracking_3d``     end-point error of 3D trajectories through the scene's motion field
``temporal``        warping error, temporal-difference PSNR and depth temporal error
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.data.sequence import VideoSequence
from recon4d.fusion.pointcloud import backproject_frames, multiview_consistency
from recon4d.gaussians.rasterizer import RasterSettings
from recon4d.gaussians.scene import GaussianScene
from recon4d.gaussians.trainer import render_views
from recon4d.geometry.align import Sim3
from recon4d.geometry.camera import invert_se3
from recon4d.metrics.depth import align_depth, depth_metrics
from recon4d.metrics.geometry import geometry_metrics
from recon4d.metrics.image import LPIPS, psnr, ssim
from recon4d.metrics.pose import align_frames, pose_metrics
from recon4d.metrics.temporal import depth_temporal_error, temporal_difference_psnr, warping_error
from recon4d.metrics.tracking import tapvid_metrics, trajectory_error_3d
from recon4d.pipeline import FrontendResult
from recon4d.utils import get_logger

logger = get_logger(__name__)

Metrics = dict[str, dict[str, float]]


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation parameters.

    Attributes:
        n_static_points, n_dynamic_points: size of the ground-truth surface samples.
        fscore_thresholds: distance thresholds of the F-score (ground-truth units).
        lpips: LPIPS backbone (``"alex"``, ``"vgg"``, ``"squeeze"``) or ``None`` to skip
            LPIPS (it needs the optional ``lpips`` package and downloads weights).
        max_train_views: number of training frames rendered for the training-view metrics.
        min_views: a ground-truth surface point is evaluated only if the training video
            sees it in at least this many frames.
    """

    n_static_points: int = 60000
    n_dynamic_points: int = 4000
    fscore_thresholds: tuple[float, ...] = (0.05, 0.1)
    lpips: str | None = None
    max_train_views: int = 8
    min_views: int = 2


def _masked_iou(pred: Tensor, target: Tensor) -> dict[str, float]:
    intersection = float((pred & target).sum())
    return {
        "iou": intersection / max(float((pred | target).sum()), 1.0),
        "precision": intersection / max(float(pred.sum()), 1.0),
        "recall": intersection / max(float(target.sum()), 1.0),
    }


def _image_metrics(
    pred: Tensor, target: Tensor, mask: Tensor | None, lpips: LPIPS | None
) -> dict[str, float]:
    if mask is not None and not mask.any():
        return {}
    metrics = {"psnr": float(psnr(pred, target, mask)), "ssim": float(ssim(pred, target, mask))}
    if lpips is not None:
        metrics["lpips"] = float(lpips(pred, target, mask))
    return metrics


def _ground_truth_flows(seq: VideoSequence) -> tuple[Tensor, Tensor, Tensor]:
    """Flow, validity and scene flow between all consecutive frames."""
    pairs = [seq.gt.oracle.scene_flow(t, t + 1) for t in range(seq.num_frames - 1)]
    return (
        torch.stack([p[0] for p in pairs]),
        torch.stack([p[1] for p in pairs]),
        torch.stack([p[2] for p in pairs]),
    )


def world_alignment(seq: VideoSequence, frontend: FrontendResult) -> Sim3:
    """Similarity mapping the reconstruction frame onto the ground-truth frame."""
    return align_frames(invert_se3(frontend.w2c), seq.gt.c2w)


def evaluate_frontend(seq: VideoSequence, frontend: FrontendResult) -> Metrics:
    """Compare the outputs of the front-end with the ground truth of a synthetic sequence."""
    gt = seq.gt
    if gt is None:
        raise ValueError("evaluation needs a sequence with ground truth")
    metrics: Metrics = {}
    est_c2w = invert_se3(frontend.w2c)
    metrics["pose"] = pose_metrics(est_c2w, gt.c2w)
    metrics["pose"]["focal_error"] = abs(float(frontend.K[0, 0]) / seq.intrinsics.fx - 1.0)
    metrics["pose"]["registered"] = float(frontend.registered.float().mean())
    scale = float(world_alignment(seq, frontend).scale)

    # Depth. The raw prediction has an arbitrary per-frame scale (or scale and shift):
    # aligning each frame to the ground truth gives the ceiling of per-frame alignment.
    mode = "disparity" if frontend.depth_kind == "disparity" else "scale"
    per_frame = torch.stack(
        [align_depth(frontend.raw_depth[t], gt.depth[t], mode=mode) for t in range(seq.num_frames)]
    )
    metrics["depth_raw"] = depth_metrics(per_frame, gt.depth)
    aligned = frontend.depth * scale
    metrics["depth_aligned"] = depth_metrics(aligned, gt.depth, frontend.depth_valid)

    # Temporal consistency of depth: one global scale for the raw prediction (what a user
    # of the raw maps would do) against our per-frame aligned maps.
    flows, valid, scene_flow = _ground_truth_flows(seq)
    K, c2w = seq.K(), gt.c2w
    raw_depth = frontend.raw_depth
    if frontend.depth_kind == "disparity":
        raw_depth = torch.stack(
            [
                align_depth(raw_depth[t], gt.depth[t], mode="disparity")
                for t in range(seq.num_frames)
            ]
        )
        metrics["temporal"] = {
            "depth_raw": float(depth_temporal_error(raw_depth, K, c2w, flows, valid, scene_flow))
        }
    else:
        global_scale = (gt.depth.median() / raw_depth.median()).item()
        metrics["temporal"] = {
            "depth_raw": float(
                depth_temporal_error(raw_depth * global_scale, K, c2w, flows, valid, scene_flow)
            )
        }
    metrics["temporal"]["depth_aligned"] = float(
        depth_temporal_error(aligned, K, c2w, flows, valid, scene_flow)
    )
    metrics["temporal"]["depth_gt"] = float(
        depth_temporal_error(gt.depth, K, c2w, flows, valid, scene_flow)
    )

    # Tracks, evaluated on the tracker's own query points.
    tracks = frontend.tracks
    query_frames, query_uv = tracks.queries()
    truth = gt.oracle.tracks(query_frames, query_uv)
    metrics["tracking"] = tapvid_metrics(
        tracks.uv, tracks.visible, truth.uv, truth.visible, query_frames, seq.width, seq.height
    )
    metrics["tracking"]["count"] = float(len(tracks))
    if truth.dynamic.any():
        d = truth.dynamic
        dynamic = tapvid_metrics(
            tracks.uv[d],
            tracks.visible[d],
            truth.uv[d],
            truth.visible[d],
            query_frames[d],
            seq.width,
            seq.height,
        )
        metrics["tracking"].update({f"dynamic_{k}": v for k, v in dynamic.items()})
        metrics["tracking"]["dynamic_count"] = float(d.sum())

    # Motion segmentation. Without moving objects IoU, precision and recall are undefined
    # (a single false positive would make them 0); the false-positive rate is not.
    static = ~gt.dynamic_mask
    false_positives = float((frontend.dynamic_mask & static).sum())
    metrics["motion_mask"] = {
        "false_positive_rate": false_positives / max(float(static.sum()), 1.0)
    }
    if gt.dynamic_mask.any():
        metrics["motion_mask"].update(_masked_iou(frontend.dynamic_mask, gt.dynamic_mask))
        if tracks.dynamic is not None and truth.dynamic.any():
            labels = _masked_iou(tracks.dynamic, truth.dynamic)
            metrics["motion_mask"].update({f"track_{k}": v for k, v in labels.items()})
    return metrics


def _static_cloud_from_depth(
    depth: Tensor, valid: Tensor, dynamic: Tensor | None, K: Tensor, w2c: Tensor, frames: list[int]
) -> Tensor:
    """Fused point cloud of the static scene from depth maps (multi-view consistent pixels)."""
    masks = []
    for position, f in enumerate(frames):
        keep = valid[f].clone()
        if dynamic is not None:
            keep &= ~dynamic[f]
        neighbours = [frames[i] for i in (position - 2, position + 2) if 0 <= i < len(frames)]
        if neighbours:
            keep &= multiview_consistency(depth, K, w2c, f, neighbours, 0.05, valid) >= 1
        masks.append(keep)
    points, _ = backproject_frames(
        depth[frames], K, invert_se3(w2c)[frames], mask=torch.stack(masks), stride=2
    )
    return points


def evaluate_scene(
    seq: VideoSequence,
    frontend: FrontendResult,
    scene: GaussianScene,
    cfg: EvalConfig | None = None,
    raster: RasterSettings | None = None,
) -> Metrics:
    """Evaluate an optimised scene: view synthesis, geometry, 3D tracking, temporal stability."""
    cfg = cfg or EvalConfig()
    gt = seq.gt
    if gt is None:
        raise ValueError("evaluation needs a sequence with ground truth")
    width, height, n_frames = seq.width, seq.height, seq.num_frames
    K, w2c = frontend.K, frontend.w2c
    sim = world_alignment(seq, frontend)
    scale = float(sim.scale)
    lpips = LPIPS(cfg.lpips) if cfg.lpips else None
    metrics: Metrics = {}

    def render(poses: Tensor, frames: list[int]) -> dict[str, Tensor]:
        return render_views(scene, K, poses, frames, width, height, raster)

    # ---------------------------------------------------------------- view synthesis
    train = seq.train_indices
    step = max(len(train) // cfg.max_train_views, 1)
    train_subset = train[::step][: cfg.max_train_views]
    out = render(w2c[train_subset], train_subset)
    metrics["nvs_train"] = _image_metrics(out["color"], seq.images[train_subset], None, lpips)

    test = seq.test_indices
    if test:
        out = render(w2c[test], test)
        metrics["nvs_test"] = _image_metrics(out["color"], seq.images[test], None, lpips)
        moving = _image_metrics(out["color"], seq.images[test], gt.dynamic_mask[test], None)
        metrics["nvs_test"].update({f"dynamic_{k}": v for k, v in moving.items()})
        valid = out["alpha"] > 0.5
        metrics["depth_rendered"] = depth_metrics(out["depth"] * scale, gt.depth[test], valid)

    # Held-out cameras: placed in the reconstruction frame through the pose alignment.
    to_reconstruction = sim.inverse()
    val_metrics: dict[str, list[float]] = {}
    td_psnr = []
    for camera in gt.val_cameras:
        pose = invert_se3(to_reconstruction.apply_to_c2w(camera.c2w[None]))[0]
        out = render(pose[None].expand(len(camera.frames), -1, -1), camera.frames)
        groups = {
            "": _image_metrics(out["color"], camera.images, camera.covisible, lpips),
            "dynamic_": _image_metrics(
                out["color"], camera.images, camera.covisible & camera.dynamic_mask, None
            ),
        }
        for prefix, values in groups.items():
            for name, value in values.items():
                val_metrics.setdefault(prefix + name, []).append(value)
        both = camera.covisible[1:] & camera.covisible[:-1]
        # A reference video that never changes (a static scene seen from a fixed camera)
        # leaves nothing to measure: the metric is undefined there, not infinite.
        changes = (camera.images[1:] != camera.images[:-1]).any(dim=-1)
        if (both & changes).any():
            td_psnr.append(float(temporal_difference_psnr(out["color"], camera.images, both)))
    if val_metrics:
        metrics["nvs_val"] = {k: sum(v) / len(v) for k, v in val_metrics.items()}

    # ---------------------------------------------------------------------- geometry
    surface = gt.oracle.surface_cloud(
        cfg.n_static_points, cfg.n_dynamic_points, cfg.min_views, frames=train
    )
    fused = _static_cloud_from_depth(
        frontend.depth, frontend.depth_valid, frontend.dynamic_mask, K, w2c, train
    )
    if fused.shape[0] > 0:
        metrics["geometry_fused"] = geometry_metrics(
            sim.apply(fused), surface.static, cfg.fscore_thresholds
        )

    # Surface actually rendered by the scene model, with its static part selected.
    step = max(len(train) // 12, 1)
    views = train[::step]
    out = render(w2c[views], views)
    static_pixels = out["alpha"] > 0.5
    if "dynamic" in out:
        static_pixels &= out["dynamic"] < 0.5
    rendered, _ = backproject_frames(
        out["depth"], K, invert_se3(w2c)[views], mask=static_pixels, stride=2
    )
    if rendered.shape[0] > 0:
        metrics["geometry_rendered"] = geometry_metrics(
            sim.apply(rendered), surface.static, cfg.fscore_thresholds
        )

    observed = surface.dynamic_observed
    if scene.is_dynamic and observed.any():
        opaque = scene.dynamic.opacities.detach().cpu() > 0.3
        trajectories = scene.dynamic_trajectories().detach().cpu()[opaque]
        frames = list(range(0, n_frames, max(n_frames // 12, 1)))
        per_frame = [
            geometry_metrics(
                sim.apply(trajectories[:, f]), surface.dynamic[observed, f], cfg.fscore_thresholds
            )
            for f in frames
            if trajectories.shape[0] > 0
        ]
        if per_frame:
            metrics["geometry_dynamic"] = {
                k: sum(m[k] for m in per_frame) / len(per_frame) for k in per_frame[0]
            }

    # -------------------------------------------------------------------- 3D tracking
    if observed.any():
        truth = surface.dynamic[observed]  # (D, T, 3) in ground-truth units
        # Each point is handed to the motion field at the first frame where the training
        # camera sees it.
        first = surface.dynamic_first_seen[observed]
        predicted = torch.zeros_like(truth)
        for f in torch.unique(first).tolist():
            select = first == f
            start = to_reconstruction.apply(truth[select, f])
            if scene.is_dynamic:
                device = scene.static.means.device
                moved = scene.transport(start.to(device), int(f)).cpu()
            else:
                moved = start[:, None].expand(-1, n_frames, -1)
            predicted[select] = sim.apply(moved)
        everywhere = torch.ones(truth.shape[:2], dtype=torch.bool)
        metrics["tracking_3d"] = trajectory_error_3d(predicted, truth, everywhere)

    # ----------------------------------------------------------------------- temporal
    flows, flow_valid, scene_flow = _ground_truth_flows(seq)
    all_frames = list(range(n_frames))
    out = render(w2c, all_frames)
    reference = float(warping_error(seq.images, flows, flow_valid))
    rendered_error = float(warping_error(out["color"], flows, flow_valid))
    metrics["temporal"] = {
        "warping_error": rendered_error,
        "warping_error_gt_video": reference,
        "depth_rendered": float(
            depth_temporal_error(
                out["depth"] * scale, seq.K(), gt.c2w, flows, flow_valid, scene_flow
            )
        ),
    }
    if td_psnr:
        metrics["temporal"]["difference_psnr_val"] = sum(td_psnr) / len(td_psnr)
    metrics["model"] = {
        "static_gaussians": float(len(scene.static)),
        "dynamic_gaussians": float(len(scene.dynamic)) if scene.dynamic is not None else 0.0,
        "alignment_scale": scale,
    }
    return metrics


def flatten(metrics: Metrics) -> dict[str, float]:
    """``{"group/name": value}`` view of nested metrics, convenient for tables."""
    return {
        f"{group}/{name}": value
        for group, values in metrics.items()
        for name, value in values.items()
    }
