"""Write the artefacts of a run: metrics, checkpoints, videos and point clouds."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from recon4d.data.sequence import VideoSequence
from recon4d.evaluation import Metrics, world_alignment
from recon4d.fusion.pointcloud import backproject_frames
from recon4d.gaussians.scene import GaussianScene
from recon4d.gaussians.trainer import render_views
from recon4d.geometry.camera import invert_se3, look_at
from recon4d.pipeline import PipelineResult
from recon4d.utils import get_logger, save_json
from recon4d.viz import (
    colorize_depth,
    draw_tracks,
    overlay_mask,
    save_gaussians,
    save_gif,
    save_image,
    save_point_cloud,
    side_by_side,
)

logger = get_logger(__name__)


def save_scene(path: str | Path, scene: GaussianScene) -> None:
    """Save the scene parameters (state dict plus the spherical-harmonics degrees)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {name: cloud.sh_degree for name, cloud in scene.clouds().items()}
    torch.save(
        {"state": {k: v.cpu() for k, v in scene.state_dict().items()}, "sh_degree": meta}, path
    )


def orbit_poses(w2c: Tensor, depth: Tensor, n_views: int = 24, amplitude: float = 0.12) -> Tensor:
    """A small circular camera path around the middle training view, looking at the scene.

    The camera moves on a circle of radius ``amplitude`` times the scene depth, in the
    image plane of the reference view, and keeps looking at the point at median depth in
    front of it: the classic "wiggle" that reveals the reconstructed 3D structure.
    """
    reference = invert_se3(w2c)[w2c.shape[0] // 2]
    distance = depth[depth > 0].median()
    target = reference[:3, 3] + distance * reference[:3, 2]
    angles = torch.linspace(0.0, 2.0 * torch.pi, n_views + 1)[:-1]
    poses = []
    for angle in angles:
        offset = (
            amplitude
            * distance
            * (torch.cos(angle) * reference[:3, 0] + torch.sin(angle) * reference[:3, 1])
        )
        # The camera's "down" axis is +y, so the world "up" passed to look_at is -y.
        poses.append(look_at(reference[:3, 3] + offset, target, up=-reference[:3, 1]))
    return invert_se3(torch.stack(poses))


def save_run(
    out_dir: str | Path,
    seq: VideoSequence,
    result: PipelineResult,
    metrics: Metrics | None = None,
    fps: float = 12.0,
    scale: int = 2,
) -> None:
    """Write the qualitative and quantitative outputs of a pipeline run to ``out_dir``.

    * ``metrics.json`` - evaluation results, when available;
    * ``scene.pt`` and ``gaussians*.ply`` - the optimised scene (PLY in the 3DGS layout);
    * ``points.ply`` - the fused static point cloud of the front-end;
    * ``frontend.gif`` - input | tracks | aligned depth | motion mask;
    * ``reconstruction.gif`` - input | render | rendered depth at the estimated poses;
    * ``wiggle.gif`` - the scene from a small orbit around the middle view, time frozen;
    * ``bullet_time.gif`` - time advancing while the camera orbits (dynamic scenes);
    * ``val*.gif`` - ground truth | render for the held-out cameras of synthetic scenes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frontend, scene = result.frontend, result.scene
    width, height, n_frames = seq.width, seq.height, seq.num_frames
    K, w2c = frontend.K, frontend.w2c
    if metrics is not None:
        save_json(out_dir / "metrics.json", metrics)
    save_scene(out_dir / "scene.pt", scene)
    save_gaussians(out_dir / "gaussians_static.ply", scene.static)

    valid = frontend.depth_valid & ~frontend.dynamic_mask
    step = max(n_frames // 8, 1)
    frames = list(range(0, n_frames, step))
    points, colors = backproject_frames(
        frontend.depth[frames],
        K,
        invert_se3(w2c)[frames],
        seq.images[frames],
        valid[frames],
        stride=2,
    )
    save_point_cloud(out_dir / "points.ply", points, colors)

    near = float(torch.quantile(frontend.depth[frontend.depth_valid][::7], 0.02))
    far = float(torch.quantile(frontend.depth[frontend.depth_valid][::7], 0.98))
    tracks_overlay = draw_tracks(seq.images, frontend.tracks)
    depth_panel = colorize_depth(frontend.depth, frontend.depth_valid, near, far)
    mask_panel = overlay_mask(seq.images, frontend.dynamic_mask)
    save_gif(
        out_dir / "frontend.gif",
        side_by_side(seq.images, tracks_overlay, depth_panel, mask_panel),
        fps,
        scale,
    )

    all_frames = list(range(n_frames))
    out = render_views(scene, K, w2c, all_frames, width, height)
    rendered_depth = colorize_depth(out["depth"], out["alpha"] > 0.5, near, far)
    save_gif(
        out_dir / "reconstruction.gif",
        side_by_side(seq.images, out["color"], rendered_depth),
        fps,
        scale,
    )
    middle = n_frames // 2
    save_image(
        out_dir / "reconstruction.png",
        side_by_side(seq.images[middle], out["color"][middle], rendered_depth[middle]),
        scale,
    )

    orbit = orbit_poses(w2c, frontend.depth)
    wiggle = render_views(scene, K, orbit, [middle] * len(orbit), width, height)
    save_gif(out_dir / "wiggle.gif", wiggle["color"], fps, scale)
    if scene.is_dynamic:
        times = torch.linspace(0, n_frames - 1, 2 * len(orbit)).round().to(torch.int64).tolist()
        bullet = render_views(scene, K, orbit.repeat(2, 1, 1), times, width, height)
        save_gif(out_dir / "bullet_time.gif", bullet["color"], fps, scale)
        dynamic = scene.dynamic
        rotation, translation = scene.motion.blend(dynamic.params["motion_logits"], middle)
        moved = (rotation @ dynamic.means[..., None])[..., 0] + translation
        save_gaussians(out_dir / "gaussians_dynamic.ply", dynamic, moved, rotation)

    if seq.gt is not None:
        to_reconstruction = world_alignment(seq, frontend).inverse()
        for camera in seq.gt.val_cameras:
            pose = invert_se3(to_reconstruction.apply_to_c2w(camera.c2w[None]))[0]
            view = render_views(
                scene,
                K,
                pose[None].expand(len(camera.frames), -1, -1),
                camera.frames,
                width,
                height,
            )
            save_gif(
                out_dir / f"{camera.name}.gif",
                side_by_side(camera.images, view["color"]),
                fps / 3.0,
                scale,
            )
    logger.info("outputs written to %s", out_dir)
