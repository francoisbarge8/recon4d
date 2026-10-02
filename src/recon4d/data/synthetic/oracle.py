"""Exact ground-truth queries on a synthetic sequence.

Because the scene is analytic, ground truth is not limited to what was pre-rendered: the
oracle can lift *any* pixel of *any* frame to a material surface point and follow it
through time. Trackers, flow estimators and reconstructions are therefore evaluated on
their own query points rather than on a fixed annotation set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from recon4d.data.synthetic.primitives import DTYPE
from recon4d.data.synthetic.scene import Scene
from recon4d.geometry.camera import (
    in_image,
    invert_se3,
    pixel_centers,
    pixel_rays,
    project,
    transform_points,
)


@dataclass(frozen=True)
class GTTracks:
    """Ground-truth trajectories of ``N`` material points over ``T`` frames."""

    uv: Tensor
    """``(N, T, 2)`` pixel coordinates (defined even when the point is not visible)."""
    visible: Tensor
    """``(N, T)`` True when the point is inside the image and unoccluded."""
    xyz: Tensor
    """``(N, T, 3)`` world coordinates."""
    depth: Tensor
    """``(N, T)`` z-depth in each frame's camera."""
    dynamic: Tensor
    """``(N,)`` True for points lying on a moving object."""
    obj: Tensor
    """``(N,)`` object index."""


@dataclass(frozen=True)
class SurfaceCloud:
    """Ground-truth surface samples used for geometric evaluation."""

    static: Tensor
    """``(M, 3)`` static surface points observed by the training video."""
    dynamic: Tensor
    """``(D, T, 3)`` positions over time of points on moving objects."""
    dynamic_observed: Tensor
    """``(D,)`` True when the point is seen in at least ``min_views`` training frames."""
    dynamic_obj: Tensor
    """``(D,)`` object index of each dynamic point."""


class SceneOracle:
    """Ground-truth oracle for a scene filmed by a known camera trajectory.

    Args:
        scene: the analytic scene.
        K: ``(3, 3)`` intrinsics.
        c2w: ``(T, 4, 4)`` camera-to-world poses of the training video.
        times: ``(T,)`` normalised timestamps.
        height, width: image size.
    """

    def __init__(
        self, scene: Scene, K: Tensor, c2w: Tensor, times: Tensor, height: int, width: int
    ) -> None:
        self.scene = scene
        self.K = K.to(DTYPE)
        self.c2w = c2w.to(DTYPE)
        self.w2c = invert_se3(self.c2w)
        self.times = [float(t) for t in times]
        self.height = height
        self.width = width

    @property
    def num_frames(self) -> int:
        return len(self.times)

    # ----------------------------------------------------------------------- lifting

    def lift(self, frame: int, uv: Tensor, c2w: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Material points seen at pixels ``uv (N, 2)`` of ``frame``.

        ``c2w`` overrides the camera (used for validation cameras). Returns the object
        index ``(N,)`` and object-local coordinates ``(N, 3)``.
        """
        pose = self.c2w[frame] if c2w is None else c2w.to(DTYPE)
        origins, dirs = pixel_rays(self.K, pose, uv.to(DTYPE))
        hits = self.scene.intersect(origins, dirs, self.times[frame])
        return hits.obj, hits.local

    def positions(self, obj: Tensor, local: Tensor, frames: list[int] | None = None) -> Tensor:
        """World positions ``(N, F, 3)`` of material points at the given frames."""
        frames = list(range(self.num_frames)) if frames is None else frames
        return torch.stack(
            [self.scene.local_to_world(obj, local, self.times[f]) for f in frames], dim=1
        )

    def observe(
        self, obj: Tensor, local: Tensor, frames: list[int] | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project material points into training frames.

        Returns ``uv (N, F, 2)``, ``depth (N, F)``, ``visible (N, F)`` and world positions
        ``xyz (N, F, 3)``. A point is visible when it is in front of the camera, inside the
        image, and the ray from the camera centre reaches it unoccluded.
        """
        frames = list(range(self.num_frames)) if frames is None else frames
        xyz = self.positions(obj, local, frames)
        uvs, depths, visibles = [], [], []
        for k, f in enumerate(frames):
            cam = transform_points(self.w2c[f], xyz[:, k])
            uv, z = project(self.K, cam)
            inside = (z > 1e-6) & in_image(uv, self.height, self.width)
            unoccluded = self.scene.visible_from(xyz[:, k], self.c2w[f, :3, 3], self.times[f])
            uvs.append(uv)
            depths.append(z)
            visibles.append(inside & unoccluded)
        return (
            torch.stack(uvs, dim=1),
            torch.stack(depths, dim=1),
            torch.stack(visibles, dim=1),
            xyz,
        )

    # ------------------------------------------------------------------------ tracks

    def tracks(self, query_frames: Tensor, query_uv: Tensor) -> GTTracks:
        """Ground-truth tracks of the surface points seen at ``query_uv`` in ``query_frames``."""
        n = query_uv.shape[0]
        obj = torch.zeros(n, dtype=torch.int64)
        local = torch.zeros(n, 3, dtype=DTYPE)
        for f in torch.unique(query_frames).tolist():
            mask = query_frames == f
            obj[mask], local[mask] = self.lift(int(f), query_uv[mask])
        return self.tracks_of(obj, local)

    def tracks_of(self, obj: Tensor, local: Tensor) -> GTTracks:
        uv, depth, visible, xyz = self.observe(obj, local)
        return GTTracks(
            uv.to(torch.float32),
            visible,
            xyz.to(torch.float32),
            depth.to(torch.float32),
            self.scene.is_dynamic(obj),
            obj,
        )

    def random_tracks(self, n: int, seed: int = 0, dynamic_fraction: float = 0.4) -> GTTracks:
        """Tracks of ``n`` random material points, ``dynamic_fraction`` of them on movers."""
        rng = np.random.RandomState(seed)  # noqa: NPY002 - frozen legacy stream
        if not self.scene.dynamic_ids():
            dynamic_fraction = 0.0
        obj, local = self.scene.sample_surface_points(n, rng, dynamic_fraction=dynamic_fraction)
        return self.tracks_of(obj, local)

    def flow(self, src: int, dst: int) -> tuple[Tensor, Tensor]:
        """Dense optical flow from frame ``src`` to frame ``dst``.

        Returns ``flow (H, W, 2)`` in pixels and ``valid (H, W)``, True where the source
        pixel's surface point is visible in ``dst``.
        """
        centers = pixel_centers(self.height, self.width, dtype=DTYPE).reshape(-1, 2)
        obj, local = self.lift(src, centers)
        uv, _, visible, _ = self.observe(obj, local, [dst])
        flow = (uv[:, 0] - centers).to(torch.float32)
        return flow.reshape(self.height, self.width, 2), visible[:, 0].reshape(
            self.height, self.width
        )

    # ---------------------------------------------------------------- surface clouds

    def surface_cloud(
        self,
        n_static: int = 60000,
        n_dynamic: int = 6000,
        min_views: int = 2,
        frames: list[int] | None = None,
        seed: int = 0,
    ) -> SurfaceCloud:
        """Sample the ground-truth surface for Chamfer / F-score evaluation.

        Only surfaces actually observed by the training video are kept: a reconstruction
        cannot be blamed for geometry the camera never saw. ``frames`` restricts the
        visibility test to a subset of frames (default: all).
        """
        rng = np.random.RandomState(seed)  # noqa: NPY002 - frozen legacy stream
        frames = list(range(self.num_frames)) if frames is None else frames
        obj, local = self.scene.sample_surface_points(n_static, rng, dynamic_fraction=0.0)
        keep = ~self.scene.is_dynamic(obj)
        obj, local = obj[keep], local[keep]
        _, _, visible, xyz = self.observe(obj, local, frames)
        observed = visible.sum(dim=1) >= min_views
        static = xyz[observed, 0].to(torch.float32)

        if self.scene.dynamic_ids() and n_dynamic > 0:
            d_obj, d_local = self.scene.sample_surface_points(n_dynamic, rng, dynamic_fraction=1.0)
            _, _, d_visible, _ = self.observe(d_obj, d_local, frames)
            dynamic = self.positions(d_obj, d_local).to(torch.float32)
            dynamic_observed = d_visible.sum(dim=1) >= min_views
        else:
            d_obj = torch.zeros(0, dtype=torch.int64)
            dynamic = torch.zeros(0, self.num_frames, 3)
            dynamic_observed = torch.zeros(0, dtype=torch.bool)
        return SurfaceCloud(static, dynamic, dynamic_observed, d_obj)

    # ------------------------------------------------------------------ covisibility

    def covisible(
        self,
        obj: Tensor,
        local: Tensor,
        depth_maps: Tensor,
        obj_maps: Tensor,
        frames: list[int],
        min_views: int = 1,
    ) -> Tensor:
        """Whether material points are observed by the training video.

        Uses a depth-buffer test against the rendered training frames instead of ray
        casting, which keeps dense (per-pixel) queries cheap.

        Args:
            obj, local: material points.
            depth_maps, obj_maps: ``(T, H, W)`` ground-truth buffers of the training video.
            frames: training frames to test against.
            min_views: number of frames in which a point has to be seen.
        """
        count = torch.zeros(obj.shape[0], dtype=torch.int64)
        for f in frames:
            world = self.scene.local_to_world(obj, local, self.times[f])
            uv, z = project(self.K, transform_points(self.w2c[f], world))
            inside = (z > 1e-6) & in_image(uv, self.height, self.width)
            col = uv[:, 0].floor().clamp(0, self.width - 1).to(torch.int64)
            row = uv[:, 1].floor().clamp(0, self.height - 1).to(torch.int64)
            buffer_depth = depth_maps[f][row, col].to(DTYPE)
            same_surface = (obj_maps[f][row, col] == obj) & ((z - buffer_depth).abs() < 0.03 * z)
            count += (inside & same_surface).to(torch.int64)
        return count >= min_views
