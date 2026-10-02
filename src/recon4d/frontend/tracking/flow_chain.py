"""Dense point tracking by chaining optical flow.

Flow is estimated once between consecutive frames, in both directions. A point is then
advanced frame by frame by sampling the flow at its current sub-pixel position. It is
declared lost (occluded or unreliable) as soon as the forward-backward round trip at its
position exceeds a threshold. This is the "flow chaining" baseline of the point-tracking
literature: accurate over short ranges, unable to recover a point after an occlusion.

Because the flow fields are shared by all points, tracking a dense grid costs almost
nothing more than tracking a few points, which makes this the tracker of choice for
covering moving objects densely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.frontend.flow import FlowEstimator, forward_backward_error
from recon4d.frontend.tracking.base import PointTracker, grid_queries
from recon4d.geometry.camera import in_image, sample_bilinear
from recon4d.types import Tracks


@dataclass
class FlowChainConfig:
    """Parameters of the flow-chaining tracker.

    Attributes:
        fb_threshold: maximum forward-backward error (pixels) of a usable flow vector.
        grid_stride: spacing (pixels) of the query grid seeded on each keyframe.
        keyframe_interval: a new grid is seeded every this many frames.
        min_distance: a new query is dropped when a live track already passes within this
            distance (pixels) of it, to avoid tracking the same surface point twice.
        border: points closer than this to the image border are considered lost.
    """

    fb_threshold: float = 0.5
    grid_stride: int = 4
    keyframe_interval: int = 6
    min_distance: float = 2.0
    border: float = 1.0


class FlowChainTracker(PointTracker):
    """Chains the flow of a :class:`~recon4d.frontend.flow.FlowEstimator`."""

    def __init__(self, flow: FlowEstimator, cfg: FlowChainConfig | None = None) -> None:
        self.flow = flow
        self.cfg = cfg or FlowChainConfig()
        self.name = f"{flow.name}-chain"
        self._cache: tuple[Tensor, Tensor, Tensor, Tensor, Tensor] | None = None

    def _flows(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward / backward flows and their round-trip errors (cached for the last video)."""
        if self._cache is None or self._cache[0] is not images:
            forward, backward = self.flow.sequence(images)
            fb_forward = torch.stack(
                [forward_backward_error(f, b) for f, b in zip(forward, backward, strict=True)]
            )
            fb_backward = torch.stack(
                [forward_backward_error(b, f) for f, b in zip(forward, backward, strict=True)]
            )
            self._cache = (images, forward, backward, fb_forward, fb_backward)
        return self._cache[1:]

    def track_queries(self, images: Tensor, query_frames: Tensor, query_uv: Tensor) -> Tracks:
        cfg = self.cfg
        forward, backward, fb_forward, fb_backward = self._flows(images)
        n_frames, height, width, _ = images.shape
        n = query_uv.shape[0]
        uv = query_uv[:, None, :].repeat(1, n_frames, 1).to(torch.float32)
        visible = torch.zeros(n, n_frames, dtype=torch.bool)
        visible[torch.arange(n), query_frames] = True
        for direction, flows, errors in ((1, forward, fb_forward), (-1, backward, fb_backward)):
            position = query_uv.clone().to(torch.float32)
            alive = torch.ones(n, dtype=torch.bool)
            frames = range(n_frames - 1) if direction == 1 else range(n_frames - 1, 0, -1)
            for t in frames:
                # Flow fields are indexed by the earlier frame of each pair.
                pair = t if direction == 1 else t - 1
                active = alive & (query_frames <= t if direction == 1 else query_frames >= t)
                if active.any():
                    maps = torch.cat([flows[pair], errors[pair][..., None]], dim=-1).permute(
                        2, 0, 1
                    )
                    sampled = sample_bilinear(maps, position[active])
                    moved = position[active] + sampled[:, :2]
                    ok = (sampled[:, 2] < cfg.fb_threshold) & in_image(
                        moved, height, width, cfg.border
                    )
                    index = torch.nonzero(active)[:, 0]
                    position[index[ok]] = moved[ok]
                    alive[index[~ok]] = False
                target = t + direction
                started = query_frames <= t if direction == 1 else query_frames >= t
                # Lost points keep their last position and are flagged invisible.
                uv[started, target] = position[started]
                visible[alive & started, target] = True
        return Tracks(uv, visible, query_frame=query_frames.clone())

    def _detect_and_track(self, images: Tensor) -> tuple[Tensor, Tensor, Tracks]:
        cfg = self.cfg
        n_frames, height, width, _ = images.shape
        grid = grid_queries(height, width, cfg.grid_stride, margin=int(cfg.border) + 1)
        query_frames: list[Tensor] = []
        query_uv: list[Tensor] = []
        tracked: Tracks | None = None
        for keyframe in range(0, n_frames, cfg.keyframe_interval):
            queries = grid
            if tracked is not None:
                live = tracked.uv[tracked.visible[:, keyframe], keyframe]
                if live.shape[0] > 0:
                    # Occupancy test on a pixel grid: cheap and good enough at this scale.
                    occupied = torch.zeros(height, width, dtype=torch.bool)
                    radius = max(1, round(cfg.min_distance))
                    col = live[:, 0].floor().clamp(0, width - 1).to(torch.int64)
                    row = live[:, 1].floor().clamp(0, height - 1).to(torch.int64)
                    occupied[row, col] = True
                    occupied = (
                        torch.nn.functional.max_pool2d(
                            occupied[None, None].float(), 2 * radius + 1, stride=1, padding=radius
                        )[0, 0]
                        > 0
                    )
                    q_col = queries[:, 0].floor().to(torch.int64)
                    q_row = queries[:, 1].floor().to(torch.int64)
                    queries = queries[~occupied[q_row, q_col]]
            if queries.shape[0] == 0:
                continue
            frames = torch.full((queries.shape[0],), keyframe, dtype=torch.int64)
            new = self.track_queries(images, frames, queries)
            tracked = new if tracked is None else Tracks.concatenate([tracked, new])
            query_frames.append(frames)
            query_uv.append(queries)
        if tracked is None:
            raise RuntimeError("the video is too small for the query grid")
        return torch.cat(query_frames), torch.cat(query_uv), tracked

    def sample_queries(self, images: Tensor) -> tuple[Tensor, Tensor]:
        query_frames, query_uv, _ = self._detect_and_track(images)
        return query_frames, query_uv

    def track(self, images: Tensor) -> Tracks:
        return self._detect_and_track(images)[2]
