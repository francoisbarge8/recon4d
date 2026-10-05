"""CoTracker3 (Karaev et al., 2024) through ``torch.hub``.

A transformer that tracks many points jointly and through occlusions, the kind of
long-range tracker 4D reconstruction methods are built on. The weights are downloaded on
first use; a GPU is strongly recommended.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from recon4d.frontend.tracking.base import PointTracker, grid_queries
from recon4d.types import Tracks


@dataclass
class CoTrackerConfig:
    """Parameters of the CoTracker wrapper.

    Attributes:
        model: ``torch.hub`` entry point (``cotracker3_offline`` sees the whole video).
        repo: ``torch.hub`` repository, or a local clone of it (no GitHub access needed;
            the weights still come from Hugging Face).
        grid_stride: spacing (pixels) of the query grid seeded on each keyframe.
        keyframe_interval: a query grid is seeded every this many frames.
        visibility_threshold: predicted visibility above which a point counts as visible.
        chunk: maximum number of query points per forward pass.
    """

    model: str = "cotracker3_offline"
    repo: str = "facebookresearch/co-tracker"
    grid_stride: int = 4
    keyframe_interval: int = 8
    visibility_threshold: float = 0.5
    chunk: int = 4096


class CoTracker(PointTracker):
    name = "cotracker3"

    def __init__(self, cfg: CoTrackerConfig | None = None, device: str = "cpu") -> None:
        self.cfg = cfg or CoTrackerConfig()
        self.device = device
        source = "local" if Path(self.cfg.repo).is_dir() else "github"
        model = torch.hub.load(self.cfg.repo, self.cfg.model, source=source)
        self.model = model.to(device).eval()

    @torch.no_grad()
    def track_queries(self, images: Tensor, query_frames: Tensor, query_uv: Tensor) -> Tracks:
        # CoTracker expects (B, T, 3, H, W) in [0, 255] and puts pixel centres at integer
        # coordinates, hence the half-pixel shifts.
        video = (images.permute(0, 3, 1, 2)[None] * 255.0).to(self.device)
        queries = torch.cat([query_frames[:, None].to(query_uv.dtype), query_uv - 0.5], dim=1)
        uv, visible = [], []
        for part in torch.split(queries, self.cfg.chunk):
            tracks, visibility = self.model(
                video, queries=part[None].to(self.device), backward_tracking=True
            )
            uv.append(tracks[0].permute(1, 0, 2).cpu() + 0.5)  # (N, T, 2)
            visibility = visibility[0].permute(1, 0).cpu()
            if visibility.dtype != torch.bool:
                visibility = visibility > self.cfg.visibility_threshold
            visible.append(visibility)
        uv, visible = torch.cat(uv), torch.cat(visible)
        visible[torch.arange(len(query_frames)), query_frames] = True
        return Tracks(uv.to(torch.float32), visible, query_frame=query_frames.clone())

    def sample_queries(self, images: Tensor) -> tuple[Tensor, Tensor]:
        n_frames, height, width, _ = images.shape
        grid = grid_queries(height, width, self.cfg.grid_stride, margin=2)
        frames = torch.arange(0, n_frames, self.cfg.keyframe_interval)
        return frames.repeat_interleave(len(grid)), grid.repeat(len(frames), 1)
