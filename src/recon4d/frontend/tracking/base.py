"""Common interface of point trackers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from recon4d.geometry.camera import pixel_centers
from recon4d.types import Tracks


def grid_queries(height: int, width: int, stride: int, margin: int = 0) -> Tensor:
    """``(N, 2)`` pixel centres on a regular grid of step ``stride``."""
    centers = pixel_centers(height, width)
    offset = stride // 2
    grid = centers[offset::stride, offset::stride].reshape(-1, 2)
    if margin > 0:
        keep = (
            (grid[:, 0] >= margin)
            & (grid[:, 0] <= width - margin)
            & (grid[:, 1] >= margin)
            & (grid[:, 1] <= height - margin)
        )
        grid = grid[keep]
    return grid


class PointTracker(ABC):
    """Tracks points through a video ``(T, H, W, 3)`` with values in ``[0, 1]``."""

    name: str = "tracker"

    @abstractmethod
    def track_queries(self, images: Tensor, query_frames: Tensor, query_uv: Tensor) -> Tracks:
        """Track given points.

        Args:
            images: ``(T, H, W, 3)`` video.
            query_frames: ``(N,)`` frame index at which each point is defined.
            query_uv: ``(N, 2)`` pixel coordinates of the points in their query frame.

        Returns:
            Tracks ``(N, T)``; every point is visible at its query frame.
        """

    @abstractmethod
    def sample_queries(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Choose the points worth tracking; returns ``(query_frames, query_uv)``."""

    def track(self, images: Tensor) -> Tracks:
        """Track an automatically chosen set of points covering the whole video."""
        query_frames, query_uv = self.sample_queries(images)
        return self.track_queries(images, query_frames, query_uv)


def empty_tracks(n: int, n_frames: int) -> Tracks:
    return Tracks(torch.zeros(n, n_frames, 2), torch.zeros(n, n_frames, dtype=torch.bool))
