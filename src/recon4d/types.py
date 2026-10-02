"""Data structures exchanged between the stages of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Tracks:
    """2D point tracks over a video.

    Attributes:
        uv: ``(N, T, 2)`` pixel coordinates (pixel centres at half-integers).
        visible: ``(N, T)`` True where the point is observed; ``uv`` is meaningless
            elsewhere.
        dynamic: optional ``(N,)`` label, True for points on independently moving objects
            (set by motion segmentation).
    """

    uv: Tensor
    visible: Tensor
    dynamic: Tensor | None = None

    def __post_init__(self) -> None:
        if self.uv.ndim != 3 or self.uv.shape[-1] != 2:
            raise ValueError(f"uv must be (N, T, 2), got {tuple(self.uv.shape)}")
        if self.visible.shape != self.uv.shape[:2]:
            raise ValueError("visible must be (N, T)")

    def __len__(self) -> int:
        return self.uv.shape[0]

    @property
    def num_frames(self) -> int:
        return self.uv.shape[1]

    def subset(self, index: Tensor) -> Tracks:
        """Select tracks with a boolean mask or an index tensor."""
        return Tracks(
            self.uv[index],
            self.visible[index],
            None if self.dynamic is None else self.dynamic[index],
        )

    def first_visible(self) -> Tensor:
        """``(N,)`` index of the first frame in which each track is observed."""
        return self.visible.to(torch.int64).argmax(dim=1)

    @classmethod
    def concatenate(cls, parts: list[Tracks]) -> Tracks:
        dynamic = None
        if all(p.dynamic is not None for p in parts):
            dynamic = torch.cat([p.dynamic for p in parts])
        return cls(torch.cat([p.uv for p in parts]), torch.cat([p.visible for p in parts]), dynamic)


@dataclass
class TrainingData:
    """Everything the scene optimisation consumes.

    Attributes:
        images: ``(T, H, W, 3)`` frames in ``[0, 1]``.
        K: ``(3, 3)`` intrinsics.
        w2c: ``(T, 4, 4)`` world-to-camera poses (estimated or ground truth).
        train_frames: indices of the frames used for optimisation.
        depth: optional ``(T, H, W)`` depth prior expressed in the same frame as ``w2c``.
        depth_valid: optional ``(T, H, W)`` mask of trustworthy prior depths.
        dynamic_mask: optional ``(T, H, W)`` mask of independently moving pixels.
        tracks: optional point tracks, with ``dynamic`` labels for 4D reconstruction.
    """

    images: Tensor
    K: Tensor
    w2c: Tensor
    train_frames: list[int]
    depth: Tensor | None = None
    depth_valid: Tensor | None = None
    dynamic_mask: Tensor | None = None
    tracks: Tracks | None = None

    @property
    def num_frames(self) -> int:
        return self.images.shape[0]

    @property
    def height(self) -> int:
        return self.images.shape[1]

    @property
    def width(self) -> int:
        return self.images.shape[2]
