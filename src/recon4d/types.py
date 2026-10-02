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
        query_frame: optional ``(N,)`` frame at which each track was initialised. There the
            position is exact by construction, which is what identifies the tracked surface
            point when a tracker is evaluated.
    """

    uv: Tensor
    visible: Tensor
    dynamic: Tensor | None = None
    query_frame: Tensor | None = None

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

    def _map(self, fn) -> Tracks:
        return Tracks(
            fn(self.uv),
            fn(self.visible),
            None if self.dynamic is None else fn(self.dynamic),
            None if self.query_frame is None else fn(self.query_frame),
        )

    def subset(self, index: Tensor) -> Tracks:
        """Select tracks with a boolean mask or an index tensor."""
        return self._map(lambda tensor: tensor[index])

    def to(self, device: torch.device | str) -> Tracks:
        return self._map(lambda tensor: tensor.to(device))

    def with_labels(self, dynamic: Tensor) -> Tracks:
        """Copy of the tracks carrying the given dynamic / static labels."""
        return Tracks(self.uv, self.visible, dynamic, self.query_frame)

    def first_visible(self) -> Tensor:
        """``(N,)`` index of the first frame in which each track is observed."""
        return self.visible.to(torch.int64).argmax(dim=1)

    def queries(self) -> tuple[Tensor, Tensor]:
        """Frame and position ``(N,), (N, 2)`` identifying the tracked surface points.

        Falls back to the first visible frame for tracks without a recorded query.
        """
        frames = self.first_visible() if self.query_frame is None else self.query_frame
        return frames, self.uv[torch.arange(len(self), device=self.uv.device), frames]

    @classmethod
    def concatenate(cls, parts: list[Tracks]) -> Tracks:
        def join(name: str) -> Tensor | None:
            values = [getattr(p, name) for p in parts]
            return None if any(v is None for v in values) else torch.cat(values)

        return cls(
            torch.cat([p.uv for p in parts]),
            torch.cat([p.visible for p in parts]),
            join("dynamic"),
            join("query_frame"),
        )


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

    def to(self, device: torch.device | str) -> TrainingData:
        """Copy of the data with every tensor on ``device``."""

        def move(tensor: Tensor | None) -> Tensor | None:
            return None if tensor is None else tensor.to(device)

        return TrainingData(
            images=self.images.to(device),
            K=self.K.to(device),
            w2c=self.w2c.to(device),
            train_frames=list(self.train_frames),
            depth=move(self.depth),
            depth_valid=move(self.depth_valid),
            dynamic_mask=move(self.dynamic_mask),
            tracks=None if self.tracks is None else self.tracks.to(device),
        )

    @property
    def num_frames(self) -> int:
        return self.images.shape[0]

    @property
    def height(self) -> int:
        return self.images.shape[1]

    @property
    def width(self) -> int:
        return self.images.shape[2]
