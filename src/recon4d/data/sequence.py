"""Containers for a video sequence and its (optional) ground truth."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from recon4d.geometry.camera import Intrinsics

if TYPE_CHECKING:
    from recon4d.data.synthetic.oracle import SceneOracle


@dataclass
class ValCamera:
    """A held-out camera observing the scene at a subset of the training timestamps."""

    name: str
    c2w: Tensor
    """``(4, 4)`` camera-to-world pose (the camera is static)."""
    frames: list[int]
    """Indices of the training timestamps at which this camera was rendered."""
    images: Tensor
    """``(F, H, W, 3)`` ground-truth images."""
    depth: Tensor
    """``(F, H, W)`` ground-truth z-depth."""
    dynamic_mask: Tensor
    """``(F, H, W)`` True on pixels showing a moving object."""
    covisible: Tensor
    """``(F, H, W)`` True where the surface point is also observed by the training video."""


@dataclass
class GroundTruth:
    """Exact ground truth of a synthetic sequence."""

    c2w: Tensor
    """``(T, 4, 4)`` camera-to-world poses."""
    depth: Tensor
    """``(T, H, W)`` z-depth."""
    obj: Tensor
    """``(T, H, W)`` object index per pixel."""
    dynamic_mask: Tensor
    """``(T, H, W)`` True on pixels showing a moving object."""
    oracle: SceneOracle
    """Answers arbitrary ground-truth queries (tracks, flow, surface point clouds)."""
    val_cameras: list[ValCamera] = field(default_factory=list)


@dataclass
class VideoSequence:
    """A monocular video, optionally with calibration and ground truth.

    Attributes:
        name: identifier used in logs and output paths.
        images: ``(T, H, W, 3)`` float32 RGB in ``[0, 1]``.
        timestamps: ``(T,)`` normalised times in ``[0, 1]``.
        intrinsics: known calibration, or ``None`` when it has to be estimated.
        test_every: every ``test_every``-th frame is held out from scene optimisation and
            used to measure view synthesis quality (0 keeps all frames for training).
        gt: ground truth, available for synthetic sequences.
    """

    name: str
    images: Tensor
    timestamps: Tensor
    intrinsics: Intrinsics | None = None
    test_every: int = 8
    gt: GroundTruth | None = None

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"images must be (T, H, W, 3), got {tuple(self.images.shape)}")
        if self.timestamps.shape != (self.images.shape[0],):
            raise ValueError("timestamps must have one entry per frame")

    @property
    def num_frames(self) -> int:
        return self.images.shape[0]

    @property
    def height(self) -> int:
        return self.images.shape[1]

    @property
    def width(self) -> int:
        return self.images.shape[2]

    @property
    def test_indices(self) -> list[int]:
        if self.test_every <= 0:
            return []
        offset = self.test_every // 2
        return [i for i in range(self.num_frames) if i % self.test_every == offset]

    @property
    def train_indices(self) -> list[int]:
        held_out = set(self.test_indices)
        return [i for i in range(self.num_frames) if i not in held_out]

    def K(self, dtype: torch.dtype = torch.float32) -> Tensor:
        if self.intrinsics is None:
            raise ValueError(f"sequence {self.name!r} has no known intrinsics")
        return self.intrinsics.matrix(dtype)
