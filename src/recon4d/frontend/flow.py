"""Dense optical flow back-ends.

Flow is used for motion segmentation, for the temporal-consistency metrics of real videos
and by the flow-chaining point tracker. Two estimators are provided:

* :class:`DISFlow` - Dense Inverse Search (Kroeger et al., ECCV 2016) from OpenCV:
  training-free, fast on CPU, needs no weights;
* :class:`RAFTFlow` - RAFT (Teed & Deng, ECCV 2020) from torchvision: far more accurate,
  downloads pretrained weights on first use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from recon4d.geometry.camera import pixel_centers


class FlowEstimator(ABC):
    """Estimates dense flow between two RGB images ``(H, W, 3)`` in ``[0, 1]``."""

    name: str = "flow"

    @abstractmethod
    def estimate(self, source: Tensor, target: Tensor) -> Tensor:
        """Flow ``(H, W, 2)`` such that ``source(x)`` matches ``target(x + flow(x))``."""

    def between(self, images: Tensor, src: int, dst: int) -> Tensor:
        """Flow from frame ``src`` to frame ``dst`` of a video ``(T, H, W, 3)``."""
        return self.estimate(images[src], images[dst])

    def sequence(self, images: Tensor, step: int = 1) -> tuple[Tensor, Tensor]:
        """Forward and backward flows between frames ``t`` and ``t + step`` of a video.

        Returns ``forward (T - step, H, W, 2)`` (from ``t`` to ``t + step``) and
        ``backward (T - step, H, W, 2)`` (from ``t + step`` to ``t``).
        """
        forward, backward = [], []
        for t in range(images.shape[0] - step):
            forward.append(self.between(images, t, t + step))
            backward.append(self.between(images, t + step, t))
        return torch.stack(forward), torch.stack(backward)


class OracleFlow(FlowEstimator):
    """Ground-truth flow of a synthetic sequence (an analysis tool, not an estimator).

    The flow of an occluded pixel points to where its surface point *would* project, as a
    perfect estimator would extrapolate; the forward-backward check then flags it, exactly
    as it does for estimated flow.
    """

    name = "oracle"

    def __init__(self, oracle) -> None:
        self.oracle = oracle

    def estimate(self, source: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError("OracleFlow needs frame indices: use between()")

    def between(self, images: Tensor, src: int, dst: int) -> Tensor:
        return self.oracle.flow(src, dst)[0].to(images.dtype)


def _to_uint8(image: Tensor) -> np.ndarray:
    return (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()


class DISFlow(FlowEstimator):
    """OpenCV's Dense Inverse Search flow.

    Args:
        preset: ``"ultrafast"``, ``"fast"`` or ``"medium"`` (most accurate).
        finest_scale: finest pyramid level used (0 = full resolution). The OpenCV presets
            stop at a coarser level for speed, which is too coarse for small images.
    """

    name = "dis"
    _PRESETS: ClassVar[dict[str, int]] = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }

    def __init__(self, preset: str = "medium", finest_scale: int = 0) -> None:
        self.dis = cv2.DISOpticalFlow_create(self._PRESETS[preset])
        self.dis.setFinestScale(finest_scale)

    def estimate(self, source: Tensor, target: Tensor) -> Tensor:
        a = cv2.cvtColor(_to_uint8(source), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(_to_uint8(target), cv2.COLOR_RGB2GRAY)
        return torch.from_numpy(self.dis.calc(a, b, None)).to(source.dtype)


class RAFTFlow(FlowEstimator):
    """torchvision's RAFT; ``small=True`` selects the 1M-parameter variant.

    Images are padded to a multiple of 8 and, when small, up-sampled so that the coarsest
    feature map is not degenerate; the flow is scaled back accordingly.
    """

    name = "raft"

    def __init__(self, small: bool = True, iterations: int = 12, min_side: int = 256) -> None:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        if small:
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self.model.eval()
        self.iterations = iterations
        self.min_side = min_side

    @torch.no_grad()
    def estimate(self, source: Tensor, target: Tensor) -> Tensor:
        height, width = source.shape[:2]
        scale = max(1.0, self.min_side / min(height, width))
        new_h = round(height * scale / 8.0) * 8
        new_w = round(width * scale / 8.0) * 8

        def prepare(image: Tensor) -> Tensor:
            x = image.permute(2, 0, 1)[None].float() * 2.0 - 1.0
            return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        flow = self.model(prepare(source), prepare(target), num_flow_updates=self.iterations)[-1]
        flow = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)[0]
        flow = flow * torch.tensor([width / new_w, height / new_h])[:, None, None]
        return flow.permute(1, 2, 0).to(source.dtype)


def forward_backward_error(forward: Tensor, backward: Tensor) -> Tensor:
    """Round-trip error ``|f(x) + b(x + f(x))|`` of a flow pair, ``(H, W)`` in pixels.

    Small values indicate a reliable, non-occluded correspondence.
    """
    height, width = forward.shape[:2]
    target = pixel_centers(height, width, dtype=forward.dtype) + forward
    grid = target / torch.tensor([width, height], dtype=forward.dtype) * 2.0 - 1.0
    back = F.grid_sample(
        backward.permute(2, 0, 1)[None],
        grid[None],
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0].permute(1, 2, 0)
    error = (forward + back).norm(dim=-1)
    outside = (
        (target[..., 0] < 0)
        | (target[..., 0] > width)
        | (target[..., 1] < 0)
        | (target[..., 1] > height)
    )
    return torch.where(outside, torch.full_like(error, float("inf")), error)
