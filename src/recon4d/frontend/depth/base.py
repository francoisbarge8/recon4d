"""Common interface of monocular depth estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class DepthEstimator(ABC):
    """Predicts one depth map per frame, independently of the other frames.

    Attributes:
        name: identifier used in logs and result tables.
        kind: what the output means, which determines how it is aligned to the camera
            poses (see :mod:`recon4d.fusion.depth_align`):

            * ``"scale"``: depth up to an unknown scale (also used for metric depth);
            * ``"disparity"``: inverse depth up to an unknown scale and shift, the
              affine-invariant output of MiDaS-style models.
    """

    name: str = "depth"
    kind: str = "scale"

    @abstractmethod
    def predict(self, images: Tensor) -> Tensor:
        """Predict ``(T, H, W)`` maps for a video ``(T, H, W, 3)`` with values in ``[0, 1]``."""
