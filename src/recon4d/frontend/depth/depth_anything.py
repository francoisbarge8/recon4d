"""Depth Anything V2 (Yang et al., NeurIPS 2024) through Hugging Face ``transformers``.

The relative checkpoints predict an affine-invariant inverse depth (larger = closer); the
metric checkpoints predict depth in metres. The weights are downloaded on first use and
``transformers`` is an optional dependency (``pip install recon4d[depth]``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from recon4d.frontend.depth.base import DepthEstimator

DEFAULT_CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2(DepthEstimator):
    """Pretrained monocular depth.

    Args:
        checkpoint: Hugging Face model id. Checkpoints with ``Metric`` in their name are
            treated as metric depth (``kind="scale"``), the others as affine-invariant
            inverse depth (``kind="disparity"``).
        device: device the network runs on.
        batch_size: number of frames per forward pass.
    """

    name = "depth-anything-v2"

    def __init__(
        self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cpu", batch_size: int = 8
    ) -> None:
        try:
            from transformers import (
                AutoImageProcessor,
                AutoModelForDepthEstimation,
            )
        except ImportError as error:
            raise ImportError(
                "Depth Anything needs the optional 'transformers' package: "
                "pip install recon4d[depth]"
            ) from error
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.kind = "scale" if "metric" in checkpoint.lower() else "disparity"

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        height, width = images.shape[1:3]
        frames = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
        outputs = []
        for start in range(0, len(frames), self.batch_size):
            batch = list(frames[start : start + self.batch_size])
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            predicted = self.model(**inputs).predicted_depth  # (B, h, w)
            resized = F.interpolate(
                predicted[:, None].float(),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            outputs.append(resized[:, 0].cpu())
        prediction = torch.cat(outputs)
        # Inverse depth must stay positive for the alignment (bicubic resizing can undershoot).
        return prediction.clamp_min(1e-6)
