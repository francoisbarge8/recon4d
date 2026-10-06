"""A compact monocular depth network trained on the synthetic benchmark distribution.

The benchmark needs a depth estimator that actually *predicts* depth from pixels, yet
general-purpose networks are trained on photographs, not on procedurally textured rooms.
This small U-Net (about 1.5 M parameters) is trained from scratch on random scenes sampled
from the benchmark's scene grammar (never on the benchmark scenes themselves), with the
scale-invariant loss of Eigen et al. (NeurIPS 2014) and a multi-scale gradient-matching
term (Li & Snavely, CVPR 2018) for sharp depth edges. Like every monocular network it
predicts depth up to scale and independently for each frame.

For real footage use :class:`recon4d.frontend.depth.depth_anything.DepthAnythingV2`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from recon4d.frontend.depth.base import DepthEstimator


@dataclass(frozen=True)
class TinyDepthConfig:
    """Architecture of :class:`TinyDepthNet`.

    Attributes:
        widths: channels of the encoder stages (each stage halves the resolution).
        groups: number of groups of the GroupNorm layers.
        coordinates: append the normalised pixel coordinates to the input, which lets the
            network learn layout priors that depend on the image position.
    """

    widths: tuple[int, ...] = (24, 48, 96, 160)
    groups: int = 8
    coordinates: bool = True


class _Block(nn.Module):
    """Two 3x3 convolutions with GroupNorm and SiLU, and a residual connection."""

    def __init__(self, in_channels: int, out_channels: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.skip = None if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.skip is None else self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + identity)


class TinyDepthNet(nn.Module):
    """U-Net predicting log-depth ``(B, H, W)`` from RGB ``(B, 3, H, W)`` in ``[0, 1]``."""

    def __init__(self, cfg: TinyDepthConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TinyDepthConfig()
        widths = self.cfg.widths
        in_channels = 3 + (2 if self.cfg.coordinates else 0)
        self.encoder = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoder.append(_Block(previous, width, self.cfg.groups))
            previous = width
        self.decoder = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.decoder.append(_Block(previous + width, width, self.cfg.groups))
            previous = width
        self.head = nn.Conv2d(previous, 1, 3, padding=1)

    def forward(self, images: Tensor) -> Tensor:
        x = images * 2.0 - 1.0
        if self.cfg.coordinates:
            batch, _, height, width = x.shape
            ys = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
            grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"))
            x = torch.cat([x, grid.expand(batch, 2, height, width)], dim=1)
        skips = []
        for index, block in enumerate(self.encoder):
            if index > 0:
                x = F.avg_pool2d(x, 2, ceil_mode=True)
            x = block(x)
            skips.append(x)
        skips.pop()
        for block in self.decoder:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)[:, 0]


def scale_invariant_loss(
    log_pred: Tensor, log_target: Tensor, variance_focus: float = 0.85
) -> Tensor:
    """Scale-invariant log loss: ``mean(d^2) - lambda * mean(d)^2`` with ``d`` the log error.

    With ``lambda = 1`` a global scale error costs nothing; values below 1 keep a small
    incentive to get the scale right, which stabilises training.
    """
    d = log_pred - log_target
    per_image = (d**2).mean(dim=(1, 2)) - variance_focus * d.mean(dim=(1, 2)) ** 2
    return per_image.mean()


def gradient_matching_loss(log_pred: Tensor, log_target: Tensor, scales: int = 3) -> Tensor:
    """Multi-scale L1 difference between the spatial gradients of the log-depth error."""
    d = (log_pred - log_target)[:, None]
    total = d.new_zeros(())
    for _ in range(scales):
        total = total + (d[..., :, 1:] - d[..., :, :-1]).abs().mean()
        total = total + (d[..., 1:, :] - d[..., :-1, :]).abs().mean()
        d = F.avg_pool2d(d, 2)
    return total / scales


def save_checkpoint(path: str | Path, model: TinyDepthNet, meta: dict | None = None) -> None:
    """Save the weights (half precision, to keep the file small) with the architecture."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: value.half() if value.is_floating_point() else value
        for name, value in model.state_dict().items()
    }
    torch.save({"config": asdict(model.cfg), "state": state, "meta": meta or {}}, path)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[TinyDepthNet, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    config["widths"] = tuple(config["widths"])
    model = TinyDepthNet(TinyDepthConfig(**config))
    model.load_state_dict(
        {
            name: value.float() if value.is_floating_point() else value
            for name, value in checkpoint["state"].items()
        }
    )
    return model.to(device).eval(), checkpoint.get("meta", {})


class LearnedDepth(DepthEstimator):
    """The in-domain :class:`TinyDepthNet`, loaded from a checkpoint.

    Frames are resized to the size the network was trained at (``image_size`` in the
    checkpoint metadata) and the prediction back to the size of the frames: the network
    learned how large things look in pixels, and at another size its error triples.
    """

    name = "tiny-depth"
    kind = "scale"

    def __init__(self, checkpoint: str | Path, device: str = "cpu", batch_size: int = 16) -> None:
        if not checkpoint or not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"depth checkpoint {str(checkpoint)!r} not found; train one with "
                "scripts/train_depth.py or pick another depth back-end"
            )
        self.model, self.meta = load_checkpoint(checkpoint, device)
        self.device = device
        self.batch_size = batch_size
        size = self.meta.get("image_size")
        # (height, width) of the training images; None runs at the size of the frames.
        self.image_size = None if size is None else (int(size[0]), int(size[1]))

    @torch.no_grad()
    def predict(self, images: Tensor) -> Tensor:
        size = images.shape[1:3]
        outputs = []
        for batch in torch.split(images, self.batch_size):
            batch = batch.permute(0, 3, 1, 2).to(self.device)
            resize = self.image_size is not None and tuple(size) != self.image_size
            if resize:
                batch = F.interpolate(
                    batch, self.image_size, mode="bilinear", align_corners=False, antialias=True
                )
            log_depth = self.model(batch)
            if resize:  # log-depth is the smoother quantity to interpolate
                log_depth = F.interpolate(
                    log_depth[:, None], size, mode="bilinear", align_corners=False
                )[:, 0]
            outputs.append(torch.exp(log_depth).cpu())
        return torch.cat(outputs)
