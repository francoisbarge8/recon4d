"""Image, video and overlay helpers for qualitative results."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image
from torch import Tensor

from recon4d.types import Tracks


def to_uint8(image: Tensor) -> np.ndarray:
    """Float image(s) in ``[0, 1]`` to ``uint8`` arrays."""
    return (image.detach().cpu().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()


def colorize(values: Tensor, vmin: float, vmax: float, cmap: str = "turbo") -> Tensor:
    """Map a scalar field ``(..., H, W)`` to RGB ``(..., H, W, 3)`` with a matplotlib colormap."""
    normalised = ((values - vmin) / max(vmax - vmin, 1e-12)).clamp(0.0, 1.0)
    lut = torch.tensor(colormaps[cmap](np.linspace(0.0, 1.0, 256))[:, :3], dtype=torch.float32)
    return lut[(normalised * 255.0).round().to(torch.int64).cpu()]


def colorize_depth(
    depth: Tensor, valid: Tensor | None = None, near: float | None = None, far: float | None = None
) -> Tensor:
    """Colour-code depth maps in inverse-depth space (near = warm, far = cold).

    ``near`` / ``far`` default to the 2nd / 98th percentiles of the valid depths; pass them
    explicitly to make several maps comparable. Invalid pixels are drawn black.
    """
    if valid is None:
        valid = depth > 0
    if not valid.any():
        return torch.zeros(*depth.shape, 3)
    values = depth[valid].flatten()
    if values.numel() > 200_000:
        values = values[:: values.numel() // 200_000]
    near = float(torch.quantile(values, 0.02)) if near is None else near
    far = float(torch.quantile(values, 0.98)) if far is None else far
    inverse = 1.0 / depth.clamp_min(1e-6)
    image = colorize(inverse, 1.0 / far, 1.0 / near, "turbo")
    return torch.where(valid[..., None].cpu(), image, torch.zeros_like(image))


def overlay_mask(image: Tensor, mask: Tensor, color=(1.0, 0.2, 0.2), alpha: float = 0.5) -> Tensor:
    """Tint the pixels of ``image (..., H, W, 3)`` selected by ``mask (..., H, W)``."""
    tint = torch.tensor(color, dtype=image.dtype)
    blended = (1.0 - alpha) * image + alpha * tint
    return torch.where(mask[..., None], blended, image)


def upscale(frames: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour enlargement of ``(..., H, W, 3)`` images by an integer factor."""
    if factor <= 1:
        return frames
    return frames.repeat(factor, axis=-3).repeat(factor, axis=-2)


def save_image(path: str | Path, image: Tensor, scale: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(upscale(to_uint8(image), scale)).save(path)


def save_gif(path: str | Path, frames: Tensor, fps: float = 12.0, scale: int = 1) -> None:
    """Write ``frames (T, H, W, 3)`` as a looping GIF with a shared adaptive palette."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = upscale(to_uint8(frames), scale)
    # A single palette for all frames avoids the flicker of per-frame quantisation.
    mosaic = Image.fromarray(np.concatenate(list(array[:: max(len(array) // 8, 1)]), axis=1))
    palette = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in array]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000.0 / fps),
        loop=0,
        optimize=False,
    )


def save_video(path: str | Path, frames: Tensor, fps: float = 12.0, scale: int = 1) -> None:
    """Write ``frames (T, H, W, 3)`` as an MP4 (OpenCV's ``mp4v`` encoder)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = upscale(to_uint8(frames), scale)
    height, width = array.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in array:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def side_by_side(*panels: Tensor, gap: int = 2) -> Tensor:
    """Concatenate image panels ``(..., H, W, 3)`` horizontally with a white gap."""
    parts = []
    for index, panel in enumerate(panels):
        if index > 0:
            parts.append(torch.ones(*panel.shape[:-2], gap, 3, dtype=panel.dtype))
        parts.append(panel.cpu())
    return torch.cat(parts, dim=-2)


def draw_tracks(
    images: Tensor, tracks: Tracks, trail: int = 8, max_tracks: int = 400, seed: int = 0
) -> Tensor:
    """Draw point tracks and their recent trails over a video.

    Static tracks are drawn in cyan and dynamic ones in orange when labels are available;
    otherwise each track gets its own colour.
    """
    frames = to_uint8(images).copy()
    n = len(tracks)
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(n, generator=generator)[:max_tracks]
    uv = tracks.uv[chosen].cpu().numpy() - 0.5  # OpenCV pixel centres are integers
    visible = tracks.visible[chosen].cpu().numpy()
    if tracks.dynamic is not None:
        dynamic = tracks.dynamic[chosen].cpu().numpy()
        colors = np.where(dynamic[:, None], (255, 140, 0), (0, 220, 255))
    else:
        colors = (
            colormaps["hsv"](np.linspace(0, 1, len(chosen), endpoint=False))[:, :3] * 255
        ).astype(int)
    for t in range(frames.shape[0]):
        canvas = np.ascontiguousarray(frames[t])
        for i in range(len(chosen)):
            if not visible[i, t]:
                continue
            color = tuple(int(c) for c in colors[i])
            start = t
            while start > 0 and t - start < trail and visible[i, start - 1]:
                start -= 1
            if start < t:
                points = np.round(uv[i, start : t + 1] * 4).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA, shift=2)
            center = tuple(int(v) for v in np.round(uv[i, t] * 4))
            cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA, shift=2)
        frames[t] = canvas
    return torch.from_numpy(frames).to(torch.float32) / 255.0
