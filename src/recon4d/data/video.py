"""Loading real footage as a :class:`~recon4d.data.sequence.VideoSequence`."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from recon4d.data.sequence import VideoSequence
from recon4d.geometry.camera import Intrinsics

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _resize(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink a frame so that its larger side is at most ``max_side`` (multiple of 8 sizes)."""
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    new_w = max(8, round(width * scale / 8) * 8)
    new_h = max(8, round(height * scale / 8) * 8)
    if (new_w, new_h) == (width, height):
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def _read_folder(path: Path) -> list[np.ndarray]:
    files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    frames = []
    for file in files:
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read image {file}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return frames


def load_video(
    path: str | Path,
    max_frames: int = 60,
    max_side: int = 384,
    start: int = 0,
    stride: int | None = None,
    fov_x_deg: float | None = None,
    test_every: int = 8,
) -> VideoSequence:
    """Load a video file or a folder of images.

    Args:
        max_frames: number of frames kept.
        max_side: frames are down-scaled so that their larger side does not exceed this.
        start: index of the first frame used.
        stride: temporal sub-sampling; by default the frames are spread uniformly over
            the rest of the video.
        fov_x_deg: horizontal field of view if known; otherwise the focal length is
            estimated by structure-from-motion.
        test_every: every ``test_every``-th frame is held out of the optimisation.
    """
    path = Path(path)
    frames = _read_folder(path) if path.is_dir() else _read_video(path)
    frames = frames[start:]
    if len(frames) < 3:
        raise ValueError(f"{path} holds {len(frames)} usable frame(s); at least 3 are needed")
    if stride is None:
        index = np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).round().astype(int)
    else:
        index = np.arange(0, len(frames), stride)[:max_frames]
    frames = [_resize(frames[i], max_side) for i in np.unique(index)]
    images = torch.from_numpy(np.stack(frames)).to(torch.float32) / 255.0
    height, width = images.shape[1:3]
    intrinsics = None if fov_x_deg is None else Intrinsics.from_fov(width, height, fov_x_deg)
    return VideoSequence(
        name=path.stem,
        images=images,
        timestamps=torch.linspace(0.0, 1.0, images.shape[0]),
        intrinsics=intrinsics,
        test_every=test_every,
    )
