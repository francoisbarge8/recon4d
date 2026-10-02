"""Kanade-Lucas-Tomasi tracking with forward-backward verification.

A classical, training-free baseline: Shi-Tomasi corners are followed frame to frame with
pyramidal Lucas-Kanade (OpenCV). A step is accepted only if tracking it back returns to
where it started (forward-backward check, Kalal et al., ICPR 2010); a point that fails is
considered lost for the rest of the video in that direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor

from recon4d.frontend.tracking.base import PointTracker, empty_tracks
from recon4d.types import Tracks


def to_gray_uint8(images: Tensor) -> np.ndarray:
    """``(T, H, W, 3)`` float RGB in ``[0, 1]`` to ``(T, H, W)`` uint8 luminance."""
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=images.dtype)
    gray = (images * weights).sum(dim=-1)
    return (gray.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()


@dataclass
class KLTConfig:
    """Parameters of the KLT tracker.

    Attributes:
        window: side of the Lucas-Kanade window in pixels.
        levels: number of pyramid levels above the full resolution.
        refine_window: side of the window of a second, single-scale Lucas-Kanade pass
            started from the result of the first one (0 disables it). A large window on
            a pyramid is robust to large motions but averages the motion of everything it
            covers; a small window refines the position on the point's own surface.
        fb_threshold: maximum forward-backward round-trip error (pixels). It has to be
            strict: the tracks that survive a loose check are precisely the ones sliding
            along occlusion boundaries, and their systematic error biases the camera poses
            (a threshold of 0.7 px instead of 0.4 px costs 2.5% of scale consistency
            between trajectory and structure on the benchmark).
        max_corners: corners detected per keyframe.
        quality: Shi-Tomasi quality level relative to the best corner.
        min_distance: minimum spacing between tracked points (pixels).
        keyframe_interval: new corners are detected every this many frames, away from the
            points already being tracked, so that newly revealed regions get covered.
        border: points closer than this to the image border are dropped.
    """

    window: int = 13
    levels: int = 2
    refine_window: int = 5
    fb_threshold: float = 0.4
    max_corners: int = 600
    quality: float = 0.01
    min_distance: int = 5
    keyframe_interval: int = 6
    border: float = 2.0


class KLTTracker(PointTracker):
    name = "klt"

    def __init__(self, cfg: KLTConfig | None = None) -> None:
        self.cfg = cfg or KLTConfig()

    def _lk(self, source: np.ndarray, target: np.ndarray, points: np.ndarray) -> tuple:
        """Pyramidal Lucas-Kanade, optionally refined with a small single-scale window."""
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        cfg = self.cfg
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            source,
            target,
            points,
            None,
            winSize=(cfg.window, cfg.window),
            maxLevel=cfg.levels,
            criteria=criteria,
        )
        if cfg.refine_window > 0:
            moved, refined_status, _ = cv2.calcOpticalFlowPyrLK(
                source,
                target,
                points,
                moved.copy(),
                winSize=(cfg.refine_window, cfg.refine_window),
                maxLevel=0,
                criteria=criteria,
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            )
            status = status & refined_status
        return moved, status

    def _step(
        self, prev: np.ndarray, nxt: np.ndarray, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Track ``points (N, 2)`` (OpenCV coordinates) from ``prev`` to ``nxt``."""
        if points.shape[0] == 0:
            return points, np.zeros(0, dtype=bool)
        p0 = points.reshape(-1, 1, 2).astype(np.float32)
        p1, status1 = self._lk(prev, nxt, p0)
        back, status0 = self._lk(nxt, prev, p1)
        round_trip = np.linalg.norm((back - p0).reshape(-1, 2), axis=1)
        p1 = p1.reshape(-1, 2)
        height, width = prev.shape
        b = self.cfg.border
        inside = (
            (p1[:, 0] >= b)
            & (p1[:, 0] <= width - 1 - b)
            & (p1[:, 1] >= b)
            & (p1[:, 1] <= height - 1 - b)
        )
        ok = (status1.reshape(-1) == 1) & (status0.reshape(-1) == 1) & inside
        ok &= round_trip < self.cfg.fb_threshold
        return p1, ok

    def track_queries(self, images: Tensor, query_frames: Tensor, query_uv: Tensor) -> Tracks:
        gray = to_gray_uint8(images)
        n_frames = gray.shape[0]
        n = query_uv.shape[0]
        # OpenCV puts pixel centres at integer coordinates: shift by half a pixel.
        uv = np.zeros((n, n_frames, 2), dtype=np.float32)
        visible = np.zeros((n, n_frames), dtype=bool)
        start = query_uv.cpu().numpy().astype(np.float32) - 0.5
        frames = query_frames.cpu().numpy()
        for q in np.unique(frames):
            index = np.flatnonzero(frames == q)
            uv[index, q] = start[index]
            visible[index, q] = True
            for direction in (1, -1):
                points = start[index].copy()
                alive = np.ones(index.shape[0], dtype=bool)
                t = int(q)
                while 0 <= t + direction < n_frames:
                    if alive.any():
                        moved, ok = self._step(gray[t], gray[t + direction], points[alive])
                        survivors = np.flatnonzero(alive)
                        points[survivors[ok]] = moved[ok]
                        alive[survivors[~ok]] = False
                    t += direction
                    # A lost point keeps its last position (and is flagged invisible).
                    uv[index, t] = points
                    visible[index[alive], t] = True
        return Tracks(
            torch.from_numpy(uv) + 0.5, torch.from_numpy(visible), query_frame=query_frames.clone()
        )

    def _detect_and_track(self, images: Tensor) -> tuple[Tensor, Tensor, Tracks]:
        """Detect Shi-Tomasi corners on keyframes and track them through the video.

        Each keyframe only receives corners away from the points already being tracked,
        which is why detection and tracking have to be interleaved.
        """
        cfg = self.cfg
        gray = to_gray_uint8(images)
        n_frames, height, width = gray.shape
        query_frames: list[Tensor] = []
        query_uv: list[Tensor] = []
        tracked: Tracks | None = None
        for keyframe in range(0, n_frames, cfg.keyframe_interval):
            mask = np.full((height, width), 255, dtype=np.uint8)
            border = int(np.ceil(cfg.border)) + 1
            mask[:border] = mask[-border:] = 0
            mask[:, :border] = mask[:, -border:] = 0
            if tracked is not None:
                alive = tracked.visible[:, keyframe]
                for x, y in (tracked.uv[alive, keyframe] - 0.5).round().to(torch.int64).tolist():
                    cv2.circle(mask, (x, y), cfg.min_distance, 0, -1)
            corners = cv2.goodFeaturesToTrack(
                gray[keyframe],
                maxCorners=cfg.max_corners,
                qualityLevel=cfg.quality,
                minDistance=cfg.min_distance,
                mask=mask,
            )
            if corners is None:
                continue
            uv = torch.from_numpy(corners.reshape(-1, 2)) + 0.5
            frames = torch.full((uv.shape[0],), keyframe, dtype=torch.int64)
            new = self.track_queries(images, frames, uv)
            tracked = new if tracked is None else Tracks.concatenate([tracked, new])
            query_frames.append(frames)
            query_uv.append(uv)
        if tracked is None:
            return torch.zeros(0, dtype=torch.int64), torch.zeros(0, 2), empty_tracks(0, n_frames)
        return torch.cat(query_frames), torch.cat(query_uv), tracked

    def sample_queries(self, images: Tensor) -> tuple[Tensor, Tensor]:
        query_frames, query_uv, _ = self._detect_and_track(images)
        return query_frames, query_uv

    def track(self, images: Tensor) -> Tracks:
        return self._detect_and_track(images)[2]
