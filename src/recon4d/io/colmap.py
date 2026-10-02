"""Sparse reconstructions in COLMAP's text format.

COLMAP uses the conventions of this package: world-to-camera poses, ``+x`` right / ``+y``
down / ``+z`` forward, pixel centres at half-integers. Models are therefore exchanged
without any change of frame. Two uses:

* **export** (:func:`export_dataset`): the front-end's poses and sparse points become an
  ``images/`` + ``sparse/0/`` folder, the layout expected by COLMAP's GUI and by the
  training scripts of the reference 3DGS implementation, gsplat and Nerfstudio;
* **import** (:func:`read_model`): poses computed by COLMAP replace the track-based
  structure-from-motion (``poses=colmap``, see :mod:`recon4d.frontend.pose.colmap`).

Only the text format (``cameras.txt``, ``images.txt``, ``points3D.txt``) is handled;
``colmap model_converter`` translates to and from the binary one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from recon4d.geometry.camera import make_se3
from recon4d.geometry.rotations import quat_to_rotmat, rotmat_to_quat
from recon4d.types import Tracks
from recon4d.utils import get_logger

logger = get_logger(__name__)

# model name -> (number of parameters, index of fx, fy, cx, cy in the parameter list)
_CAMERA_MODELS: dict[str, tuple[int, tuple[int, int, int, int]]] = {
    "SIMPLE_PINHOLE": (3, (0, 0, 1, 2)),
    "PINHOLE": (4, (0, 1, 2, 3)),
    "SIMPLE_RADIAL": (4, (0, 0, 1, 2)),
    "RADIAL": (5, (0, 0, 1, 2)),
    "OPENCV": (8, (0, 1, 2, 3)),
}


@dataclass
class ColmapModel:
    """A sparse reconstruction with a single (pinhole) camera.

    Images are ordered by name; point indices refer to rows of ``points``.
    """

    K: Tensor
    """``(3, 3)`` intrinsics, float64."""
    width: int
    height: int
    names: list[str]
    """File name of every image."""
    w2c: Tensor
    """``(M, 4, 4)`` world-to-camera poses, float64."""
    points: Tensor = field(default_factory=lambda: torch.zeros(0, 3, dtype=torch.float64))
    """``(P, 3)`` 3D points."""
    colors: Tensor = field(default_factory=lambda: torch.zeros(0, 3, dtype=torch.uint8))
    """``(P, 3)`` RGB colours, uint8."""
    errors: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.float64))
    """``(P,)`` mean reprojection error of every point, in pixels."""
    keypoints: list[Tensor] = field(default_factory=list)
    """Per image, ``(n, 2)`` observed pixel positions (empty list: no observations)."""
    point_ids: list[Tensor] = field(default_factory=list)
    """Per image, ``(n,)`` index of the 3D point seen at each keypoint (-1: none)."""


def model_from_tracks(
    K: Tensor,
    width: int,
    height: int,
    w2c: Tensor,
    names: list[str],
    tracks: Tracks,
    points: Tensor,
    inlier: Tensor,
    colors: Tensor | None = None,
    errors: Tensor | None = None,
) -> ColmapModel:
    """Build a model from point tracks and their triangulated points.

    Args:
        points: ``(N, 3)`` one 3D point per track.
        inlier: ``(N, T)`` observations consistent with their point; tracks without any
            are left out, and frame ``t`` corresponds to ``names[t]``.
        colors: ``(N, 3)`` in ``[0, 1]`` (grey when omitted).
    """
    keep = torch.nonzero(inlier.any(dim=1))[:, 0]
    new_index = torch.full((len(tracks),), -1, dtype=torch.int64)
    new_index[keep] = torch.arange(keep.numel())
    keypoints, point_ids = [], []
    for frame in range(len(names)):
        seen = torch.nonzero(inlier[:, frame])[:, 0]
        keypoints.append(tracks.uv[seen, frame].to(torch.float64))
        point_ids.append(new_index[seen])
    if colors is None:
        rgb = torch.full((keep.numel(), 3), 128, dtype=torch.uint8)
    else:
        rgb = (colors[keep].clamp(0, 1) * 255).round().to(torch.uint8)
    error = torch.zeros(keep.numel(), dtype=torch.float64) if errors is None else errors[keep]
    return ColmapModel(
        K=K.to(torch.float64),
        width=width,
        height=height,
        names=list(names),
        w2c=w2c.to(torch.float64),
        points=points[keep].to(torch.float64),
        colors=rgb,
        errors=error.to(torch.float64),
        keypoints=keypoints,
        point_ids=point_ids,
    )


def write_model(path: str | Path, model: ColmapModel) -> None:
    """Write ``cameras.txt``, ``images.txt`` and ``points3D.txt`` in the folder ``path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    K = model.K
    camera = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {model.width} {model.height} "
        f"{float(K[0, 0])!r} {float(K[1, 1])!r} {float(K[0, 2])!r} {float(K[1, 2])!r}\n"
    )
    (path / "cameras.txt").write_text(camera, encoding="utf-8")

    n_images = len(model.names)
    has_observations = len(model.keypoints) == n_images
    quats = rotmat_to_quat(model.w2c[:, :3, :3]).tolist()
    translations = model.w2c[:, :3, 3].tolist()
    n_observations = sum(len(k) for k in model.keypoints) if has_observations else 0
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {n_images}, mean observations per image: "
        f"{n_observations / max(n_images, 1):.1f}",
    ]
    # COLMAP identifiers start at 1.
    observers: list[list[str]] = [[] for _ in range(model.points.shape[0])]
    for index, name in enumerate(model.names):
        pose = " ".join(repr(v) for v in (*quats[index], *translations[index]))
        lines.append(f"{index + 1} {pose} 1 {name}")
        if not has_observations:
            lines.append("")
            continue
        uv, ids = model.keypoints[index].tolist(), model.point_ids[index].tolist()
        lines.append(
            " ".join(
                f"{x:.4f} {y:.4f} {i + 1 if i >= 0 else -1}"
                for (x, y), i in zip(uv, ids, strict=True)
            )
        )
        for slot, i in enumerate(ids):
            if i >= 0:
                observers[i].append(f"{index + 1} {slot}")
    (path / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    mean_length = sum(len(o) for o in observers) / max(len(observers), 1)
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(observers)}, mean track length: {mean_length:.2f}",
    ]
    xyz, rgb, errors = model.points.tolist(), model.colors.tolist(), model.errors.tolist()
    for index, track in enumerate(observers):
        position = " ".join(repr(v) for v in xyz[index])
        colour = " ".join(str(v) for v in rgb[index])
        lines.append(
            f"{index + 1} {position} {colour} {errors[index]:.6f} {' '.join(track)}".rstrip()
        )
    (path / "points3D.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _data_lines(file: Path) -> list[str]:
    return [l for l in file.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]


def _read_camera(file: Path) -> tuple[Tensor, int, int]:
    cameras = [l.split() for l in _data_lines(file) if l.strip()]
    if not cameras:
        raise ValueError(f"{file} lists no camera")
    signatures = {tuple(camera[1:]) for camera in cameras}
    if len(signatures) > 1:
        raise ValueError(
            f"{file} lists {len(signatures)} different cameras; a video has a single one "
            "(run COLMAP with --ImageReader.single_camera 1)"
        )
    _, name, width, height, *params = cameras[0]
    if name not in _CAMERA_MODELS:
        raise ValueError(f"unsupported camera model {name!r}; supported: {sorted(_CAMERA_MODELS)}")
    count, (fx, fy, cx, cy) = _CAMERA_MODELS[name]
    values = [float(p) for p in params]
    if len(values) != count:
        raise ValueError(f"camera model {name} takes {count} parameters, got {len(values)}")
    used = {fx, fy, cx, cy}
    distortion = [v for i, v in enumerate(values) if i not in used]
    if any(abs(v) > 1e-6 for v in distortion):
        logger.warning(
            "lens distortion %s of the COLMAP camera is ignored: undistort the images first "
            "(colmap image_undistorter)",
            distortion,
        )
    K = torch.tensor(
        [[values[fx], 0.0, values[cx]], [0.0, values[fy], values[cy]], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    return K, int(width), int(height)


def read_model(path: str | Path) -> ColmapModel:
    """Read a text model from the folder ``path`` (images are returned sorted by name)."""
    path = Path(path)
    K, width, height = _read_camera(path / "cameras.txt")

    point_index: dict[int, int] = {}
    xyz, rgb, errors = [], [], []
    points_file = path / "points3D.txt"
    if points_file.exists():
        for line in _data_lines(points_file):
            tokens = line.split()
            if len(tokens) < 8:
                continue
            point_index[int(tokens[0])] = len(xyz)
            xyz.append([float(v) for v in tokens[1:4]])
            rgb.append([int(v) for v in tokens[4:7]])
            errors.append(float(tokens[7]))

    records = []
    lines = _data_lines(path / "images.txt")
    position = 0
    while position < len(lines):
        if not lines[position].strip():
            position += 1
            continue
        header = lines[position].split(maxsplit=9)
        observations = lines[position + 1].split() if position + 1 < len(lines) else []
        position += 2
        if len(header) != 10:
            raise ValueError(f"malformed image line in {path / 'images.txt'}: {header}")
        values = np.array(observations, dtype=np.float64).reshape(-1, 3)
        ids = [point_index.get(int(i), -1) for i in values[:, 2]]
        records.append(
            (
                header[9],
                [float(v) for v in header[1:5]],
                [float(v) for v in header[5:8]],
                torch.from_numpy(values[:, :2].copy()),
                torch.tensor(ids, dtype=torch.int64),
            )
        )
    if not records:
        raise ValueError(f"{path / 'images.txt'} lists no image")
    records.sort(key=lambda record: record[0])
    quats = torch.tensor([r[1] for r in records], dtype=torch.float64)
    translations = torch.tensor([r[2] for r in records], dtype=torch.float64)
    return ColmapModel(
        K=K,
        width=width,
        height=height,
        names=[r[0] for r in records],
        w2c=make_se3(quat_to_rotmat(quats), translations),
        points=torch.tensor(xyz, dtype=torch.float64).reshape(-1, 3),
        colors=torch.tensor(rgb, dtype=torch.uint8).reshape(-1, 3),
        errors=torch.tensor(errors, dtype=torch.float64),
        keypoints=[r[3] for r in records],
        point_ids=[r[4] for r in records],
    )


def frame_name(index: int) -> str:
    """File name of frame ``index`` in exported datasets."""
    return f"frame_{index:05d}.png"


def write_frames(folder: str | Path, images: Tensor) -> list[str]:
    """Write ``images (T, H, W, 3)`` in ``[0, 1]`` as PNG files; returns their names."""
    from PIL import Image

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for index, image in enumerate(images):
        pixels = (image.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        Image.fromarray(pixels).save(folder / frame_name(index))
        names.append(frame_name(index))
    return names


def export_dataset(
    out_dir: str | Path,
    images: Tensor,
    K: Tensor,
    w2c: Tensor,
    tracks: Tracks,
    points: Tensor,
    inlier: Tensor,
    errors: Tensor | None = None,
    registered: Tensor | None = None,
) -> Path:
    """Write a video and its sparse reconstruction as a COLMAP dataset.

    The result, ``out_dir/images/*.png`` and ``out_dir/sparse/0/*.txt``, opens in COLMAP's
    GUI and is the input layout of the common 3DGS and NeRF training scripts. Frames that
    were not registered are left out of the model (their image is still written).

    Args:
        tracks, points, inlier: tracks, their 3D points ``(N, 3)`` and the observations
            ``(N, T)`` consistent with them (as returned by the structure-from-motion).
        errors: ``(N,)`` mean reprojection error of the points.
        registered: ``(T,)`` frames with a pose (default: all).
    """
    out_dir = Path(out_dir)
    n_frames, height, width, _ = images.shape
    names = write_frames(out_dir / "images", images)
    frames = torch.arange(n_frames) if registered is None else torch.nonzero(registered)[:, 0]
    inlier = inlier[:, frames]
    sub_tracks = Tracks(tracks.uv[:, frames], tracks.visible[:, frames])

    # Colour of a point: the image at its first inlier observation.
    first = inlier.to(torch.int64).argmax(dim=1)
    uv = sub_tracks.uv[torch.arange(len(tracks)), first]
    col = uv[:, 0].floor().clamp(0, width - 1).to(torch.int64)
    row = uv[:, 1].floor().clamp(0, height - 1).to(torch.int64)
    colors = images[frames[first], row, col]

    model = model_from_tracks(
        K,
        width,
        height,
        w2c[frames],
        [names[f] for f in frames.tolist()],
        sub_tracks,
        points,
        inlier,
        colors,
        errors,
    )
    write_model(out_dir / "sparse" / "0", model)
    logger.info(
        "COLMAP dataset written to %s (%d images, %d points)",
        out_dir,
        len(model.names),
        model.points.shape[0],
    )
    return out_dir
