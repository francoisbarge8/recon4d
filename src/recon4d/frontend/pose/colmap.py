"""Camera poses from COLMAP, through its Python bindings (``pip install pycolmap``).

COLMAP is the reference structure-from-motion system: SIFT features, sequential or
exhaustive matching, incremental mapping. It serves here as an alternative pose back-end
(``poses=colmap``) and as the yardstick of the track-based structure-from-motion of this
package: both are scored by the same metrics on the same sequences.

The reconstruction is exchanged through COLMAP's text format (see :mod:`recon4d.io.colmap`)
rather than through the objects of the bindings, whose interface changes between releases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import Tensor

from recon4d.io.colmap import ColmapModel, read_model, write_frames
from recon4d.utils import get_logger

logger = get_logger(__name__)

MATCHERS = ("sequential", "exhaustive")


def _set(options, **values) -> None:
    """Set the options that this version of the bindings knows about."""
    for name, value in values.items():
        if hasattr(options, name):
            setattr(options, name, value)
        else:
            logger.debug("pycolmap: option %s is not available in this version", name)


def run_colmap(
    images: Tensor,
    K: Tensor | None = None,
    work_dir: str | Path | None = None,
    matcher: str = "sequential",
    focal_guess: float = 1.2,
) -> ColmapModel:
    """Reconstruct a video with COLMAP.

    Args:
        images: ``(T, H, W, 3)`` in ``[0, 1]``.
        K: known intrinsics, kept fixed; ``None`` lets COLMAP estimate a pinhole camera
            starting from a focal length of ``focal_guess`` times the larger image side.
        work_dir: where the frames, the database and the model are written (a temporary
            folder by default).
        matcher: ``"sequential"`` (neighbouring frames, the right choice for a video) or
            ``"exhaustive"``.

    Returns:
        The largest reconstructed model. Its images are named after the frames they come
        from (see :func:`frame_indices`); frames COLMAP could not register are absent.
    """
    try:
        import pycolmap
    except ImportError as error:
        raise ImportError("the COLMAP back-end needs pycolmap: pip install pycolmap") from error
    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r}; choose among {MATCHERS}")

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(work_dir) if work_dir is not None else Path(scratch)
        image_dir, database, sparse = root / "images", root / "database.db", root / "sparse"
        sparse.mkdir(parents=True, exist_ok=True)
        if database.exists():
            database.unlink()
        write_frames(image_dir, images)

        height, width = images.shape[1:3]
        if K is None:
            focal = focal_guess * max(width, height)
            params = [focal, focal, width / 2, height / 2]
        else:
            params = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        reader = pycolmap.ImageReaderOptions()
        _set(reader, camera_model="PINHOLE", camera_params=",".join(repr(p) for p in params))
        extract = {"camera_mode": pycolmap.CameraMode.SINGLE, "reader_options": reader}
        # pycolmap 3 overrides the model of the reader options with an argument of its own
        # (SIMPLE_RADIAL by default); pycolmap 4 has no such argument.
        if "camera_model" in (pycolmap.extract_features.__doc__ or ""):
            extract["camera_model"] = "PINHOLE"
        pycolmap.extract_features(database, image_dir, **extract)
        if matcher == "sequential":
            pycolmap.match_sequential(database)
        else:
            pycolmap.match_exhaustive(database)

        options = pycolmap.IncrementalPipelineOptions()
        if K is not None:
            _set(
                options,
                ba_refine_focal_length=False,
                ba_refine_principal_point=False,
                ba_refine_extra_params=False,
            )
        maps = pycolmap.incremental_mapping(database, image_dir, sparse, options=options)
        if not maps:
            raise RuntimeError("COLMAP could not reconstruct the video")
        best = max(maps.values(), key=lambda reconstruction: reconstruction.num_reg_images())
        text = sparse / "text"
        text.mkdir(exist_ok=True)
        best.write_text(str(text))
        model = read_model(text)
    logger.info(
        "COLMAP: %d/%d frames registered, %d points",
        len(model.names),
        images.shape[0],
        model.points.shape[0],
    )
    return model


def frame_indices(names: list[str]) -> list[int]:
    """Frame index of images named by :func:`recon4d.io.colmap.frame_name`."""
    return [int(Path(name).stem.rsplit("_", 1)[-1]) for name in names]


def colmap_poses(
    images: Tensor,
    K: Tensor | None = None,
    matcher: str = "sequential",
    focal_guess: float = 1.2,
    work_dir: str | Path | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Poses of a video estimated by COLMAP.

    Returns:
        ``K (3, 3)``, ``w2c (T, 4, 4)`` (identity for frames that were not registered) and
        ``registered (T,)``, all float64 except the mask.
    """
    model = run_colmap(images, K, work_dir, matcher, focal_guess)
    n_frames = images.shape[0]
    w2c = torch.eye(4, dtype=torch.float64).repeat(n_frames, 1, 1)
    registered = torch.zeros(n_frames, dtype=torch.bool)
    index = torch.tensor(frame_indices(model.names), dtype=torch.int64)
    w2c[index] = model.w2c
    registered[index] = True
    if registered.sum() < 2:
        raise RuntimeError(f"COLMAP registered only {int(registered.sum())} frame(s)")
    return (model.K if K is None else K.to(torch.float64)), w2c, registered
