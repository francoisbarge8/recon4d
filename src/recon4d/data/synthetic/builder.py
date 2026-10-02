"""Turn a scene specification into a :class:`~recon4d.data.sequence.VideoSequence`."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from recon4d.data.sequence import GroundTruth, ValCamera, VideoSequence
from recon4d.data.synthetic.oracle import SceneOracle
from recon4d.data.synthetic.primitives import DTYPE
from recon4d.data.synthetic.render import render_frame
from recon4d.data.synthetic.scenes import SceneSpec, build_scene
from recon4d.geometry.camera import Intrinsics

GENERATOR_VERSION = 1
"""Bump whenever the scenes or the renderer change, to invalidate cached sequences."""

CACHE_ENV = "RECON4D_CACHE"


@dataclass(frozen=True)
class SyntheticConfig:
    """Rendering parameters of a synthetic sequence.

    Attributes:
        scene: name of the scene (see :data:`recon4d.data.synthetic.SCENE_NAMES`).
        n_frames: length of the training video.
        width, height: image size in pixels.
        fov_x_deg: horizontal field of view.
        spp: the image is supersampled on an ``spp x spp`` grid per pixel.
        seed: seed of the scene layout, textures and camera shake.
        test_every: every ``test_every``-th training frame is held out.
        val_every: validation cameras are rendered every ``val_every`` frames (0 disables
            them).
        covis_stride: stride of the training frames used for the co-visibility test.
    """

    scene: str = "rolling"
    n_frames: int = 48
    width: int = 192
    height: int = 144
    fov_x_deg: float = 60.0
    spp: int = 3
    seed: int = 0
    test_every: int = 8
    val_every: int = 4
    covis_stride: int = 3

    def cache_key(self) -> str:
        payload = repr(sorted(asdict(self).items())) + f"|v{GENERATOR_VERSION}"
        return f"{self.scene}_{hashlib.sha1(payload.encode()).hexdigest()[:12]}"


def _render_buffers(spec: SceneSpec, cfg: SyntheticConfig) -> dict[str, np.ndarray]:
    """Ray trace every buffer of the sequence; returns plain arrays (cacheable).

    Images are rounded to 8 bits, like any real video frame, which also makes the disk
    cache lossless.
    """
    K = Intrinsics.from_fov(cfg.width, cfg.height, cfg.fov_x_deg).matrix(DTYPE)
    c2w = spec.train_c2w(cfg.n_frames)
    times = torch.linspace(0.0, 1.0, cfg.n_frames, dtype=DTYPE)
    scene = spec.scene

    frames = [
        render_frame(scene, K, c2w[i], float(times[i]), cfg.height, cfg.width, cfg.spp)
        for i in range(cfg.n_frames)
    ]
    depth = torch.stack([f.depth for f in frames])
    obj = torch.stack([f.obj for f in frames])
    buffers = {
        "images": torch.stack([f.rgb for f in frames]),
        "depth": depth,
        "obj": obj,
    }

    if cfg.val_every > 0 and spec.val_cameras:
        oracle = SceneOracle(scene, K, c2w, times, cfg.height, cfg.width)
        val_frames = list(range(0, cfg.n_frames, cfg.val_every))
        held_out = {i for i in range(cfg.n_frames) if _is_test_frame(i, cfg.test_every)}
        # Co-visibility is defined w.r.t. the frames the scene model is trained on.
        covis_frames = [i for i in range(0, cfg.n_frames, cfg.covis_stride) if i not in held_out]
        for index, pose in enumerate(spec.val_c2w()):
            rendered = [
                render_frame(scene, K, pose, float(times[i]), cfg.height, cfg.width, cfg.spp)
                for i in val_frames
            ]
            buffers[f"val{index}_images"] = torch.stack([r.rgb for r in rendered])
            buffers[f"val{index}_depth"] = torch.stack([r.depth for r in rendered])
            buffers[f"val{index}_obj"] = torch.stack([r.obj for r in rendered])
            buffers[f"val{index}_covisible"] = torch.stack(
                [
                    oracle.covisible(
                        r.obj.reshape(-1), r.local.reshape(-1, 3), depth, obj, covis_frames
                    ).reshape(cfg.height, cfg.width)
                    for r in rendered
                ]
            )

    out: dict[str, np.ndarray] = {}
    for key, value in buffers.items():
        if key.endswith("images"):
            out[key] = torch.round(value.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
        elif key.endswith("obj"):
            out[key] = value.to(torch.uint8).numpy()
        else:
            out[key] = value.numpy()
    return out


def _is_test_frame(index: int, test_every: int) -> bool:
    return test_every > 0 and index % test_every == test_every // 2


def render_sequence(
    spec: SceneSpec, cfg: SyntheticConfig, cache_dir: str | Path | None = None
) -> VideoSequence:
    """Render the training video, validation cameras and ground truth of ``spec``.

    With ``cache_dir`` set, the ray-traced buffers are stored as a compressed ``.npz`` and
    reused on the next call with the same configuration. The analytic oracle is always
    rebuilt from the specification, so nothing about the ground truth depends on the cache.
    """
    buffers: dict[str, np.ndarray] | None = None
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{cfg.cache_key()}.npz"
        if cache_file.exists():
            with np.load(cache_file) as data:
                buffers = {key: data[key] for key in data.files}
    if buffers is None:
        buffers = _render_buffers(spec, cfg)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, **buffers)

    intrinsics = Intrinsics.from_fov(cfg.width, cfg.height, cfg.fov_x_deg)
    K = intrinsics.matrix(DTYPE)
    c2w = spec.train_c2w(cfg.n_frames)
    times = torch.linspace(0.0, 1.0, cfg.n_frames, dtype=DTYPE)
    scene = spec.scene

    def images(key: str) -> Tensor:
        return torch.from_numpy(buffers[key]).to(torch.float32) / 255.0

    obj = torch.from_numpy(buffers["obj"]).to(torch.int64)
    gt = GroundTruth(
        c2w=c2w.to(torch.float32),
        depth=torch.from_numpy(buffers["depth"]),
        obj=obj,
        dynamic_mask=scene.is_dynamic(obj),
        oracle=SceneOracle(scene, K, c2w, times, cfg.height, cfg.width),
    )
    if "val0_images" in buffers:
        val_frames = list(range(0, cfg.n_frames, cfg.val_every))
        for index, pose in enumerate(spec.val_c2w()):
            val_obj = torch.from_numpy(buffers[f"val{index}_obj"]).to(torch.int64)
            gt.val_cameras.append(
                ValCamera(
                    name=f"val{index}",
                    c2w=pose.to(torch.float32),
                    frames=val_frames,
                    images=images(f"val{index}_images"),
                    depth=torch.from_numpy(buffers[f"val{index}_depth"]),
                    dynamic_mask=scene.is_dynamic(val_obj),
                    covisible=torch.from_numpy(buffers[f"val{index}_covisible"]),
                )
            )
    return VideoSequence(
        name=spec.name,
        images=images("images"),
        timestamps=times.to(torch.float32),
        intrinsics=intrinsics,
        test_every=cfg.test_every,
        gt=gt,
    )


def build_synthetic_sequence(
    cfg: SyntheticConfig, cache_dir: str | Path | None = None
) -> VideoSequence:
    """Build a named benchmark sequence from its configuration.

    ``cache_dir`` defaults to the ``RECON4D_CACHE`` environment variable (no caching when
    it is unset).
    """
    if cache_dir is None:
        cache_dir = os.environ.get(CACHE_ENV) or None
    return render_sequence(build_scene(cfg.scene, cfg.seed), cfg, cache_dir)
