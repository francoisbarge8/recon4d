"""Optimisation of a (dynamic) Gaussian scene against a monocular video."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from recon4d.gaussians.densify import (
    DensificationStats,
    DensifyConfig,
    densify_and_prune,
    reset_opacity,
)
from recon4d.gaussians.model import CloudOptimizer
from recon4d.gaussians.rasterizer import RasterSettings
from recon4d.gaussians.render import Camera
from recon4d.gaussians.scene import MOTION_LOGITS, GaussianScene, SceneRender
from recon4d.geometry.camera import project, sample_bilinear, sample_depth, transform_points
from recon4d.metrics.image import ssim
from recon4d.types import TrainingData
from recon4d.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters of the scene optimisation.

    Learning rates follow 3DGS; ``lr_means`` and the translation part of ``lr_bases`` are
    multiplied by the scene extent so that they are expressed relative to the scene size.

    Loss weights:
        lambda_dssim: weight of ``1 - SSIM`` against L1 in the photometric loss.
        lambda_depth: log-depth L1 between the rendered depth and the depth prior.
        lambda_mask: L1 between the rendered dynamic share and the motion mask.
        lambda_track: reprojection error (in units of the image size) of the rendered 3D
            correspondences against the 2D point tracks.
        lambda_track_depth: log-depth L1 between tracked points transported to another
            frame and the depth prior there.
        lambda_rigid: as-rigid-as-possible penalty on the distances between neighbouring
            dynamic Gaussians (relative to the scene extent).
        lambda_smooth: squared acceleration of the motion bases.

    Attributes:
        crop: ``(width, height)`` of the random window rendered at each step, or ``None``
            to render full frames. Rendering cost is proportional to the number of pixels,
            so on CPU it pays to take many cheap steps on crops rather than a few
            expensive ones on whole images (the stochastic-gradient trade-off of NeRF's
            ray batches, applied to splatting).
    """

    iterations: int = 2000
    crop: tuple[int, int] | None = None
    lr_means: float = 1.6e-4
    lr_means_final: float = 1.6e-6
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacity: float = 5e-2
    lr_sh_dc: float = 2.5e-3
    lr_sh_rest: float = 1.25e-4
    lr_motion_logits: float = 1e-2
    lr_bases: float = 5e-4
    lambda_dssim: float = 0.2
    lambda_depth: float = 0.1
    lambda_mask: float = 0.5
    lambda_track: float = 2.0
    lambda_track_depth: float = 0.1
    lambda_rigid: float = 1.0
    lambda_smooth: float = 0.1
    track_batch: int = 1024
    rigid_neighbors: int = 8
    sh_interval: int = 500
    max_dynamic_gaussians: int = 4000
    densify: DensifyConfig = field(default_factory=DensifyConfig)
    raster: RasterSettings = field(default_factory=RasterSettings)
    log_every: int = 100
    seed: int = 0


@dataclass(frozen=True)
class Window:
    """The image rectangle rendered in one optimisation step."""

    x0: int
    y0: int
    width: int
    height: int

    def crop(self, image: Tensor) -> Tensor:
        """Cut the window out of a per-pixel map ``(H, W, ...)``."""
        return image[self.y0 : self.y0 + self.height, self.x0 : self.x0 + self.width]

    def intrinsics(self, K: Tensor) -> Tensor:
        """Intrinsics of the camera whose image is this window."""
        K = K.clone()
        K[0, 2] -= self.x0
        K[1, 2] -= self.y0
        return K

    def contains(self, uv: Tensor, margin: float = 0.5) -> Tensor:
        return (
            (uv[:, 0] >= self.x0 + margin)
            & (uv[:, 0] <= self.x0 + self.width - margin)
            & (uv[:, 1] >= self.y0 + margin)
            & (uv[:, 1] <= self.y0 + self.height - margin)
        )

    def to_local(self, uv: Tensor) -> Tensor:
        return uv - uv.new_tensor([self.x0, self.y0])


class SceneTrainer:
    """Fits a :class:`GaussianScene` to a :class:`TrainingData`.

    Args:
        extent: characteristic size of the scene (for instance the median depth), used to
            make learning rates, densification thresholds and regularisers scale-free.
    """

    def __init__(
        self, scene: GaussianScene, data: TrainingData, cfg: TrainConfig, extent: float
    ) -> None:
        self.scene = scene
        self.data = data
        self.cfg = cfg
        self.extent = float(extent)
        self.iteration = 0
        self.history: list[dict[str, float]] = []
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self._neighbors: Tensor | None = None
        self._start = time.perf_counter()

        self.optimizers: dict[str, CloudOptimizer] = {}
        self.stats: dict[str, DensificationStats] = {}
        for name, cloud in scene.clouds().items():
            lrs = {
                "means": cfg.lr_means * self.extent,
                "log_scales": cfg.lr_scales,
                "quats": cfg.lr_quats,
                "opacity_logits": cfg.lr_opacity,
                "sh_dc": cfg.lr_sh_dc,
                "sh_rest": cfg.lr_sh_rest,
            }
            if MOTION_LOGITS in cloud.params:
                lrs[MOTION_LOGITS] = cfg.lr_motion_logits
            self.optimizers[name] = CloudOptimizer(cloud, lrs)
            self.stats[name] = DensificationStats(len(cloud), cloud.means.device)
        self.motion_optimizer = None
        if scene.motion is not None:
            self.motion_optimizer = torch.optim.Adam(
                [
                    {"params": [scene.motion.rot6d], "lr": cfg.lr_bases},
                    {"params": [scene.motion.trans], "lr": cfg.lr_bases * self.extent},
                ]
            )

    # ------------------------------------------------------------------------ losses

    def _randint(self, high: int) -> int:
        return int(torch.randint(high, (1,), generator=self.generator))

    def _window(self) -> Window:
        data, crop = self.data, self.cfg.crop
        if crop is None:
            return Window(0, 0, data.width, data.height)
        width, height = min(crop[0], data.width), min(crop[1], data.height)
        return Window(
            self._randint(data.width - width + 1),
            self._randint(data.height - height + 1),
            width,
            height,
        )

    def _track_loss(
        self, out: SceneRender, frame: int, target: int, window: Window
    ) -> dict[str, Tensor]:
        """Reprojection error of the rendered 3D correspondences against the 2D tracks.

        ``out.positions[0]`` holds, for every pixel of ``frame``, where the surface seen
        there is at time ``target``. Sampled at a track's location in ``frame`` and
        projected into the camera of ``target``, it should land on the track's location in
        ``target`` and at the depth predicted there.
        """
        data, cfg = self.data, self.cfg
        tracks = data.tracks
        usable = tracks.visible[:, frame] & tracks.visible[:, target]
        index = torch.nonzero(usable & window.contains(tracks.uv[:, frame]))[:, 0]
        if index.numel() == 0:
            return {}
        if index.numel() > cfg.track_batch:
            choice = torch.randperm(index.numel(), generator=self.generator)[: cfg.track_batch]
            index = index[choice.to(index.device)]
        uv_src = window.to_local(tracks.uv[index, frame])
        uv_dst = tracks.uv[index, target]
        maps = torch.cat([out.positions[0], out.alpha[..., None]], dim=-1).permute(2, 0, 1)
        sampled = sample_bilinear(maps, uv_src)
        positions, alpha = sampled[:, :3], sampled[:, 3]
        uv_pred, z = project(data.K, transform_points(data.w2c[target], positions))
        ok = (alpha.detach() > 0.5) & (z.detach() > 1e-4)
        if not ok.any():
            return {}
        weight = ok.to(z.dtype) / ok.sum()
        error = (uv_pred - uv_dst).abs().sum(dim=-1) / max(data.width, data.height)
        terms = {"track": (error * weight).sum()}
        if data.depth is not None and cfg.lambda_track_depth > 0:
            depth_valid = None if data.depth_valid is None else data.depth_valid[target]
            prior, sampled_ok = sample_depth(data.depth[target], uv_dst, valid=depth_valid)
            valid = ok & sampled_ok
            if valid.any():
                log_error = (torch.log(z.clamp_min(1e-4)) - torch.log(prior.clamp_min(1e-4))).abs()
                terms["track_depth"] = (log_error * valid).sum() / valid.sum()
        return terms

    def _rigidity_loss(self, frame_a: int, frame_b: int) -> Tensor:
        """As-rigid-as-possible: neighbouring dynamic Gaussians keep their distances."""
        dynamic = self.scene.dynamic
        n = len(dynamic)
        k = min(self.cfg.rigid_neighbors, n - 1)
        if k < 1:
            return dynamic.means.new_zeros(())
        if self._neighbors is None or self._neighbors.shape[0] != n:
            canonical = dynamic.means.detach().cpu().numpy().astype(np.float64)
            _, neighbors = cKDTree(canonical).query(canonical, k=k + 1)
            self._neighbors = torch.as_tensor(
                neighbors[:, 1:], dtype=torch.int64, device=dynamic.means.device
            )
        xa = self.scene.dynamic_means(frame_a)
        xb = self.scene.dynamic_means(frame_b)
        da = torch.sqrt(((xa[:, None] - xa[self._neighbors]) ** 2).sum(dim=-1) + 1e-12)
        db = torch.sqrt(((xb[:, None] - xb[self._neighbors]) ** 2).sum(dim=-1) + 1e-12)
        return (da - db).abs().mean() / self.extent

    def _losses(
        self, out: SceneRender, frame: int, target: int | None, window: Window
    ) -> dict[str, Tensor]:
        data, cfg = self.data, self.cfg
        image = window.crop(data.images[frame])
        terms = {
            "l1": (out.color - image).abs().mean(),
            "dssim": 1.0 - ssim(out.color, image),
        }
        if data.depth is not None and cfg.lambda_depth > 0:
            prior = window.crop(data.depth[frame])
            valid = (out.alpha.detach() > 0.5) & (out.depth.detach() > 0) & (prior > 0)
            if data.depth_valid is not None:
                valid = valid & window.crop(data.depth_valid[frame])
            if valid.any():
                terms["depth"] = (
                    (torch.log(out.depth[valid]) - torch.log(prior[valid])).abs().mean()
                )
        if out.dynamic is not None and data.dynamic_mask is not None and cfg.lambda_mask > 0:
            mask = window.crop(data.dynamic_mask[frame]).to(out.dynamic.dtype)
            terms["mask"] = (out.dynamic - mask).abs().mean()
        if target is not None:
            terms.update(self._track_loss(out, frame, target, window))
        if self.scene.is_dynamic:
            other = target if target is not None else self._randint(data.num_frames)
            if cfg.lambda_rigid > 0:
                terms["rigid"] = self._rigidity_loss(frame, other)
            if cfg.lambda_smooth > 0:
                terms["smooth"] = self.scene.motion.smoothness(self.extent)
        return terms

    def _total(self, terms: dict[str, Tensor]) -> Tensor:
        cfg = self.cfg
        weights = {
            "l1": 1.0 - cfg.lambda_dssim,
            "dssim": cfg.lambda_dssim,
            "depth": cfg.lambda_depth,
            "mask": cfg.lambda_mask,
            "track": cfg.lambda_track,
            "track_depth": cfg.lambda_track_depth,
            "rigid": cfg.lambda_rigid,
            "smooth": cfg.lambda_smooth,
        }
        return sum(weights[name] * value for name, value in terms.items())

    # -------------------------------------------------------------------------- step

    def _means_lr(self) -> float:
        cfg = self.cfg
        t = min(self.iteration / max(cfg.iterations, 1), 1.0)
        log_lr = (1.0 - t) * math.log(cfg.lr_means) + t * math.log(cfg.lr_means_final)
        return math.exp(log_lr) * self.extent

    def step(self) -> dict[str, float]:
        """One optimisation step on a random training frame; returns the loss terms."""
        cfg, data, scene = self.cfg, self.data, self.scene
        self.iteration += 1
        frames = data.train_frames
        frame = frames[self._randint(len(frames))]
        target = None
        if data.tracks is not None and cfg.lambda_track > 0 and len(frames) > 1:
            target = frames[self._randint(len(frames))]
            while target == frame:
                target = frames[self._randint(len(frames))]

        window = self._window()
        camera = Camera(window.intrinsics(data.K), data.w2c[frame], window.width, window.height)
        out = scene.render(
            camera, frame, cfg.raster, position_frames=() if target is None else (target,)
        )
        terms = self._losses(out, frame, target, window)
        loss = self._total(terms)
        loss.backward()

        # The loss is a mean over the window: rescale the gradients to what a full-frame
        # mean would give, so that the densification threshold does not depend on the crop.
        grad = out.projection.means2d.grad
        if grad is not None:
            grad = grad * (window.width * window.height) / (data.width * data.height)
        visible = out.projection.visible
        offset = 0
        for name, cloud in scene.clouds().items():
            n = len(cloud)
            if grad is not None and offset + n <= grad.shape[0]:
                self.stats[name].update(
                    grad[offset : offset + n], visible[offset : offset + n], data.width, data.height
                )
            offset += n

        lr = self._means_lr()
        for optimizer in self.optimizers.values():
            optimizer.set_lr("means", lr)
            optimizer.step()
            optimizer.zero_grad()
        if self.motion_optimizer is not None:
            self.motion_optimizer.step()
            self.motion_optimizer.zero_grad(set_to_none=True)

        self._update_structure()
        record = {name: float(value) for name, value in terms.items()}
        record["loss"] = float(loss)
        return record

    def _update_structure(self) -> None:
        cfg = self.cfg
        it = self.iteration
        if cfg.sh_interval > 0 and it % cfg.sh_interval == 0:
            for cloud in self.scene.clouds().values():
                cloud.raise_sh_degree()
        if cfg.densify.is_due(it):
            for name, cloud in self.scene.clouds().items():
                densify_cfg = cfg.densify
                if name == "dynamic":
                    densify_cfg = replace(densify_cfg, max_gaussians=cfg.max_dynamic_gaussians)
                counts = densify_and_prune(
                    cloud,
                    self.optimizers[name],
                    self.stats[name],
                    densify_cfg,
                    self.extent,
                    self.generator,
                )
                logger.debug("iteration %d, %s cloud: %s", it, name, counts)
            self._neighbors = None
        if cfg.densify.reset_is_due(it):
            for name, cloud in self.scene.clouds().items():
                reset_opacity(cloud, self.optimizers[name], cfg.densify.opacity_reset_value)

    def fit(self, iterations: int | None = None) -> list[dict[str, float]]:
        """Run the optimisation; returns the logged history (one entry per ``log_every``)."""
        total = self.cfg.iterations if iterations is None else iterations
        running: dict[str, float] = {}
        count = 0
        while self.iteration < total:
            record = self.step()
            for name, value in record.items():
                running[name] = running.get(name, 0.0) + value
            count += 1
            if self.iteration % self.cfg.log_every == 0 or self.iteration == total:
                entry = {name: value / count for name, value in running.items()}
                entry["iteration"] = self.iteration
                entry["seconds"] = time.perf_counter() - self._start
                entry["n_static"] = len(self.scene.static)
                entry["n_dynamic"] = (
                    len(self.scene.dynamic) if self.scene.dynamic is not None else 0
                )
                self.history.append(entry)
                logger.info(
                    "it %5d | loss %.4f | l1 %.4f | gaussians %d + %d | %.0fs",
                    self.iteration,
                    entry["loss"],
                    entry["l1"],
                    entry["n_static"],
                    entry["n_dynamic"],
                    entry["seconds"],
                )
                running, count = {}, 0
        return self.history


@torch.no_grad()
def render_views(
    scene: GaussianScene,
    K: Tensor,
    w2c: Tensor,
    frames: list[int],
    width: int,
    height: int,
    settings: RasterSettings | None = None,
) -> dict[str, Tensor]:
    """Render the scene from ``w2c (F, 4, 4)`` at the timestamps ``frames`` (no gradients).

    The cameras may live on any device: they are moved to the scene's, and the results are
    returned on the CPU as ``color (F, H, W, 3)``, ``depth (F, H, W)``, ``alpha (F, H, W)``
    and, for dynamic scenes, ``dynamic (F, H, W)``.
    """
    device = scene.static.means.device
    K = K.to(device)
    outputs: dict[str, list[Tensor]] = {"color": [], "depth": [], "alpha": [], "dynamic": []}
    for pose, frame in zip(w2c.to(device), frames, strict=True):
        out = scene.render(Camera(K, pose, width, height), frame, settings)
        outputs["color"].append(out.color.clamp(0.0, 1.0).cpu())
        outputs["depth"].append(out.depth.cpu())
        outputs["alpha"].append(out.alpha.cpu())
        if out.dynamic is not None:
            outputs["dynamic"].append(out.dynamic.cpu())
    return {name: torch.stack(values) for name, values in outputs.items() if values}
