"""Training of the in-domain depth network on random synthetic scenes."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from recon4d.data.synthetic.primitives import DTYPE
from recon4d.data.synthetic.render import render_frame
from recon4d.data.synthetic.scenes import random_scene
from recon4d.frontend.depth.tiny_net import (
    TinyDepthConfig,
    TinyDepthNet,
    gradient_matching_loss,
    save_checkpoint,
    scale_invariant_loss,
)
from recon4d.geometry.camera import Intrinsics
from recon4d.metrics.depth import align_depth, depth_metrics
from recon4d.utils import get_logger

logger = get_logger(__name__)

TRAIN_SEED_OFFSET = 100_000
"""Scene seeds used for training start here; the benchmark scenes use small seeds and a
different sampling path, so the two sets never overlap."""


@dataclass(frozen=True)
class DepthTrainConfig:
    """Training set and optimisation settings of the depth network.

    Attributes:
        n_scenes: number of random training scenes.
        frames_per_scene: frames rendered per scene, at random times of its trajectory.
        n_val_scenes: scenes held out for validation.
        width, height: size of the training images.
        fov_x_deg: field of view of the rendered images.
        spp: supersampling of the renderer.
        epochs, batch_size, lr, weight_decay: optimisation hyper-parameters (AdamW with a
            cosine schedule).
        gradient_weight: weight of the gradient-matching loss.
        seed: seed of the data and of the initialisation.
    """

    n_scenes: int = 400
    frames_per_scene: int = 6
    n_val_scenes: int = 40
    width: int = 192
    height: int = 144
    fov_x_deg: float = 60.0
    spp: int = 2
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-4
    gradient_weight: float = 0.5
    seed: int = 0


def render_scene_samples(
    scene_seed: int, frames: int, width: int, height: int, fov_x_deg: float, spp: int
) -> tuple[np.ndarray, np.ndarray]:
    """Render ``frames`` views of one random scene; returns uint8 images and float32 depths."""
    spec = random_scene(scene_seed)
    rng = np.random.RandomState(scene_seed)
    K = Intrinsics.from_fov(width, height, fov_x_deg).matrix(DTYPE)
    n_poses = 48
    c2w = spec.train_c2w(n_poses)
    images, depths = [], []
    for index in rng.choice(n_poses, size=frames, replace=False):
        frame = render_frame(spec.scene, K, c2w[index], index / (n_poses - 1), height, width, spp)
        images.append((frame.rgb.clamp(0, 1) * 255).round().to(torch.uint8).numpy())
        depths.append(frame.depth.numpy())
    return np.stack(images), np.stack(depths)


def _render_job(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    torch.set_num_threads(1)
    return render_scene_samples(*args)


def make_dataset(
    seeds: list[int], cfg: DepthTrainConfig, workers: int = 0, cache: str | Path | None = None
) -> tuple[Tensor, Tensor]:
    """Render (or load from ``cache``) the samples of the given scene seeds.

    Returns ``images (N, H, W, 3)`` uint8 and ``depth (N, H, W)`` float32.
    """
    if cache is not None and Path(cache).exists():
        with np.load(cache) as data:
            return torch.from_numpy(data["images"]), torch.from_numpy(data["depth"])
    jobs = [(s, cfg.frames_per_scene, cfg.width, cfg.height, cfg.fov_x_deg, cfg.spp) for s in seeds]
    start = time.perf_counter()

    def progress(done: int) -> None:
        if done % max(1, len(jobs) // 10) == 0 or done == len(jobs):
            elapsed = time.perf_counter() - start
            logger.info(
                "rendering the depth dataset: %d/%d scenes, %.0fs elapsed, about %.0fs left",
                done,
                len(jobs),
                elapsed,
                elapsed / done * (len(jobs) - done),
            )

    results = []
    if workers > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap(_render_job, jobs, chunksize=2):
                results.append(result)
                progress(len(results))
    else:
        for job in jobs:
            results.append(render_scene_samples(*job))
            progress(len(results))
    images = np.concatenate([r[0] for r in results])
    depth = np.concatenate([r[1] for r in results])
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, images=images, depth=depth)
    return torch.from_numpy(images), torch.from_numpy(depth)


def augment(images: Tensor, depth: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    """Random horizontal flip and photometric jitter of a batch ``(B, 3, H, W)`` in [0, 1]."""
    batch = images.shape[0]
    device = images.device
    flip = torch.rand(batch, generator=generator) < 0.5
    flip = flip.to(device)
    images = torch.where(flip[:, None, None, None], images.flip(-1), images)
    depth = torch.where(flip[:, None, None], depth.flip(-1), depth)
    gain = (0.8 + 0.4 * torch.rand(batch, 1, 1, 1, generator=generator)).to(device)
    gamma = (0.8 + 0.4 * torch.rand(batch, 1, 1, 1, generator=generator)).to(device)
    tint = (0.92 + 0.16 * torch.rand(batch, 3, 1, 1, generator=generator)).to(device)
    images = (images.clamp_min(1e-4) ** gamma * gain * tint).clamp(0.0, 1.0)
    noise = 0.01 * torch.randn(images.shape, generator=generator).to(device)
    return (images + noise).clamp(0.0, 1.0), depth


@torch.no_grad()
def evaluate(
    model: TinyDepthNet, images: Tensor, depth: Tensor, device, batch_size: int = 32
) -> dict:
    """Depth metrics after per-image median scaling (the model is scale-invariant)."""
    model.eval()
    predictions = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device).permute(0, 3, 1, 2).float() / 255.0
        predictions.append(torch.exp(model(batch)).cpu())
    pred = torch.cat(predictions)
    aligned = torch.stack(
        [align_depth(p, g, mode="scale") for p, g in zip(pred, depth, strict=True)]
    )
    return depth_metrics(aligned, depth)


def train_depth_network(
    cfg: DepthTrainConfig,
    out: str | Path,
    device: str = "cpu",
    workers: int = 0,
    cache_dir: str | Path | None = None,
    model_cfg: TinyDepthConfig | None = None,
) -> dict:
    """Render the training set, train the network and save the best checkpoint to ``out``.

    Returns the validation metrics of the saved checkpoint.
    """
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_seeds = [TRAIN_SEED_OFFSET + i for i in range(cfg.n_scenes)]
    val_seeds = [TRAIN_SEED_OFFSET + cfg.n_scenes + i for i in range(cfg.n_val_scenes)]
    cache = None if cache_dir is None else Path(cache_dir)
    start = time.perf_counter()
    train_images, train_depth = make_dataset(
        train_seeds, cfg, workers, None if cache is None else cache / "depth_train.npz"
    )
    val_images, val_depth = make_dataset(
        val_seeds, cfg, workers, None if cache is None else cache / "depth_val.npz"
    )
    logger.info(
        "depth dataset: %d training and %d validation images (%.0fs)",
        len(train_images),
        len(val_images),
        time.perf_counter() - start,
    )

    model = TinyDepthNet(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_images) / cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1
    )
    best: dict = {"abs_rel": float("inf")}
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(train_images), generator=generator)
        running = 0.0
        for step in range(steps_per_epoch):
            index = order[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            images = train_images[index].to(device).permute(0, 3, 1, 2).float() / 255.0
            depth = train_depth[index].to(device)
            images, depth = augment(images, depth, generator)
            log_pred = model(images)
            log_target = torch.log(depth)
            loss = scale_invariant_loss(log_pred, log_target)
            loss = loss + cfg.gradient_weight * gradient_matching_loss(log_pred, log_target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        metrics = evaluate(model, val_images, val_depth, device)
        logger.info(
            "epoch %3d | loss %.4f | val abs_rel %.4f delta1 %.3f",
            epoch + 1,
            running / steps_per_epoch,
            metrics["abs_rel"],
            metrics["delta1"],
        )
        if metrics["abs_rel"] < best["abs_rel"]:
            best = {**metrics, "epoch": epoch + 1}
            save_checkpoint(out, model, {"val": best, "train_images": len(train_images)})
    return best
