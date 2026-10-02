"""Train the in-domain depth network on random synthetic scenes.

Example (about 10 minutes on a GPU, the rendering of the training set included)::

    python scripts/train_depth.py --out assets/checkpoints/tiny_depth.pth --workers 4
"""

from __future__ import annotations

import argparse
import os

import torch

from recon4d.frontend.depth.train import DepthTrainConfig, train_depth_network
from recon4d.utils import get_logger, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="assets/checkpoints/tiny_depth.pth")
    parser.add_argument("--scenes", type=int, default=400, help="number of training scenes")
    parser.add_argument("--frames", type=int, default=6, help="frames rendered per scene")
    parser.add_argument("--val-scenes", type=int, default=40)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--cache", default=None, help="directory caching the rendered dataset")
    args = parser.parse_args()

    logger = get_logger("train_depth")
    cfg = DepthTrainConfig(
        n_scenes=args.scenes,
        frames_per_scene=args.frames,
        n_val_scenes=args.val_scenes,
        width=args.width,
        height=args.height,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    best = train_depth_network(cfg, args.out, args.device, args.workers, args.cache)
    save_json(os.path.splitext(args.out)[0] + ".json", {"config": vars(args), "validation": best})
    logger.info(
        "best checkpoint: epoch %d, abs_rel %.4f -> %s", best["epoch"], best["abs_rel"], args.out
    )


if __name__ == "__main__":
    main()
