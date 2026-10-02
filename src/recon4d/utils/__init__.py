"""Small shared helpers: seeding, logging, timing and JSON serialisation."""

from __future__ import annotations

import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s %(levelname).1s %(name)s | %(message)s"


def get_logger(name: str = "recon4d") -> logging.Logger:
    """Package logger; configures a console handler on first use."""
    root = logging.getLogger("recon4d")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return logging.getLogger(name)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - global seeding is the point of this function
    torch.manual_seed(seed)


@contextmanager
def timed(label: str, sink: dict[str, float] | None = None, logger: logging.Logger | None = None):
    """Time a block; store the duration (seconds) in ``sink[label]`` and/or log it."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if sink is not None:
            sink[label] = sink.get(label, 0.0) + elapsed
        if logger is not None:
            logger.info("%s: %.2fs", label, elapsed)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert tensors / arrays / paths into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["get_logger", "load_json", "save_json", "seed_everything", "timed", "to_jsonable"]
