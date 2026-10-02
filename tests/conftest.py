"""Shared fixtures.

The synthetic sequences used in tests are tiny (low resolution, few frames, no
supersampling) so that the whole suite runs in about a minute on a laptop CPU.
"""

from __future__ import annotations

import pytest
import torch

from recon4d.data.synthetic import SyntheticConfig, build_synthetic_sequence


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


def tiny_config(scene: str = "rolling", **overrides) -> SyntheticConfig:
    params = {
        "scene": scene,
        "n_frames": 8,
        "width": 64,
        "height": 48,
        "spp": 1,
        "test_every": 4,
        "val_every": 4,
        "covis_stride": 1,
    }
    params.update(overrides)
    return SyntheticConfig(**params)


@pytest.fixture(scope="session")
def tiny_rolling():
    """8-frame 64x48 version of the ``rolling`` scene (two moving balls)."""
    return build_synthetic_sequence(tiny_config("rolling"))


@pytest.fixture(scope="session")
def tiny_still():
    """8-frame 64x48 version of the static ``still`` scene."""
    return build_synthetic_sequence(tiny_config("still"))
