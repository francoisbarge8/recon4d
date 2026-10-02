"""Configuration round-trips and overrides."""

import pytest

from recon4d.config import apply_overrides, from_dict, load_yaml, save_yaml, to_dict
from recon4d.gaussians.densify import DensifyConfig
from recon4d.gaussians.trainer import TrainConfig
from recon4d.pipeline import PipelineConfig


def test_round_trip_through_dictionaries():
    cfg = PipelineConfig()
    data = to_dict(cfg)
    assert data["train"]["densify"]["max_gaussians"] == cfg.train.densify.max_gaussians
    assert isinstance(data["motion"]["steps"], list), "tuples become lists (YAML friendly)"
    rebuilt = from_dict(PipelineConfig, data)
    assert rebuilt == cfg
    assert isinstance(rebuilt.train, TrainConfig) and isinstance(
        rebuilt.train.densify, DensifyConfig
    )
    assert isinstance(rebuilt.motion.steps, tuple)


def test_yaml_round_trip(tmp_path):
    cfg = apply_overrides(PipelineConfig(), ["train.crop=[96,72]", "depth=oracle"])
    save_yaml(tmp_path / "nested" / "config.yaml", cfg)
    assert load_yaml(tmp_path / "nested" / "config.yaml", PipelineConfig) == cfg
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    assert load_yaml(tmp_path / "empty.yaml", PipelineConfig) == PipelineConfig()


def test_overrides():
    cfg = apply_overrides(
        PipelineConfig(),
        [
            "train.iterations=123",
            "dynamic=false",
            "train.crop=[96,72]",
            "train.densify.max_gaussians=5000",
            "align.grid=null",
            "train.lr_means=1",
            "depth=oracle",
        ],
    )
    assert cfg.train.iterations == 123 and cfg.dynamic is False
    assert cfg.train.crop == (96, 72)
    assert cfg.train.densify.max_gaussians == 5000
    assert cfg.align.grid is None
    assert cfg.train.lr_means == 1.0 and isinstance(cfg.train.lr_means, float)
    assert cfg.depth == "oracle"
    # The original is untouched.
    assert PipelineConfig().train.iterations != 123


def test_typos_are_rejected():
    with pytest.raises(KeyError, match="unknown configuration option"):
        apply_overrides(PipelineConfig(), ["train.iteration=10"])
    with pytest.raises(KeyError, match="unknown configuration section"):
        apply_overrides(PipelineConfig(), ["trainer.iterations=10"])
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(PipelineConfig(), ["train.iterations"])
    with pytest.raises(KeyError, match="unknown option"):
        from_dict(PipelineConfig, {"depht": "oracle"})
