"""End-to-end pipeline, evaluation, benchmark and command-line interface on tiny sequences."""

import math

import pytest
import torch
from typer.testing import CliRunner

from conftest import tiny_config
from recon4d.benchmark import PROFILES, VARIANTS, collect_results, run_one, summarize
from recon4d.cli import app
from recon4d.config import apply_overrides
from recon4d.data.synthetic import build_synthetic_sequence
from recon4d.evaluation import evaluate_frontend, flatten, world_alignment
from recon4d.geometry import invert_se3
from recon4d.geometry.camera import project, transform_points
from recon4d.io.colmap import read_model
from recon4d.metrics.pose import align_frames, align_trajectory
from recon4d.pipeline import PipelineConfig, run_frontend, training_data
from recon4d.report import export_colmap

SMOKE = PROFILES["smoke"]


def smoke_config(*overrides: str) -> PipelineConfig:
    return apply_overrides(PipelineConfig(), [*SMOKE.overrides, *overrides])


@pytest.fixture(scope="module")
def oracle_frontend(tiny_rolling):
    cfg = smoke_config("depth=oracle", "pose_tracker=oracle", "tracker=oracle", "flow=oracle")
    return run_frontend(tiny_rolling, cfg)


def test_front_end_with_exact_inputs_recovers_the_ground_truth(tiny_rolling, oracle_frontend):
    seq, front = tiny_rolling, oracle_frontend
    metrics = evaluate_frontend(seq, front)
    # Not exactly zero: on such a short clip a few points of the slow ball pass for static.
    assert metrics["pose"]["ate"] < 2e-3 and metrics["pose"]["rpe_rot_deg"] < 0.05
    assert metrics["pose"]["registered"] == 1.0
    assert metrics["depth_aligned"]["abs_rel"] < 0.01
    assert metrics["tracking"]["delta_avg"] > 0.99
    assert metrics["motion_mask"]["iou"] > 0.6
    assert metrics["motion_mask"]["track_recall"] > 0.6
    # Depth consistent over time once aligned (exact depth: at the level of the GT).
    assert metrics["temporal"]["depth_aligned"] < 5 * metrics["temporal"]["depth_gt"] + 1e-3
    # The reconstruction frame differs from the ground-truth one by a similarity only.
    sim = world_alignment(seq, front)
    aligned = sim.apply_to_c2w(invert_se3(front.w2c))
    assert (aligned[:, :3, 3] - seq.gt.c2w[:, :3, 3]).norm(dim=1).max() < 5e-3
    assert front.n_pose_tracks <= len(front.tracks)
    assert front.tracks.dynamic.any() and not front.tracks.dynamic.all()


def test_training_data_drops_unregistered_frames(tiny_rolling, oracle_frontend):
    data = training_data(tiny_rolling, oracle_frontend)
    assert data.train_frames == tiny_rolling.train_indices
    oracle_frontend.registered[1] = False
    try:
        assert 1 not in training_data(tiny_rolling, oracle_frontend).train_frames
    finally:
        oracle_frontend.registered[1] = True


def test_front_end_exports_a_consistent_colmap_dataset(tmp_path, tiny_rolling, oracle_frontend):
    seq, front = tiny_rolling, oracle_frontend
    out = export_colmap(tmp_path, seq, front)
    assert len(list((out / "images").glob("*.png"))) == seq.num_frames
    model = read_model(out / "sparse" / "0")
    assert (model.width, model.height) == (seq.width, seq.height)
    assert torch.allclose(model.K, front.K.to(torch.float64))
    assert torch.allclose(model.w2c, front.w2c.to(torch.float64))
    # Only the tracks reconstructed as static points are in the model.
    assert model.points.shape[0] == int(front.sfm.inlier.any(dim=1).sum())
    assert 0 < model.points.shape[0] < front.n_pose_tracks
    inlier_threshold = 2.0 * smoke_config().sfm.reproj_threshold
    for frame in range(seq.num_frames):
        seen = model.points[model.point_ids[frame]]
        uv, depth = project(model.K, transform_points(model.w2c[frame], seen))
        error = (uv - model.keypoints[frame]).norm(dim=-1)
        assert (depth > 0).all()
        assert error.median() < 0.1 and error.max() < inlier_threshold


def test_pose_alignment_with_orientations_matches_the_position_only_one():
    seq = build_synthetic_sequence(tiny_config("still", val_every=0))
    gt = seq.gt.c2w
    scaled = gt.clone()
    scaled[:, :3, 3] *= 0.4
    a, b = align_trajectory(scaled, gt), align_frames(scaled, gt)
    assert float(a.scale) == pytest.approx(2.5, rel=1e-4)
    assert float(b.scale) == pytest.approx(2.5, rel=1e-4)
    assert torch.allclose(b.R, torch.eye(3), atol=1e-5)


@pytest.mark.parametrize("variant", ["full", "static-only", "oracle-all"])
def test_benchmark_variant_runs_end_to_end(tmp_path, variant):
    metrics = run_one("rolling", variant, "smoke", tmp_path / variant, cache_dir=tmp_path / "cache")
    expected = {
        "pose",
        "depth_raw",
        "depth_aligned",
        "tracking",
        "nvs_train",
        "nvs_test",
        "nvs_val",
    }
    expected |= {
        "geometry_fused",
        "geometry_rendered",
        "tracking_3d",
        "temporal",
        "model",
        "timings",
    }
    assert expected <= set(metrics)
    values = flatten({k: v for k, v in metrics.items() if k != "timings"})
    assert all(math.isfinite(v) for v in values.values()), [
        k for k, v in values.items() if not math.isfinite(v)
    ]
    assert metrics["nvs_test"]["psnr"] > 12.0
    assert metrics["pose"]["registered"] == 1.0
    assert (tmp_path / variant / "config.yaml").exists()
    assert (tmp_path / variant / "metrics.json").exists()
    if variant == "static-only":
        assert metrics["model"]["dynamic_gaussians"] == 0
        assert "motion_mask" not in metrics or metrics["motion_mask"]["iou"] == 0
    else:
        assert metrics["model"]["dynamic_gaussians"] > 0
        assert (tmp_path / variant / "bullet_time.gif").exists()
    for name in (
        "reconstruction.gif",
        "frontend.gif",
        "wiggle.gif",
        "val0.gif",
        "points.ply",
        "scene.pt",
    ):
        assert (tmp_path / variant / name).stat().st_size > 0
    # A finished run is loaded back instead of being recomputed.
    assert run_one("rolling", variant, "smoke", tmp_path / variant) == metrics


def test_results_are_collected_and_summarised(tmp_path):
    fake = {
        "nvs_test": {"psnr": 20.0, "ssim": 0.5},
        "pose": {"ate": 0.01, "rpe_trans": float("nan")},
    }
    better = {"nvs_test": {"psnr": 30.0, "ssim": 0.9}, "pose": {"ate": 0.03}}
    from recon4d.utils import save_json

    save_json(tmp_path / "rolling" / "full" / "metrics.json", fake)
    save_json(tmp_path / "still" / "full" / "metrics.json", better)
    save_json(tmp_path / "still" / "oracle-all" / "metrics.json", better)
    results = collect_results(tmp_path, "smoke")
    assert set(results) == {"rolling", "still"} and set(results["still"]) == {"full", "oracle-all"}
    report = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert report == summarize(results, "smoke")
    assert "| rolling | 1.00 |" in report, "ATE is reported in centimetres"
    # Variants are averaged over the scenes they were run on.
    assert "| full | 25.00 | 0.700 |" in report and "| oracle-all | 30.00 | 0.900 |" in report
    assert "RPE-t" not in report, "columns without any value are dropped"
    assert collect_results(tmp_path / "nothing") == {}


def test_a_failing_run_does_not_stop_the_benchmark(tmp_path, monkeypatch):
    from recon4d import benchmark
    from recon4d.utils import save_json

    def fake_run(scene, variant, profile, out_dir, *args, **kwargs):
        if variant == "static-only":
            raise ValueError("boom")
        save_json(out_dir / "metrics.json", {"nvs_test": {"psnr": 20.0}})

    monkeypatch.setattr(benchmark, "run_one", fake_run)
    with pytest.raises(RuntimeError, match=r"1 run\(s\) failed: still/static-only"):
        benchmark.run_benchmark(
            tmp_path, "smoke", scenes=["still"], variants=["full", "static-only", "oracle-all"]
        )
    assert "ValueError: boom" in (tmp_path / "still" / "static-only" / "error.txt").read_text()
    # The runs after the failure were made and the tables hold what succeeded.
    assert set(collect_results(tmp_path)["still"]) == {"full", "oracle-all"}
    assert (tmp_path / "results.md").exists()
    with pytest.raises(ValueError, match="unknown variant"):
        benchmark.run_benchmark(tmp_path, "smoke", variants=["nope"])


def test_every_variant_is_a_valid_configuration():
    for name, (description, overrides) in VARIANTS.items():
        assert description
        for profile in PROFILES.values():
            cfg = apply_overrides(PipelineConfig(), [*profile.overrides, *overrides])
            assert isinstance(cfg, PipelineConfig), name


def test_command_line_interface(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["config", str(tmp_path / "default.yaml")])
    assert result.exit_code == 0 and (tmp_path / "default.yaml").exists()
    result = runner.invoke(
        app,
        [
            "synth",
            "still",
            "--out",
            str(tmp_path / "synth"),
            "--frames",
            "4",
            "--width",
            "48",
            "--height",
            "36",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "synth" / "still.gif").exists() and (
        tmp_path / "synth" / "still_val0.gif"
    ).exists()
    result = runner.invoke(app, ["collect", str(tmp_path / "empty")])
    assert result.exit_code == 0 and "0 runs" in result.output
    assert runner.invoke(app, ["run", "nope"]).exit_code != 0
