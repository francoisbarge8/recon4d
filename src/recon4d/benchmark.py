"""The benchmark: every scene x every pipeline variant, with result tables.

A *variant* is the default pipeline with a few options changed. Two families:

* **oracle analysis** - one stage at a time is replaced by its ground truth, which
  attributes the final error to depth, tracking or pose estimation;
* **ablations** - one loss or one mechanism of the scene optimisation is removed.

Runs are resumable: a (scene, variant) pair whose ``metrics.json`` exists is not recomputed.
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from pathlib import Path

from recon4d.config import apply_overrides
from recon4d.data.synthetic import SyntheticConfig, build_synthetic_sequence
from recon4d.evaluation import EvalConfig, Metrics, evaluate_frontend, evaluate_scene
from recon4d.pipeline import PipelineConfig, run_pipeline
from recon4d.report import save_run
from recon4d.utils import get_logger, load_json, save_json

logger = get_logger(__name__)


@dataclass(frozen=True)
class Profile:
    """Size of the benchmark: sequence resolution / length and optimisation budget."""

    synth: dict
    overrides: tuple[str, ...]
    description: str


PROFILES: dict[str, Profile] = {
    "smoke": Profile(
        {"n_frames": 8, "width": 64, "height": 48, "spp": 1, "test_every": 4, "covis_stride": 1},
        (
            "train.iterations=40",
            "train.crop=[48,36]",
            "train.densify.start=15",
            "train.densify.stop=35",
            "train.densify.interval=10",
            "train.densify.max_gaussians=1200",
            "train.max_dynamic_gaussians=300",
            "train.log_every=20",
            "init.n_static=800",
            "init.n_dynamic=200",
            # 8 frames of 64 x 48: few tracks of the small moving objects last 4 frames, and
            # how many do depends on the OpenCV version; 2 frames keeps the dynamic path tested.
            "init.min_track_frames=2",
            "init.keyframes=3",
            "init.stride=1",
            "init.basis_fit_iterations=50",
            "init.num_bases=3",
            "motion.steps=[1,2]",
            "motion.min_area=6",
            "motion.morph_radius=1",
            "flow_chain.keyframe_interval=3",
            "klt.keyframe_interval=3",
        ),
        "seconds per run; checks that everything executes",
    ),
    "cpu": Profile(
        {"n_frames": 24, "width": 128, "height": 96},
        (
            "train.iterations=800",
            "train.crop=[96,72]",
            "train.densify.start=150",
            "train.densify.stop=600",
            "train.densify.max_gaussians=7000",
            "train.max_dynamic_gaussians=1500",
            "init.n_static=4000",
            "init.n_dynamic=800",
            "motion.min_area=15",
        ),
        "a few minutes per run on a laptop CPU",
    ),
    "gpu": Profile(
        {"n_frames": 60, "width": 384, "height": 288},
        (
            "train.iterations=7000",
            "train.densify.start=500",
            "train.densify.stop=5000",
            "train.densify.max_gaussians=120000",
            "train.max_dynamic_gaussians=20000",
            "train.track_batch=4096",
            "init.n_static=40000",
            "init.n_dynamic=6000",
            "init.keyframes=10",
            "klt.max_corners=2000",
            "sfm.max_ba_points=4000",
        ),
        "full-size runs, meant for a GPU",
    ),
}

VARIANTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "full": ("default pipeline: everything is estimated from the video", ()),
    "static-only": ("plain 3DGS, moving objects ignored", ("dynamic=false",)),
    "oracle-depth": ("ground-truth depth", ("depth=oracle",)),
    "oracle-tracks": (
        "ground-truth point tracks and flow",
        ("pose_tracker=oracle", "tracker=oracle", "flow=oracle"),
    ),
    "oracle-poses": ("ground-truth camera poses", ("poses=gt",)),
    "oracle-all": (
        "ground-truth depth, tracks, flow and poses (upper bound of the back-end)",
        ("depth=oracle", "pose_tracker=oracle", "tracker=oracle", "flow=oracle", "poses=gt"),
    ),
    "no-depth-loss": (
        "no depth supervision during optimisation",
        ("train.lambda_depth=0", "train.lambda_track_depth=0"),
    ),
    "no-track-loss": (
        "no track supervision during optimisation",
        ("train.lambda_track=0", "train.lambda_track_depth=0"),
    ),
    "no-rigidity": ("no as-rigid-as-possible regulariser", ("train.lambda_rigid=0",)),
    "no-depth-correction": (
        "global depth alignment only (no correction field)",
        ("align.grid=null",),
    ),
    "depth-prior-ba": (
        "bundle adjustment with the monocular depth prior (off by default)",
        ("sfm.depth_weight=10",),
    ),
    # Back-end swaps: these need pretrained weights (downloaded on first use) or pycolmap.
    "depth-anything": (
        "zero-shot Depth Anything V2 (small) instead of the default depth back-end",
        ("depth=depth-anything",),
    ),
    "cotracker": ("CoTracker3 as the dense tracker", ("tracker=cotracker",)),
    "raft": ("RAFT optical flow instead of DIS", ("flow=raft",)),
    "colmap": (
        "camera poses from COLMAP (pycolmap) instead of the track-based SfM",
        ("poses=colmap",),
    ),
}

OPTIONAL_VARIANTS = ("depth-anything", "cotracker", "raft", "colmap")
"""Variants that need pretrained weights or an optional dependency."""

DEFAULT_VARIANTS = tuple(name for name in VARIANTS if name not in OPTIONAL_VARIANTS)
"""Variants that need nothing beyond the package itself."""

_WITH_FIGURES = ("full", "oracle-all", "static-only")


def run_one(
    scene: str,
    variant: str,
    profile: str,
    out_dir: Path,
    device: str = "cpu",
    lpips: str | None = None,
    seed: int = 0,
    extra_overrides: tuple[str, ...] = (),
    cache_dir: Path | None = None,
    figures: bool = True,
) -> Metrics:
    """Run one (scene, variant) pair, or load its metrics if it has already been run.

    ``figures`` also writes the qualitative outputs (videos, point clouds, checkpoint) of
    the main variants; metrics are always written first, so a run is never lost to a
    failure while rendering figures.
    """
    metrics_file = out_dir / "metrics.json"
    if metrics_file.exists():
        return load_json(metrics_file)
    spec = PROFILES[profile]
    seq = build_synthetic_sequence(SyntheticConfig(scene=scene, seed=seed, **spec.synth), cache_dir)
    # A variant has the last word: it must be able to undo what the extra overrides set.
    overrides = [*spec.overrides, *extra_overrides, *VARIANTS[variant][1]]
    cfg = apply_overrides(PipelineConfig(device=device, seed=seed), overrides)
    logger.info("=== %s / %s (%s profile) ===", scene, variant, profile)
    result = run_pipeline(seq, cfg, out_dir)
    metrics = evaluate_frontend(seq, result.frontend)
    scene_metrics = evaluate_scene(seq, result.frontend, result.scene, EvalConfig(lpips=lpips))
    scene_metrics["temporal"].update(metrics.pop("temporal"))
    metrics.update(scene_metrics)
    metrics["timings"] = dict(result.timings)
    save_json(metrics_file, metrics)
    if figures and variant in _WITH_FIGURES:
        try:
            save_run(out_dir, seq, result)
        except OSError as error:  # e.g. a full disk: the metrics are safe, carry on
            logger.warning("could not write the figures of %s/%s: %s", scene, variant, error)
    return metrics


def run_benchmark(
    out: str | Path,
    profile: str = "cpu",
    scenes: list[str] | None = None,
    variants: list[str] | None = None,
    device: str = "cpu",
    lpips: str | None = None,
    seed: int = 0,
    extra_overrides: tuple[str, ...] = (),
    figures: bool = True,
) -> dict[str, dict[str, Metrics]]:
    """Run the benchmark and write ``results.json`` and ``results.md`` in ``out``.

    A run that fails does not stop the others: its traceback goes to ``error.txt`` in its
    folder, the tables are written from the runs that succeeded, and a ``RuntimeError``
    naming the failed runs is raised at the end.

    Returns ``{scene: {variant: metrics}}``.
    """
    from recon4d.data.synthetic import SCENE_NAMES

    out = Path(out)
    scenes = list(SCENE_NAMES) if scenes is None else scenes
    variants = list(DEFAULT_VARIANTS) if variants is None else variants
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown or profile not in PROFILES:
        raise ValueError(f"unknown variant(s) {unknown} or profile {profile!r}")
    failures = []
    for scene in scenes:
        for variant in variants:
            run_dir = out / scene / variant
            try:
                run_one(
                    scene,
                    variant,
                    profile,
                    run_dir,
                    device,
                    lpips,
                    seed,
                    extra_overrides,
                    cache_dir=out / "cache",
                    figures=figures,
                )
            except Exception:
                logger.exception("run %s/%s failed", scene, variant)
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                failures.append(f"{scene}/{variant}")
    results = collect_results(out, profile)
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed: {', '.join(failures)}")
    return results


def collect_results(out: str | Path, profile: str = "") -> dict[str, dict[str, Metrics]]:
    """Gather every ``<scene>/<variant>/metrics.json`` under ``out`` and write the tables.

    Reading the results back from disk (rather than keeping them in memory) lets several
    processes, e.g. one per GPU, fill the same output directory.
    """
    from recon4d.data.synthetic import SCENE_NAMES

    out = Path(out)
    results: dict[str, dict[str, Metrics]] = {}
    for scene in SCENE_NAMES:
        for variant in VARIANTS:
            metrics_file = out / scene / variant / "metrics.json"
            if metrics_file.exists():
                results.setdefault(scene, {})[variant] = load_json(metrics_file)
    if results:
        save_json(out / "results.json", results)
        (out / "results.md").write_text(summarize(results, profile), encoding="utf-8")
        logger.info("results collected in %s", out / "results.md")
    return results


# ------------------------------------------------------------------------------ tables

# (column title, metric group, metric name, multiplier, number format)
_COLUMNS: dict[str, list[tuple[str, str, str, float, str]]] = {
    "Camera poses and depth": [
        ("ATE (cm)", "pose", "ate", 100.0, ".2f"),
        ("RPE-t (cm)", "pose", "rpe_trans", 100.0, ".2f"),
        ("RPE-r (deg)", "pose", "rpe_rot_deg", 1.0, ".3f"),
        ("AbsRel raw", "depth_raw", "abs_rel", 1.0, ".3f"),
        ("AbsRel aligned", "depth_aligned", "abs_rel", 1.0, ".3f"),
        ("AbsRel rendered", "depth_rendered", "abs_rel", 1.0, ".3f"),
    ],
    "Novel view synthesis": [
        ("PSNR", "nvs_test", "psnr", 1.0, ".2f"),
        ("SSIM", "nvs_test", "ssim", 1.0, ".3f"),
        ("LPIPS", "nvs_test", "lpips", 1.0, ".3f"),
        ("val PSNR", "nvs_val", "psnr", 1.0, ".2f"),
        ("val SSIM", "nvs_val", "ssim", 1.0, ".3f"),
        ("val LPIPS", "nvs_val", "lpips", 1.0, ".3f"),
        ("val PSNR (moving)", "nvs_val", "dynamic_psnr", 1.0, ".2f"),
    ],
    "Geometry and motion": [
        ("Chamfer (cm)", "geometry_rendered", "chamfer", 100.0, ".1f"),
        ("F@5cm", "geometry_rendered", "fscore@0.05", 1.0, ".3f"),
        ("F@10cm", "geometry_rendered", "fscore@0.1", 1.0, ".3f"),
        ("Chamfer moving (cm)", "geometry_dynamic", "chamfer", 100.0, ".1f"),
        ("3D EPE (cm)", "tracking_3d", "epe_3d", 100.0, ".1f"),
        ("Mask IoU", "motion_mask", "iou", 1.0, ".3f"),
        ("Track d_avg", "tracking", "delta_avg", 1.0, ".3f"),
    ],
    "Temporal consistency": [
        ("Depth TE raw", "temporal", "depth_raw", 1.0, ".4f"),
        ("Depth TE aligned", "temporal", "depth_aligned", 1.0, ".4f"),
        ("Depth TE rendered", "temporal", "depth_rendered", 1.0, ".4f"),
        ("Warp error (x1e-3)", "temporal", "warping_error", 1000.0, ".3f"),
        ("Warp error GT video", "temporal", "warping_error_gt_video", 1000.0, ".3f"),
        ("td-PSNR val", "temporal", "difference_psnr_val", 1.0, ".2f"),
    ],
}


def _value(metrics: Metrics, group: str, name: str) -> float | None:
    value = metrics.get(group, {}).get(name)
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def _table(rows: dict[str, list[Metrics]], columns: list[tuple[str, str, str, float, str]]) -> str:
    """Markdown table with one row per key of ``rows``; cells average the listed metrics."""
    used = [
        c
        for c in columns
        if any(_value(m, c[1], c[2]) is not None for ms in rows.values() for m in ms)
    ]
    lines = ["| | " + " | ".join(c[0] for c in used) + " |", "|---|" + "---:|" * len(used)]
    for label, group in rows.items():
        cells = []
        for _, metric_group, name, multiplier, fmt in used:
            values = [v for m in group if (v := _value(m, metric_group, name)) is not None]
            cells.append(format(multiplier * sum(values) / len(values), fmt) if values else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(results: dict[str, dict[str, Metrics]], profile: str = "") -> str:
    """Markdown report: per-scene results of the full pipeline, then every variant.

    A variant is averaged over the scenes it was run on.
    """
    scenes = list(results)
    variants = [v for v in VARIANTS if any(v in results[s] for s in scenes)]
    parts = [f"# Benchmark results ({profile} profile)" if profile else "# Benchmark results", ""]
    reference = "full" if "full" in variants else variants[0]
    parts.append(f"## Per scene, `{reference}` pipeline\n")
    for title, columns in _COLUMNS.items():
        rows = {s: [results[s][reference]] for s in scenes if reference in results[s]}
        parts += [f"**{title}**\n", _table(rows, columns), ""]
    if len(variants) > 1:
        parts.append("## Variants, averaged over the scenes\n")
        for title, columns in _COLUMNS.items():
            rows = {v: [results[s][v] for s in scenes if v in results[s]] for v in variants}
            parts += [f"**{title}**\n", _table(rows, columns), ""]
        parts.append("Variants:\n")
        parts += [f"- `{name}`: {VARIANTS[name][0]}" for name in variants]
        parts.append("")
    return "\n".join(parts)
