"""Command-line interface.

Examples::

    recon4d synth rolling --out outputs/rolling_preview
    recon4d run rolling --out outputs/rolling train.iterations=3000
    recon4d video my_clip.mp4 --out outputs/my_clip depth=depth-anything flow=raft
    recon4d benchmark --profile cpu --out results/cpu
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
import typer

from recon4d.config import apply_overrides, load_yaml, save_yaml
from recon4d.data.synthetic import SCENE_NAMES, SyntheticConfig, build_synthetic_sequence
from recon4d.pipeline import PipelineConfig, run_pipeline
from recon4d.utils import get_logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="3D/4D reconstruction from monocular video with Gaussian splatting.",
)
logger = get_logger("recon4d.cli")

Overrides = Annotated[
    list[str] | None, typer.Argument(help="Configuration overrides such as train.iterations=3000")
]


def _pipeline_config(config: Path | None, overrides: list[str] | None) -> PipelineConfig:
    cfg = load_yaml(config, PipelineConfig) if config else PipelineConfig()
    return apply_overrides(cfg, overrides or [])


@app.command()
def synth(
    scene: Annotated[str, typer.Argument(help=f"One of {', '.join(SCENE_NAMES)}")],
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs/synth"),
    frames: int = 48,
    width: int = 192,
    height: int = 144,
    seed: int = 0,
) -> None:
    """Render a synthetic benchmark sequence and save a preview of it."""
    from recon4d.viz import colorize_depth, overlay_mask, save_gif, side_by_side

    cfg = SyntheticConfig(scene=scene, n_frames=frames, width=width, height=height, seed=seed)
    seq = build_synthetic_sequence(cfg)
    gt = seq.gt
    panels = side_by_side(
        seq.images, colorize_depth(gt.depth), overlay_mask(seq.images, gt.dynamic_mask)
    )
    save_gif(out / f"{scene}.gif", panels, scale=2)
    for camera in gt.val_cameras:
        save_gif(out / f"{scene}_{camera.name}.gif", camera.images, fps=4.0, scale=2)
    logger.info("preview of %r written to %s", scene, out)


@app.command()
def run(
    scene: Annotated[str, typer.Argument(help=f"One of {', '.join(SCENE_NAMES)}")],
    overrides: Overrides = None,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs/run"),
    config: Annotated[Path | None, typer.Option(help="YAML configuration file")] = None,
    frames: int = 48,
    width: int = 192,
    height: int = 144,
    seed: int = 0,
    lpips: Annotated[str | None, typer.Option(help="LPIPS backbone (alex, vgg, squeeze)")] = None,
) -> None:
    """Reconstruct a synthetic scene and evaluate the result against its ground truth."""
    from recon4d.evaluation import EvalConfig, evaluate_frontend, evaluate_scene
    from recon4d.report import save_run

    cfg = apply_overrides(_pipeline_config(config, overrides), [f"seed={seed}"])
    seq = build_synthetic_sequence(
        SyntheticConfig(scene=scene, n_frames=frames, width=width, height=height, seed=seed)
    )
    result = run_pipeline(seq, cfg, out)
    metrics = evaluate_frontend(seq, result.frontend)
    scene_metrics = evaluate_scene(seq, result.frontend, result.scene, EvalConfig(lpips=lpips))
    scene_metrics["temporal"].update(metrics.pop("temporal"))
    metrics.update(scene_metrics)
    save_run(out, seq, result, metrics)
    for group in ("pose", "nvs_test", "nvs_val", "geometry_rendered", "tracking_3d"):
        if group in metrics:
            values = ", ".join(f"{k} {v:.4g}" for k, v in metrics[group].items())
            typer.echo(f"{group:18s} {values}")


@app.command()
def video(
    path: Annotated[Path, typer.Argument(help="Video file or folder of images")],
    overrides: Overrides = None,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs/video"),
    config: Annotated[Path | None, typer.Option(help="YAML configuration file")] = None,
    max_frames: int = 60,
    max_side: int = 384,
    fov: Annotated[float | None, typer.Option(help="Horizontal field of view in degrees")] = None,
) -> None:
    """Reconstruct a real video (no ground truth: only qualitative outputs are written)."""
    from recon4d.data.video import load_video
    from recon4d.report import save_run

    cfg = _pipeline_config(config, overrides)
    uses_oracle = "oracle" in (cfg.pose_tracker, cfg.tracker, cfg.flow)
    if cfg.depth.startswith("oracle") or uses_oracle or cfg.poses == "gt":
        defaults = [
            "depth=depth-anything",
            "pose_tracker=klt",
            "tracker=flow-chain",
            "flow=raft",
            "poses=sfm",
        ]
        logger.info("real footage has no ground truth: switching to %s", " ".join(defaults))
        cfg = apply_overrides(cfg, defaults)
    seq = load_video(path, max_frames=max_frames, max_side=max_side, fov_x_deg=fov)
    logger.info("loaded %d frames of %dx%d from %s", seq.num_frames, seq.width, seq.height, path)
    result = run_pipeline(seq, cfg, out)
    save_run(out, seq, result)


@app.command()
def benchmark(
    overrides: Overrides = None,
    out: Annotated[Path, typer.Option(help="Output directory")] = Path("results"),
    profile: Annotated[str, typer.Option(help="smoke, cpu (minutes) or gpu (full size)")] = "cpu",
    scenes: Annotated[str, typer.Option(help="Comma-separated scenes")] = ",".join(SCENE_NAMES),
    variants: Annotated[str | None, typer.Option(help="Comma-separated variants")] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lpips: Annotated[str | None, typer.Option(help="LPIPS backbone (alex, vgg, squeeze)")] = None,
    seed: int = 0,
    figures: Annotated[bool, typer.Option(help="Also write videos and point clouds")] = True,
) -> None:
    """Run the benchmark (scenes x pipeline variants) and write the result tables.

    Overrides apply to every run, e.g. ``depth=learned depth_checkpoint=tiny_depth.pth``.
    """
    from recon4d.benchmark import run_benchmark

    run_benchmark(
        out,
        profile=profile,
        scenes=scenes.split(","),
        variants=variants.split(",") if variants else None,
        device=device,
        lpips=lpips,
        seed=seed,
        extra_overrides=tuple(overrides or ()),
        figures=figures,
    )


@app.command()
def collect(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    profile: str = "",
) -> None:
    """Rebuild ``results.json`` and ``results.md`` from the runs found in a directory."""
    from recon4d.benchmark import collect_results

    results = collect_results(out, profile)
    typer.echo(f"{sum(len(v) for v in results.values())} runs collected in {out / 'results.md'}")


@app.command()
def config(out: Annotated[Path, typer.Argument(help="Where to write the YAML file")]) -> None:
    """Write the default configuration to a YAML file, as a starting point."""
    save_yaml(out, PipelineConfig())
    typer.echo(f"default configuration written to {out}")


if __name__ == "__main__":
    app()
