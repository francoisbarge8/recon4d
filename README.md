# recon4d

**3D/4D reconstruction from monocular video**: depth prediction, camera pose estimation
and point tracking, fused into a scene of static and moving 3D Gaussians that can be
rendered from any viewpoint at any instant of the video.

[![CI](https://github.com/francoisbarge8/recon4d/actions/workflows/ci.yml/badge.svg)](https://github.com/francoisbarge8/recon4d/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Everything is PyTorch, written from scratch, and runs on a laptop CPU as well as on a GPU:
the differentiable Gaussian-splatting rasterizer, bundle adjustment, structure-from-motion
on point tracks, depth alignment, motion segmentation, and the 4D scene optimisation. The
project also ships its own benchmark, with exact ground truth for every quantity the
pipeline estimates.

## Pipeline

```mermaid
flowchart LR
    V[video] --> D[depth prediction]
    V --> T[point tracking]
    V --> F[optical flow]
    T --> P[camera poses<br/>SfM + bundle adjustment]
    D -. optional prior .-> P
    P --> A[depth alignment]
    D --> A
    A --> M[motion segmentation]
    F --> M
    P --> M
    M --> I[initialisation]
    A --> I
    T --> I
    I --> O[4D Gaussian<br/>optimisation]
    O --> R[novel views, depth,<br/>3D tracks, point clouds]
```

| stage | what it does | back-ends |
|---|---|---|
| depth | one depth map per frame, up to scale | Depth Anything V2, in-domain U-Net, oracle (+ noise model) |
| tracking | 2D point tracks through the video | KLT, flow chaining, CoTracker3, oracle |
| optical flow | dense motion between frames | DIS, RAFT, oracle |
| camera poses | incremental SfM on the tracks, bundle adjustment | built-in, COLMAP (pycolmap), ground truth |
| depth alignment | makes the depth maps consistent with the poses and with each other | scale / affine + smooth correction field |
| motion segmentation | which pixels and tracks move on their own | rigid-flow residuals |
| scene | static 3D Gaussians + dynamic Gaussians driven by a few SE(3) motion bases | pure-PyTorch rasterizer |

Each stage can be replaced by its ground truth (`oracle`), which is how the benchmark
attributes the final error to depth, tracking or pose estimation. [docs/DESIGN.md](docs/DESIGN.md)
explains every stage and the reasons behind the design.

## What is implemented

* **A differentiable 3D Gaussian splatting rasterizer in pure PyTorch**
  ([rasterizer.py](src/recon4d/gaussians/rasterizer.py)): EWA projection, tile binning,
  front-to-back compositing, exact opacity-aware bounding boxes. Verified against a
  brute-force reference to 1e-10 and by `gradcheck`.
* **A 4D scene model** in the spirit of Shape of Motion: dynamic Gaussians follow a convex
  blend of a few rigid motion bases; 2D tracks supervise motion through rendered 3D
  correspondences; as-rigid-as-possible and smoothness regularisers; adaptive density
  control.
* **Structure-from-motion on point tracks** with a **Levenberg-Marquardt bundle
  adjustment** written in PyTorch (Schur complement, analytic Jacobians checked against
  autograd, Huber loss, optional shared focal length, optional monocular depth prior with
  per-camera scale and shift).
* **Depth alignment** of per-frame monocular depth to the sparse SfM structure (robust
  global fit plus a smooth correction field).
* **Training-free motion segmentation** from the disagreement between observed and rigid
  optical flow.
* **A procedural 4D benchmark with exact ground truth** ([docs/BENCHMARK.md](docs/BENCHMARK.md)):
  ray-traced scenes with moving objects, an oracle answering arbitrary queries (tracks,
  flow, scene flow, surface samples), held-out cameras at the training instants with
  co-visibility masks.
* **Metrics**: PSNR, SSIM, LPIPS; Chamfer distance, accuracy / completeness, F-score;
  ATE, RPE, orientation error; AbsRel / RMSE / delta depth metrics; TAP-Vid tracking
  metrics; 3D trajectory error; temporal consistency (depth temporal error, warping
  error, temporal-difference PSNR).
* **COLMAP interoperability**: export of the front-end as a COLMAP dataset (opens in
  COLMAP's GUI, feeds other 3DGS / NeRF trainers), and COLMAP as an alternative pose
  back-end.

* **TSDF fusion** of posed depth maps with surface extraction as points or as a triangle
  mesh (naive surface nets), in PyTorch ([tsdf.py](src/recon4d/fusion/tsdf.py)).

## Results

The complete benchmark at the `cpu` profile: 4 scenes x 12 variants x 3 seeds, that is 144
reconstructions of 24 frames at 128 x 96 with 800 optimisation steps, run on a 4-core CPU
without a GPU (53 minutes, one process per scene). A seed changes the layout, the textures
and the camera shake of every scene; each number is the **mean ± standard deviation over
the 3 seeds**. Depth comes from the noisy oracle (the default of this profile, see
[docs/BENCHMARK.md](docs/BENCHMARK.md)). Every metric of every variant:
[results/cpu/results.md](results/cpu/results.md); the metrics of each run:
`results/cpu/seed*/results.json`; the commands:
[docs/BENCHMARK.md](docs/BENCHMARK.md#several-seeds).

![input frame, reconstruction and rendered depth](results/cpu/figures/sliding_full.png)

*`sliding`, default pipeline, seed 0: input frame, reconstruction, rendered depth
([video](results/cpu/figures/sliding_full_reconstruction.gif),
[orbiting camera, time frozen](results/cpu/figures/sliding_full_bullet_time.gif)).*

**Default pipeline** (KLT + flow chaining with DIS, SfM + bundle adjustment, depth
alignment, motion segmentation, 4D Gaussians), per scene:

| | ATE (cm) | aligned depth AbsRel (%) | PSNR held-out frames | PSNR held-out cameras | PSNR moving objects | Chamfer (cm) | 3D EPE (cm) | mask IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 0.52 ± 0.04 | 2.6 ± 1.4 | 26.66 ± 0.73 | 23.39 ± 1.05 | - | 7.6 ± 3.6 | - | - |
| rolling | 0.47 ± 0.04 | 2.4 ± 0.9 | 25.04 ± 0.36 | 22.38 ± 0.76 | 17.53 ± 1.12 | 6.6 ± 2.3 | 35.9 ± 14.7 | 0.726 ± 0.019 |
| sliding | 0.60 ± 0.11 | 3.3 ± 1.5 | 25.16 ± 0.45 | 21.74 ± 1.57 | 17.92 ± 0.25 | 8.7 ± 3.3 | 41.4 ± 14.4 | 0.769 ± 0.013 |
| squash | 0.78 ± 0.21 | 2.8 ± 0.8 | 26.10 ± 0.27 | 22.18 ± 0.87 | 17.87 ± 0.76 | 9.1 ± 1.4 | 17.4 ± 5.6 | 0.725 ± 0.010 |

PSNR on held-out cameras is computed on the pixels the video observes; "moving objects"
restricts it to them. 3D EPE: true surface points handed to the scene's motion field and
compared with their true trajectory over the whole video.

**Where the error comes from.** One stage at a time is replaced by its ground truth,
averaged over the four scenes:

| | PSNR held-out cameras | PSNR moving objects | Chamfer (cm) | Chamfer moving (cm) | 3D EPE (cm) | track δ_avg |
|---|---:|---:|---:|---:|---:|---:|
| `full` | 22.42 ± 0.22 | 17.77 ± 0.62 | 8.0 ± 0.6 | 12.6 ± 2.4 | 31.6 ± 7.6 | 0.429 ± 0.016 |
| `oracle-depth` | 22.53 ± 0.18 | 17.92 ± 0.72 | 7.8 ± 0.9 | 12.1 ± 1.7 | 28.5 ± 1.6 | 0.430 ± 0.016 |
| `oracle-poses` | 23.75 ± 0.14 | 18.15 ± 0.62 | 5.2 ± 0.1 | 10.4 ± 1.7 | 28.0 ± 7.3 | 0.430 ± 0.016 |
| `oracle-tracks` | 24.16 ± 0.12 | 19.32 ± 0.41 | 4.4 ± 0.0 | 4.0 ± 0.3 | 6.7 ± 1.0 | 1.000 ± 0.000 |
| `oracle-all` | 24.57 ± 0.11 | 21.03 ± 0.15 | 3.4 ± 0.0 | 2.8 ± 0.3 | 2.7 ± 0.8 | 1.000 ± 0.000 |
| `static-only` | 20.26 ± 0.28 | 11.24 ± 0.55 | 8.3 ± 0.7 | - | 74.2 ± 0.2 | 0.483 ± 0.009 |

* **Point tracking is the bottleneck.** Exact tracks and flow (which also make the SfM
  poses exact) bring the 3D trajectory error from 32 cm to 7 cm and the moving surfaces
  from 12.6 cm to 4.0 cm. The dense tracker chains optical flow and drops points at
  occlusions (TAP-Vid δ_avg 0.43).
* Exact poses mostly help the static geometry (Chamfer 8.0 to 5.2 cm, +1.3 dB on the
  held-out cameras), although the estimated trajectory is already within 0.6 cm.
* Exact depth changes little (+0.1 dB): once aligned to the poses, the noisy depth is
  already within 2.8% of the truth.
* With every input exact, the moving objects (21.0 dB) remain 3.5 dB below the whole image:
  that gap belongs to the scene optimisation at this budget.
* Ignoring the motion (`static-only`, plain 3DGS) costs 2.2 dB on the held-out cameras and
  6.5 dB on the moving objects.

**Ablations and pose back-ends**, averaged over the four scenes:

| | ATE (cm) | RPE-r (deg) | aligned depth AbsRel (%) | PSNR held-out cameras | PSNR moving objects | Chamfer (cm) | 3D EPE (cm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.59 ± 0.07 | 0.077 ± 0.003 | 2.8 ± 0.5 | 22.42 ± 0.22 | 17.77 ± 0.62 | 8.0 ± 0.6 | 31.6 ± 7.6 |
| `no-depth-loss` | 0.59 ± 0.07 | 0.077 ± 0.003 | 2.8 ± 0.5 | 22.37 ± 0.24 | 17.56 ± 0.74 | 8.1 ± 0.4 | 32.1 ± 7.1 |
| `no-track-loss` | 0.59 ± 0.07 | 0.077 ± 0.003 | 2.8 ± 0.5 | 22.21 ± 0.12 | 16.33 ± 0.41 | 7.6 ± 0.5 | 31.3 ± 6.0 |
| `no-rigidity` | 0.59 ± 0.07 | 0.077 ± 0.003 | 2.8 ± 0.5 | 22.40 ± 0.22 | 17.94 ± 0.61 | 8.0 ± 0.6 | 34.0 ± 5.5 |
| `no-depth-correction` | 0.59 ± 0.07 | 0.077 ± 0.003 | 4.1 ± 0.5 | 22.01 ± 0.20 | 16.89 ± 0.67 | 7.7 ± 0.3 | 35.8 ± 10.6 |
| `depth-prior-ba` | 0.63 ± 0.18 | 0.075 ± 0.021 | 2.6 ± 1.4 | 22.42 ± 1.55 | 17.47 ± 0.95 | 10.1 ± 5.3 | 34.8 ± 5.6 |
| `colmap` | 2.05 ± 0.90 | 0.215 ± 0.030 | 4.4 ± 2.8 | 21.33 ± 1.63 | 16.62 ± 1.36 | 14.7 ± 8.4 | 33.6 ± 8.6 |

* The track loss is what animates the moving objects: without it they lose 1.4 dB.
* The depth correction field brings the aligned depth from 4.1% to 2.8% AbsRel (+0.4 dB on
  the held-out cameras).
* The depth loss and the rigidity regulariser make no difference larger than the spread
  over the seeds at this budget.
* The monocular depth prior in bundle adjustment does not help on average and makes the
  geometry unpredictable (Chamfer 10.1 ± 5.3 cm against 8.0 ± 0.6 cm), which is why it is
  off by default.
* COLMAP is close to the track-based SfM on the static scene (ATE 0.75 against 0.52 cm);
  the gap grows with the moving objects, which COLMAP is not told about (4.4 against
  0.8 cm on `squash`).

**Depth predicted from pixels.** The same benchmark with the in-domain U-Net as depth
back-end (`depth=learned`, the setting of the Kaggle notebook), trained here on CPU for 12
epochs instead of 40 (2400 frames of random scenes, 1.8% AbsRel on its validation scenes).
Every table: [results/cpu-learned/results.md](results/cpu-learned/results.md).

| depth back-end | raw depth AbsRel (%) | aligned depth AbsRel (%) | PSNR held-out cameras | PSNR moving objects | Chamfer (cm) | 3D EPE (cm) |
|---|---:|---:|---:|---:|---:|---:|
| noisy oracle (default of the profile) | 3.5 ± 0.3 | 2.8 ± 0.5 | 22.42 ± 0.22 | 17.77 ± 0.62 | 8.0 ± 0.6 | 31.6 ± 7.6 |
| in-domain U-Net | 2.1 ± 0.2 | 2.8 ± 0.4 | 22.07 ± 0.16 | 16.55 ± 0.16 | 8.1 ± 0.6 | 40.5 ± 7.8 |
| in-domain U-Net, depth prior in BA | 2.1 ± 0.2 | 2.1 ± 0.2 | 22.57 ± 0.26 | 16.73 ± 0.17 | 6.3 ± 0.3 | 36.8 ± 4.8 |

* The network beats the noise model on the static scene (2.0% against 3.5% AbsRel on the
  static pixels of the three dynamic scenes, and far more consistent from frame to frame)
  but not on the moving objects (5.1% against 3.7%), which is where it loses end to end: 1.2 dB on the moving
  objects, 41 cm of 3D trajectory error against 32.
* With this network, the depth prior in bundle adjustment **helps**: Chamfer 8.1 to
  6.3 cm, +0.5 dB on the held-out cameras, aligned depth 2.8% to 2.1%, with a small
  spread. With the noise model it made the geometry unpredictable (above). It stays off by
  default, the default depth of the profile being the noise model; with a network, turn it
  on with `sfm.depth_weight=10`.
* The network has to run at the size it was trained at (192 x 144): applied to the
  128 x 96 frames directly, its error on a benchmark scene went from 1.5% to 4.3% (and to
  5.5% on the 384 x 288 frames of the `gpu` profile). The `learned` back-end now resizes
  the frames.

**Not run yet.** The full-size `gpu` profile (60 frames at 384 x 288, 7000 steps) needs a
GPU: it is what [the Kaggle notebook](notebooks/kaggle_benchmark.ipynb) runs. The container
used for the CPU runs could not download pretrained weights (Hugging Face and
download.pytorch.org were out of reach), so LPIPS and the `depth-anything`, `cotracker` and
`raft` variants are missing too, as is the cross-check against gsplat (CUDA only).

## Installation

```bash
git clone https://github.com/francoisbarge8/recon4d.git
cd recon4d
pip install -e .
```

Python 3.10+ and PyTorch 2.1+. Optional extras:

| extra | adds |
|---|---|
| `.[dev]` | pytest, ruff |
| `.[lpips]` | the LPIPS metric |
| `.[depth]` | Depth Anything V2 (through `transformers`) |
| `.[colmap]` | COLMAP poses (through `pycolmap`) |

## Quick start

```bash
# Render a benchmark scene and look at it (video | depth | moving objects)
recon4d synth rolling --out outputs/preview

# Reconstruct it and evaluate against the ground truth
recon4d run rolling --out outputs/rolling --frames 24 --width 128 --height 96 \
    train.iterations=800 "train.crop=[96,72]"

# Reconstruct your own video (Depth Anything V2 + RAFT, unknown focal length)
pip install -e ".[depth]"
recon4d video clip.mp4 --out outputs/clip device=cuda

# The benchmark: scenes x pipeline variants, with result tables
recon4d benchmark --profile cpu --out results/cpu
```

Any option of the pipeline can be overridden on the command line as `key=value`
(`recon4d config default.yaml` writes them all to a file). A run writes its metrics, the
optimised scene (`.pt` and 3DGS-compatible `.ply`), a fused point cloud and videos:
reconstruction against input, the scene from an orbiting camera with time frozen and with
time running, and the held-out cameras.

From Python:

```python
from recon4d.benchmark import PROFILES
from recon4d.config import apply_overrides
from recon4d.data.synthetic import SyntheticConfig, build_synthetic_sequence
from recon4d.evaluation import evaluate_scene
from recon4d.gaussians.render import Camera
from recon4d.pipeline import PipelineConfig, run_pipeline

seq = build_synthetic_sequence(SyntheticConfig(scene="sliding", n_frames=24, width=128, height=96))
cfg = apply_overrides(PipelineConfig(), PROFILES["cpu"].overrides)  # a CPU-sized budget
result = run_pipeline(seq, cfg)

front = result.frontend            # poses, aligned depth, tracks, motion masks
camera = Camera(front.K, front.w2c[0], seq.width, seq.height)
view = result.scene.render(camera, frame=12)   # the scene at time 12 seen from camera 0
print(view.color.shape, view.depth.shape)

metrics = evaluate_scene(seq, front, result.scene)
print(metrics["nvs_val"]["psnr"], metrics["tracking_3d"]["epe_3d"])
```

## Full-size runs on a GPU

The CPU profile is deliberately small. [notebooks/kaggle_benchmark.ipynb](notebooks/kaggle_benchmark.ipynb)
runs the full-size benchmark (60 frames at 384 x 288, 7000 optimisation steps) on a free
Kaggle notebook with two T4 GPUs: it installs the package, runs the tests, trains the
in-domain depth network, runs every variant with one worker per GPU and zips the results.

```bash
python scripts/train_depth.py --out assets/checkpoints/tiny_depth.pth
python scripts/run_benchmark_multi_gpu.py --out results/gpu --profile gpu --lpips alex \
    depth=learned depth_checkpoint=assets/checkpoints/tiny_depth.pth flow=raft
```

## Repository layout

```
src/recon4d/
  data/synthetic/   procedural scenes, ray tracer, ground-truth oracle
  frontend/         depth, optical flow, point tracking, SfM + bundle adjustment, motion segmentation
  fusion/           depth alignment, point-cloud fusion
  gaussians/        rasterizer, scene model, motion bases, initialisation, trainer
  geometry/         cameras, rotations, alignment, triangulation
  metrics/          image, geometry, pose, depth, tracking and temporal metrics
  io/               COLMAP models
  pipeline.py       the end-to-end pipeline
  evaluation.py     evaluation against ground truth
  benchmark.py      scenes x variants, result tables
  cli.py            command-line interface
scripts/            depth-network training, multi-GPU benchmark, gsplat cross-check
notebooks/          Kaggle notebook for the full-size benchmark
tests/              unit and integration tests
docs/               design notes, benchmark protocol
```

## Tests

```bash
pip install -e ".[dev]"
pytest
ruff check src tests scripts
```

The suite covers the geometry, the ray tracer and its oracle, the rasterizer (against a
brute-force reference, `gradcheck`, tile-size invariance), every metric, bundle adjustment
(Jacobians against autograd, convergence, robustness), SfM, trackers, depth alignment,
motion bases, the scene model and the end-to-end pipeline on tiny sequences.

## Limitations

Stated in full in [docs/DESIGN.md](docs/DESIGN.md#limitations). In short: the CPU
rasterizer is orders of magnitude slower than a CUDA kernel, so local runs are small; the
motion model suits rigid and articulated motion, not fluids or topology changes; motion
segmentation needs parallax; the benchmark is synthetic (real videos run, but without
quantitative evaluation).

## References

* Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
* Zwicker et al., *EWA Splatting*, IEEE TVCG 2002.
* Wang et al., *Shape of Motion: 4D Reconstruction from a Single Video*, 2024.
* Luiten et al., *Dynamic 3D Gaussians: Tracking by Persistent Dynamic View Synthesis*, 3DV 2024.
* Schönberger and Frahm, *Structure-from-Motion Revisited* (COLMAP), CVPR 2016.
* Triggs et al., *Bundle Adjustment: A Modern Synthesis*, 1999.
* Yang et al., *Depth Anything V2*, NeurIPS 2024.
* Karaev et al., *CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos*, 2024.
* Teed and Deng, *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020.
* Doersch et al., *TAP-Vid: A Benchmark for Tracking Any Point in a Video*, NeurIPS 2022.
* Gao et al., *Monocular Dynamic View Synthesis: A Reality Check* (DyCheck), NeurIPS 2022.
* Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS), CVPR 2018.
* Knapitsch et al., *Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction*, SIGGRAPH 2017 (F-score protocol).

## License

MIT, see [LICENSE](LICENSE).
