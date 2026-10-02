# Benchmark protocol

The benchmark answers one question: *how good is each stage of the pipeline, and how much
does each stage cost the final reconstruction?* That requires ground truth for everything
at once (depth, camera poses, point tracks, motion masks, static and dynamic geometry,
novel views at the same instants), which no real dataset provides. The benchmark is
therefore procedural: scenes are analytic, images are ray traced, and ground truth is
exact.

## Scenes

A scene is a textured 6 x 3 x 6 m room holding analytic primitives (spheres, boxes,
cylinders) with procedural solid textures. Static props sit in the centre and at the back;
moving objects travel in a ring around the centre. Four scenes are defined
(`recon4d.data.synthetic.scenes`):

| scene | motion |
|---|---|
| `still` | none: measures camera poses, depth and static geometry in isolation |
| `rolling` | two balls rolling without slipping, one passing behind the centre-piece |
| `sliding` | a spinning crate sliding on the floor and a bouncing, tumbling ball |
| `squash` | a non-rigid squash-and-stretch blob swaying sideways, and a sliding can |

The camera orbits the scene on a 54 degree arc with a hand-held shake. A sequence is fully
determined by a `SyntheticConfig`; nothing is stored, and
`python -m recon4d.cli synth <scene>` renders a preview.

### Why the images are not an inverse crime

The images come from a ray tracer (`recon4d.data.synthetic.render`): analytic
ray-primitive intersections, Lambertian shading, 3 x 3 supersampling, 8-bit quantisation.
The reconstruction renders with Gaussian splatting. The two image formation models share
no code and no approximation, so a reconstruction cannot score well merely by matching the
generator.

### The oracle

Because the scene is analytic, ground truth is not limited to what was rendered.
`SceneOracle` lifts *any* pixel of *any* frame to a material point (object index and
object-local coordinates) and follows it through time:

* `tracks(query_frames, query_uv)`: exact 2D/3D trajectories and visibility of arbitrary
  query points. A tracker is evaluated on its own queries, not on a fixed annotation set.
* `scene_flow(src, dst)`: dense optical flow, its validity, and the 3D displacement that
  causes it.
* `surface_cloud(...)`: samples of the static surfaces and of the moving surfaces over
  time, restricted to what the training video actually observes.

Visibility is decided by casting a ray from the camera to the point, not by comparing with
a depth buffer.

## Data split

* **Training video**: the monocular sequence. Every 8th frame is held out of the scene
  optimisation (the front-end sees all frames, as COLMAP-based pipelines do).
* **Validation cameras**: two fixed cameras off the training trajectory, rendered at the
  training timestamps. They measure novel-view synthesis *at the same instant*, the
  quantity a 4D reconstruction is supposed to deliver and that a monocular video cannot
  measure on itself (the DyCheck argument).
* **Co-visibility**: a validation pixel is evaluated only if its surface point is seen by
  the training video. For points on moving objects the point is transported with the
  object's true motion before the test.

## Alignment

A monocular reconstruction lives in an arbitrary similarity frame. One Sim(3) maps it to
the ground-truth frame; it is estimated **from the camera poses only**
(`recon4d.metrics.pose.align_frames`), never from ground-truth geometry.

* ATE is reported with the standard position-only Umeyama alignment.
* Geometry and validation cameras use an alignment that also uses the camera
  orientations. Position-only alignment leaves the rotation about the direction of travel
  poorly determined for near-straight camera paths: a tilt of 1 degree displaces a scene
  5 m away by 9 cm.

## Metrics

| group | metric | definition |
|---|---|---|
| pose | ATE | RMSE of camera positions after Sim(3) alignment |
| | RPE-t, RPE-r | frame-to-frame relative pose error (translation, rotation) |
| | orientation | error after the best global rotation of the orientations |
| depth | AbsRel, RMSE, delta<1.25 | standard; reported for the raw prediction with the best per-frame alignment (a ceiling), for the aligned depth and for the depth rendered from the scene |
| tracking | delta_avg, AJ, OA | TAP-Vid, at 256 x 256 |
| | EPE (tracked) | error of the points the tracker itself reports as visible |
| motion | mask IoU | dynamic mask against the true one |
| view synthesis | PSNR, SSIM, LPIPS | held-out frames; validation cameras (co-visible pixels); moving objects only |
| geometry | accuracy, completeness, Chamfer | mean nearest-neighbour distances, both ways and their average |
| | F-score @ 5 cm / 10 cm | harmonic mean of precision and recall |
| 3D tracking | EPE, delta@5cm, delta@10cm | true surface points handed to the scene's motion field at the frame they are first seen and compared with their true trajectory over the whole video |
| temporal | depth temporal error | 3D inconsistency between consecutive depth maps along true correspondences, relative to depth |
| | warping error | squared difference between each frame and its flow-warped successor |
| | temporal-difference PSNR | fidelity of the frame-to-frame *changes* on the validation cameras |

Three geometries are scored: the point cloud fused from the aligned depth maps
(`geometry_fused`), the surface rendered by the optimised scene (`geometry_rendered`), and
the moving Gaussians at each instant against the moving surfaces at that instant
(`geometry_dynamic`).

## Variants

A variant is the default pipeline with a few options changed
(`recon4d.benchmark.VARIANTS`).

*Oracle analysis*: one stage at a time is replaced by its ground truth.

| variant | replaced by ground truth |
|---|---|
| `oracle-depth` | depth maps |
| `oracle-tracks` | point tracks and optical flow |
| `oracle-poses` | camera poses |
| `oracle-all` | all of the above: the upper bound of the scene optimisation |

*Ablations*: `static-only` (plain 3DGS), `no-depth-loss`, `no-track-loss`, `no-rigidity`,
`no-depth-correction`; and one optional mechanism switched on: `depth-prior-ba` (monocular
depth prior in bundle adjustment).

*Back-end swaps* (need pretrained weights or an optional dependency): `depth-anything`,
`cotracker`, `raft`, `colmap`.

## Profiles

| profile | sequence | optimisation | purpose |
|---|---|---|---|
| `smoke` | 8 frames, 64 x 48 | 40 steps | integration tests |
| `cpu` | 24 frames, 128 x 96 | 800 steps on random crops | a laptop CPU, minutes per run |
| `gpu` | 60 frames, 384 x 288 | 7000 steps | full size |

```bash
recon4d benchmark --profile cpu --out results/cpu
python scripts/run_benchmark_multi_gpu.py --out results/gpu --profile gpu --lpips alex
```

Runs are resumable, and `recon4d collect <dir>` rebuilds the tables from the runs found on
disk.

## The depth back-ends of the benchmark

* `oracle-noisy` (default of the CPU profile): ground-truth depth degraded by a model of
  the failure modes of monocular networks: an unknown scale per frame, low-frequency
  multiplicative errors, blurred discontinuities, pixel noise. It makes the difficulty of
  the depth prior a dial rather than an accident of a particular network.
* `learned`: a compact U-Net trained on random scenes from the same scene grammar, never
  on the benchmark scenes (`scripts/train_depth.py`). It predicts depth from pixels, up
  to scale, one frame at a time.
* `depth-anything`: Depth Anything V2, zero-shot.
