# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.53 | 0.39 | 0.059 | 0.015 | 0.017 | 0.018 |
| rolling | 0.50 | 0.43 | 0.057 | 0.018 | 0.013 | 0.017 |
| sliding | 0.66 | 0.56 | 0.088 | 0.034 | 0.040 | 0.036 |
| squash | 1.00 | 0.94 | 0.119 | 0.024 | 0.025 | 0.034 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| still | 26.23 | 0.873 | 22.61 | 0.776 | - |
| rolling | 25.64 | 0.865 | 23.14 | 0.789 | 17.66 |
| sliding | 25.01 | 0.869 | 20.78 | 0.678 | 15.91 |
| squash | 26.21 | 0.876 | 22.37 | 0.755 | 16.60 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 5.7 | 0.599 | 0.857 | - | - | - | 0.004 | 0.455 |
| rolling | 4.5 | 0.722 | 0.926 | 27.9 | 55.4 | 0.722 | 0.020 | 0.284 |
| sliding | 10.3 | 0.416 | 0.652 | 33.0 | 62.2 | 0.764 | 0.013 | 0.515 |
| squash | 9.4 | 0.370 | 0.674 | 10.1 | 12.6 | 0.732 | 0.013 | 0.511 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.0045 | 0.0046 | 0.0021 | 0.282 | 1.786 | - |
| rolling | 0.0071 | 0.0054 | 0.0043 | 0.854 | 1.782 | 30.66 |
| sliding | 0.0073 | 0.0069 | 0.0048 | 1.236 | 1.587 | 27.81 |
| squash | 0.0087 | 0.0072 | 0.0048 | 0.778 | 1.289 | 28.57 |

## Variants, averaged over the scenes

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.026 |
| static-only | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.037 |
| oracle-depth | 0.67 | 0.58 | 0.081 | 0.000 | 0.017 | 0.023 |
| oracle-tracks | 0.00 | 0.00 | 0.000 | 0.023 | 0.012 | 0.015 |
| oracle-poses | 0.00 | 0.00 | 0.000 | 0.023 | 0.016 | 0.022 |
| oracle-all | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.010 |
| no-depth-loss | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.028 |
| no-track-loss | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.028 |
| no-rigidity | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.027 |
| no-depth-correction | 0.67 | 0.58 | 0.081 | 0.023 | 0.025 | 0.028 |
| depth-prior-ba | 0.58 | 0.47 | 0.067 | 0.023 | 0.023 | 0.023 |
| colmap | 1.53 | 1.46 | 0.193 | 0.023 | 0.027 | 0.033 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| full | 25.77 | 0.870 | 22.22 | 0.750 | 16.72 |
| static-only | 23.24 | 0.829 | 20.47 | 0.719 | 11.70 |
| oracle-depth | 26.19 | 0.876 | 22.69 | 0.765 | 18.42 |
| oracle-tracks | 26.29 | 0.882 | 23.51 | 0.804 | 17.08 |
| oracle-poses | 25.93 | 0.874 | 23.13 | 0.794 | 16.77 |
| oracle-all | 26.62 | 0.893 | 24.45 | 0.833 | 20.90 |
| no-depth-loss | 25.92 | 0.871 | 22.17 | 0.750 | 16.84 |
| no-track-loss | 24.27 | 0.841 | 21.88 | 0.745 | 15.70 |
| no-rigidity | 25.84 | 0.872 | 22.18 | 0.748 | 16.90 |
| no-depth-correction | 25.57 | 0.868 | 22.21 | 0.747 | 17.04 |
| depth-prior-ba | 25.81 | 0.871 | 22.28 | 0.760 | 16.73 |
| colmap | 25.56 | 0.861 | 22.20 | 0.733 | 16.48 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 7.5 | 0.527 | 0.777 | 23.7 | 43.4 | 0.739 | 0.013 | 0.441 |
| static-only | 7.6 | 0.519 | 0.770 | - | 74.2 | 0.000 | 0.000 | 0.488 |
| oracle-depth | 6.7 | 0.565 | 0.790 | 14.0 | 29.5 | 0.758 | 0.012 | 0.440 |
| oracle-tracks | 4.6 | 0.737 | 0.915 | 13.9 | 24.4 | 0.958 | 0.000 | 1.000 |
| oracle-poses | 5.6 | 0.643 | 0.873 | 18.2 | 34.9 | 0.740 | 0.013 | 0.443 |
| oracle-all | 3.4 | 0.843 | 0.975 | 3.1 | 3.7 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 7.7 | 0.512 | 0.762 | 23.2 | 43.1 | 0.739 | 0.013 | 0.441 |
| no-track-loss | 7.3 | 0.536 | 0.786 | 24.9 | 43.2 | 0.739 | 0.013 | 0.441 |
| no-rigidity | 7.4 | 0.531 | 0.777 | 24.3 | 41.4 | 0.739 | 0.013 | 0.441 |
| no-depth-correction | 7.4 | 0.508 | 0.795 | 21.8 | 36.4 | 0.734 | 0.013 | 0.442 |
| depth-prior-ba | 6.5 | 0.571 | 0.797 | 19.9 | 40.5 | 0.741 | 0.012 | 0.441 |
| colmap | 9.4 | 0.398 | 0.687 | 14.5 | 42.2 | 0.675 | 0.021 | 0.433 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.0069 | 0.0060 | 0.0040 | 0.787 | 1.611 | 29.01 |
| static-only | 0.0069 | 0.0060 | 0.0042 | 1.111 | 1.611 | 26.32 |
| oracle-depth | 0.0001 | 0.0034 | 0.0038 | 0.777 | 1.611 | 30.35 |
| oracle-tracks | 0.0069 | 0.0039 | 0.0028 | 0.690 | 1.611 | 29.44 |
| oracle-poses | 0.0069 | 0.0058 | 0.0036 | 0.705 | 1.611 | 28.91 |
| oracle-all | 0.0001 | 0.0002 | 0.0021 | 0.662 | 1.611 | 32.45 |
| no-depth-loss | 0.0069 | 0.0060 | 0.0041 | 0.772 | 1.611 | 28.96 |
| no-track-loss | 0.0069 | 0.0060 | 0.0047 | 1.140 | 1.611 | 27.81 |
| no-rigidity | 0.0069 | 0.0060 | 0.0041 | 0.838 | 1.611 | 28.98 |
| no-depth-correction | 0.0069 | 0.0065 | 0.0041 | 0.769 | 1.611 | 29.24 |
| depth-prior-ba | 0.0069 | 0.0060 | 0.0038 | 0.781 | 1.611 | 28.89 |
| colmap | 0.0069 | 0.0067 | 0.0050 | 0.759 | 1.611 | 28.52 |

Variants:

- `full`: default pipeline: everything is estimated from the video
- `static-only`: plain 3DGS, moving objects ignored
- `oracle-depth`: ground-truth depth
- `oracle-tracks`: ground-truth point tracks and flow
- `oracle-poses`: ground-truth camera poses
- `oracle-all`: ground-truth depth, tracks, flow and poses (upper bound of the back-end)
- `no-depth-loss`: no depth supervision during optimisation
- `no-track-loss`: no track supervision during optimisation
- `no-rigidity`: no as-rigid-as-possible regulariser
- `no-depth-correction`: global depth alignment only (no correction field)
- `depth-prior-ba`: bundle adjustment with the monocular depth prior (off by default)
- `colmap`: camera poses from COLMAP (pycolmap) instead of the track-based SfM
