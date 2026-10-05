# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.53 | 0.39 | 0.059 | 0.034 | 0.018 | 0.018 |
| rolling | 0.50 | 0.43 | 0.057 | 0.033 | 0.014 | 0.017 |
| sliding | 0.66 | 0.56 | 0.088 | 0.035 | 0.036 | 0.033 |
| squash | 1.00 | 0.94 | 0.119 | 0.034 | 0.023 | 0.030 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| still | 26.28 | 0.873 | 22.76 | 0.779 | - |
| rolling | 25.24 | 0.855 | 23.17 | 0.788 | 18.79 |
| sliding | 25.33 | 0.871 | 21.33 | 0.696 | 17.91 |
| squash | 26.41 | 0.886 | 23.12 | 0.776 | 18.36 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 5.6 | 0.610 | 0.869 | - | - | - | 0.004 | 0.454 |
| rolling | 4.4 | 0.755 | 0.930 | 15.8 | 52.9 | 0.712 | 0.021 | 0.283 |
| sliding | 9.8 | 0.415 | 0.646 | 21.5 | 56.6 | 0.773 | 0.015 | 0.508 |
| squash | 9.3 | 0.375 | 0.637 | 6.2 | 11.0 | 0.730 | 0.014 | 0.507 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.2496 | 0.0103 | 0.0021 | 0.275 | 1.786 | - |
| rolling | 0.2495 | 0.0104 | 0.0042 | 0.802 | 1.782 | 30.44 |
| sliding | 0.2501 | 0.0111 | 0.0046 | 1.181 | 1.587 | 29.64 |
| squash | 0.2506 | 0.0112 | 0.0047 | 0.792 | 1.289 | 29.76 |

## Variants, averaged over the scenes

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.67 | 0.58 | 0.081 | 0.034 | 0.023 | 0.025 |
| static-only | 0.67 | 0.58 | 0.081 | 0.034 | 0.023 | 0.037 |
| oracle-depth | 0.67 | 0.58 | 0.081 | 0.000 | 0.017 | 0.023 |
| oracle-tracks | 0.00 | 0.00 | 0.000 | 0.034 | 0.012 | 0.013 |
| oracle-poses | 0.00 | 0.00 | 0.000 | 0.034 | 0.015 | 0.018 |
| oracle-all | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.010 |
| no-depth-loss | 0.67 | 0.58 | 0.081 | 0.034 | 0.023 | 0.026 |
| no-track-loss | 0.67 | 0.58 | 0.081 | 0.034 | 0.023 | 0.026 |
| no-rigidity | 0.67 | 0.58 | 0.081 | 0.034 | 0.023 | 0.025 |
| no-depth-correction | 0.67 | 0.58 | 0.081 | 0.034 | 0.035 | 0.028 |
| depth-prior-ba | 0.51 | 0.48 | 0.062 | 0.034 | 0.019 | 0.024 |
| colmap | 1.50 | 1.45 | 0.191 | 0.034 | 0.025 | 0.030 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| full | 25.81 | 0.871 | 22.60 | 0.760 | 18.35 |
| static-only | 23.27 | 0.831 | 20.58 | 0.723 | 11.80 |
| oracle-depth | 26.19 | 0.876 | 22.69 | 0.765 | 18.42 |
| oracle-tracks | 25.95 | 0.881 | 24.25 | 0.821 | 19.80 |
| oracle-poses | 26.10 | 0.876 | 23.90 | 0.812 | 18.83 |
| oracle-all | 26.62 | 0.893 | 24.45 | 0.833 | 20.90 |
| no-depth-loss | 25.80 | 0.870 | 22.53 | 0.756 | 18.23 |
| no-track-loss | 24.68 | 0.848 | 22.34 | 0.755 | 16.80 |
| no-rigidity | 25.70 | 0.869 | 22.58 | 0.759 | 18.45 |
| no-depth-correction | 25.42 | 0.863 | 22.19 | 0.738 | 17.64 |
| depth-prior-ba | 25.98 | 0.874 | 23.43 | 0.792 | 18.51 |
| colmap | 25.62 | 0.864 | 22.84 | 0.754 | 18.01 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 7.3 | 0.539 | 0.770 | 14.5 | 40.2 | 0.739 | 0.014 | 0.438 |
| static-only | 7.5 | 0.526 | 0.761 | - | 74.2 | 0.000 | 0.000 | 0.488 |
| oracle-depth | 6.7 | 0.565 | 0.790 | 14.0 | 29.5 | 0.758 | 0.012 | 0.440 |
| oracle-tracks | 4.4 | 0.750 | 0.931 | 3.8 | 6.1 | 0.951 | 0.001 | 1.000 |
| oracle-poses | 5.3 | 0.671 | 0.888 | 12.3 | 36.3 | 0.742 | 0.013 | 0.440 |
| oracle-all | 3.4 | 0.843 | 0.975 | 3.1 | 3.7 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 7.7 | 0.517 | 0.755 | 14.7 | 40.0 | 0.739 | 0.014 | 0.438 |
| no-track-loss | 7.0 | 0.546 | 0.783 | 15.6 | 38.1 | 0.739 | 0.014 | 0.438 |
| no-rigidity | 7.3 | 0.539 | 0.772 | 15.1 | 40.0 | 0.739 | 0.014 | 0.438 |
| no-depth-correction | 7.3 | 0.504 | 0.783 | 15.7 | 47.7 | 0.726 | 0.016 | 0.433 |
| depth-prior-ba | 7.5 | 0.470 | 0.740 | 14.2 | 39.8 | 0.741 | 0.013 | 0.439 |
| colmap | 8.9 | 0.460 | 0.722 | 14.8 | 42.9 | 0.677 | 0.021 | 0.430 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.2499 | 0.0108 | 0.0039 | 0.762 | 1.611 | 29.95 |
| static-only | 0.2499 | 0.0108 | 0.0042 | 1.086 | 1.611 | 26.32 |
| oracle-depth | 0.0001 | 0.0034 | 0.0038 | 0.777 | 1.611 | 30.35 |
| oracle-tracks | 0.2499 | 0.0086 | 0.0027 | 0.752 | 1.611 | 31.55 |
| oracle-poses | 0.2499 | 0.0106 | 0.0031 | 0.629 | 1.611 | 30.68 |
| oracle-all | 0.0001 | 0.0002 | 0.0021 | 0.662 | 1.611 | 32.45 |
| no-depth-loss | 0.2499 | 0.0108 | 0.0040 | 0.770 | 1.611 | 29.95 |
| no-track-loss | 0.2499 | 0.0108 | 0.0045 | 1.082 | 1.611 | 28.75 |
| no-rigidity | 0.2499 | 0.0108 | 0.0039 | 0.732 | 1.611 | 29.91 |
| no-depth-correction | 0.2499 | 0.0341 | 0.0049 | 0.764 | 1.611 | 29.48 |
| depth-prior-ba | 0.2499 | 0.0106 | 0.0038 | 0.687 | 1.611 | 30.35 |
| colmap | 0.2499 | 0.0113 | 0.0055 | 0.760 | 1.611 | 29.78 |

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
