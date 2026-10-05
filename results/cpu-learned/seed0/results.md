# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.55 | 0.43 | 0.083 | 0.016 | 0.039 | 0.035 |
| rolling | 0.49 | 0.38 | 0.066 | 0.018 | 0.034 | 0.032 |
| sliding | 0.48 | 0.53 | 0.076 | 0.025 | 0.017 | 0.020 |
| squash | 0.74 | 0.55 | 0.072 | 0.017 | 0.024 | 0.029 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| still | 27.48 | 0.881 | 22.80 | 0.725 | - |
| rolling | 24.56 | 0.853 | 21.06 | 0.707 | 15.94 |
| sliding | 25.01 | 0.859 | 22.70 | 0.780 | 17.01 |
| squash | 24.81 | 0.889 | 21.06 | 0.750 | 16.29 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 11.5 | 0.401 | 0.659 | - | - | - | 0.002 | 0.493 |
| rolling | 8.9 | 0.486 | 0.712 | 20.5 | 81.6 | 0.719 | 0.012 | 0.305 |
| sliding | 5.2 | 0.650 | 0.900 | 21.4 | 42.3 | 0.771 | 0.012 | 0.448 |
| squash | 7.3 | 0.551 | 0.756 | 9.9 | 15.4 | 0.747 | 0.014 | 0.520 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.0065 | 0.0055 | 0.0023 | 0.270 | 1.465 | - |
| rolling | 0.0051 | 0.0057 | 0.0037 | 0.847 | 1.932 | 29.31 |
| sliding | 0.0077 | 0.0068 | 0.0043 | 1.388 | 1.289 | 28.89 |
| squash | 0.0058 | 0.0062 | 0.0056 | 1.210 | 1.395 | 27.35 |

## Variants, averaged over the scenes

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.57 | 0.47 | 0.074 | 0.019 | 0.029 | 0.029 |
| static-only | 0.57 | 0.47 | 0.074 | 0.019 | 0.029 | 0.040 |
| oracle-depth | 0.57 | 0.47 | 0.074 | 0.000 | 0.023 | 0.026 |
| oracle-tracks | 0.00 | 0.00 | 0.000 | 0.019 | 0.011 | 0.015 |
| oracle-poses | 0.00 | 0.00 | 0.000 | 0.019 | 0.015 | 0.019 |
| oracle-all | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.010 |
| no-depth-loss | 0.57 | 0.47 | 0.074 | 0.019 | 0.029 | 0.029 |
| no-track-loss | 0.57 | 0.47 | 0.074 | 0.019 | 0.029 | 0.028 |
| no-rigidity | 0.57 | 0.47 | 0.074 | 0.019 | 0.029 | 0.029 |
| no-depth-correction | 0.57 | 0.47 | 0.074 | 0.019 | 0.030 | 0.027 |
| depth-prior-ba | 0.53 | 0.49 | 0.063 | 0.019 | 0.019 | 0.022 |
| colmap | 3.03 | 1.68 | 0.252 | 0.019 | 0.077 | 0.082 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| full | 25.47 | 0.871 | 21.91 | 0.740 | 16.41 |
| static-only | 23.13 | 0.827 | 20.13 | 0.702 | 11.20 |
| oracle-depth | 26.08 | 0.883 | 22.57 | 0.759 | 18.24 |
| oracle-tracks | 25.58 | 0.875 | 23.33 | 0.805 | 16.58 |
| oracle-poses | 25.92 | 0.880 | 23.43 | 0.804 | 17.19 |
| oracle-all | 26.57 | 0.897 | 24.58 | 0.836 | 21.01 |
| no-depth-loss | 25.68 | 0.875 | 21.90 | 0.741 | 16.25 |
| no-track-loss | 24.96 | 0.859 | 21.93 | 0.744 | 15.74 |
| no-rigidity | 25.65 | 0.875 | 21.95 | 0.743 | 16.57 |
| no-depth-correction | 25.71 | 0.868 | 22.03 | 0.739 | 16.42 |
| depth-prior-ba | 25.83 | 0.873 | 22.65 | 0.777 | 16.56 |
| colmap | 24.99 | 0.846 | 19.64 | 0.513 | 15.20 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8.2 | 0.522 | 0.757 | 17.3 | 46.4 | 0.745 | 0.010 | 0.441 |
| static-only | 8.6 | 0.501 | 0.740 | - | 73.9 | 0.000 | 0.000 | 0.488 |
| oracle-depth | 8.1 | 0.542 | 0.764 | 10.6 | 29.4 | 0.763 | 0.010 | 0.439 |
| oracle-tracks | 4.5 | 0.743 | 0.920 | 12.2 | 20.3 | 0.957 | 0.000 | 1.000 |
| oracle-poses | 5.4 | 0.645 | 0.883 | 16.0 | 39.8 | 0.746 | 0.010 | 0.440 |
| oracle-all | 3.4 | 0.844 | 0.976 | 2.6 | 2.2 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 8.2 | 0.519 | 0.752 | 18.2 | 41.6 | 0.745 | 0.010 | 0.441 |
| no-track-loss | 7.9 | 0.544 | 0.767 | 18.2 | 47.3 | 0.745 | 0.010 | 0.441 |
| no-rigidity | 8.2 | 0.521 | 0.758 | 18.9 | 45.5 | 0.745 | 0.010 | 0.441 |
| no-depth-correction | 7.3 | 0.532 | 0.796 | 18.2 | 42.6 | 0.739 | 0.010 | 0.440 |
| depth-prior-ba | 6.0 | 0.602 | 0.846 | 17.8 | 38.4 | 0.743 | 0.010 | 0.441 |
| colmap | 24.9 | 0.228 | 0.416 | 17.1 | 40.3 | 0.676 | 0.019 | 0.433 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.0063 | 0.0060 | 0.0040 | 0.929 | 1.520 | 28.52 |
| static-only | 0.0063 | 0.0060 | 0.0042 | 1.051 | 1.520 | 25.65 |
| oracle-depth | 0.0001 | 0.0033 | 0.0036 | 0.843 | 1.520 | 30.22 |
| oracle-tracks | 0.0063 | 0.0037 | 0.0031 | 0.937 | 1.520 | 28.87 |
| oracle-poses | 0.0063 | 0.0058 | 0.0032 | 0.786 | 1.520 | 29.30 |
| oracle-all | 0.0001 | 0.0002 | 0.0020 | 0.679 | 1.520 | 32.62 |
| no-depth-loss | 0.0063 | 0.0060 | 0.0040 | 0.879 | 1.520 | 28.42 |
| no-track-loss | 0.0063 | 0.0060 | 0.0044 | 1.108 | 1.520 | 27.98 |
| no-rigidity | 0.0063 | 0.0060 | 0.0041 | 0.981 | 1.520 | 28.69 |
| no-depth-correction | 0.0063 | 0.0060 | 0.0037 | 0.880 | 1.520 | 28.89 |
| depth-prior-ba | 0.0063 | 0.0059 | 0.0038 | 0.824 | 1.520 | 28.79 |
| colmap | 0.0063 | 0.0088 | 0.0069 | 0.926 | 1.520 | 26.99 |

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
