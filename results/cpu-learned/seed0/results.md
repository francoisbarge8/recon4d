# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.55 | 0.43 | 0.083 | 0.016 | 0.039 | 0.035 |
| rolling | 0.49 | 0.38 | 0.066 | 0.018 | 0.034 | 0.031 |
| sliding | 0.48 | 0.53 | 0.076 | 0.025 | 0.017 | 0.020 |
| squash | 0.74 | 0.55 | 0.072 | 0.017 | 0.024 | 0.028 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| still | 27.48 | 0.881 | 0.053 | 22.80 | 0.725 | 0.098 | - |
| rolling | 24.61 | 0.851 | 0.101 | 21.22 | 0.714 | 0.136 | 15.92 |
| sliding | 25.01 | 0.859 | 0.085 | 22.70 | 0.780 | 0.124 | 17.01 |
| squash | 24.66 | 0.889 | 0.071 | 21.07 | 0.751 | 0.107 | 16.36 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 11.5 | 0.401 | 0.659 | - | - | - | 0.002 | 0.493 |
| rolling | 8.9 | 0.491 | 0.711 | 21.4 | 72.1 | 0.719 | 0.012 | 0.305 |
| sliding | 5.2 | 0.650 | 0.900 | 21.4 | 42.3 | 0.771 | 0.012 | 0.448 |
| squash | 7.3 | 0.554 | 0.757 | 10.4 | 16.0 | 0.747 | 0.014 | 0.520 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.0065 | 0.0055 | 0.0023 | 0.270 | 1.465 | - |
| rolling | 0.0051 | 0.0057 | 0.0041 | 0.934 | 1.932 | 29.42 |
| sliding | 0.0077 | 0.0068 | 0.0043 | 1.388 | 1.289 | 28.89 |
| squash | 0.0058 | 0.0062 | 0.0059 | 1.451 | 1.395 | 27.31 |

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
| no-track-loss | 0.57 | 0.47 | 0.074 | 0.019 | 0.029 | 0.029 |
| no-rigidity | 0.57 | 0.47 | 0.074 | 0.019 | 0.029 | 0.029 |
| no-depth-correction | 0.57 | 0.47 | 0.074 | 0.019 | 0.030 | 0.027 |
| depth-prior-ba | 0.53 | 0.49 | 0.063 | 0.019 | 0.019 | 0.022 |
| depth-anything | 0.57 | 0.47 | 0.074 | 0.023 | 0.032 | 0.030 |
| raft | 0.57 | 0.47 | 0.074 | 0.019 | 0.029 | 0.029 |
| colmap | 2.97 | 1.67 | 0.246 | 0.019 | 0.075 | 0.081 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 25.44 | 0.870 | 0.078 | 21.95 | 0.743 | 0.116 | 16.43 |
| static-only | 23.16 | 0.828 | 0.142 | 20.14 | 0.703 | 0.174 | 11.21 |
| oracle-depth | 25.99 | 0.883 | 0.071 | 22.50 | 0.757 | 0.107 | 18.12 |
| oracle-tracks | 25.58 | 0.875 | 0.069 | 23.33 | 0.805 | 0.096 | 16.58 |
| oracle-poses | 25.92 | 0.880 | 0.076 | 23.43 | 0.804 | 0.100 | 17.19 |
| oracle-all | 26.57 | 0.897 | 0.056 | 24.58 | 0.836 | 0.078 | 21.01 |
| no-depth-loss | 25.73 | 0.875 | 0.077 | 21.90 | 0.740 | 0.118 | 16.21 |
| no-track-loss | 24.95 | 0.858 | 0.077 | 21.95 | 0.745 | 0.116 | 15.71 |
| no-rigidity | 25.68 | 0.874 | 0.079 | 21.95 | 0.742 | 0.117 | 16.56 |
| no-depth-correction | 25.64 | 0.868 | 0.082 | 21.97 | 0.737 | 0.116 | 16.31 |
| depth-prior-ba | 25.76 | 0.872 | 0.080 | 22.66 | 0.778 | 0.111 | 16.73 |
| depth-anything | 25.59 | 0.872 | 0.086 | 21.64 | 0.727 | 0.122 | 15.85 |
| raft | 25.67 | 0.869 | 0.084 | 21.86 | 0.738 | 0.122 | 15.97 |
| colmap | 24.98 | 0.848 | 0.093 | 19.69 | 0.518 | 0.149 | 15.25 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8.2 | 0.524 | 0.757 | 17.7 | 43.4 | 0.745 | 0.010 | 0.441 |
| static-only | 8.6 | 0.502 | 0.740 | - | 73.9 | 0.000 | 0.000 | 0.488 |
| oracle-depth | 8.0 | 0.540 | 0.764 | 10.8 | 29.6 | 0.763 | 0.010 | 0.439 |
| oracle-tracks | 4.5 | 0.743 | 0.920 | 12.2 | 20.3 | 0.957 | 0.000 | 1.000 |
| oracle-poses | 5.4 | 0.645 | 0.883 | 16.0 | 39.8 | 0.746 | 0.010 | 0.440 |
| oracle-all | 3.4 | 0.844 | 0.976 | 2.6 | 2.2 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 8.2 | 0.521 | 0.753 | 17.9 | 42.0 | 0.745 | 0.010 | 0.441 |
| no-track-loss | 7.9 | 0.542 | 0.767 | 18.0 | 42.6 | 0.745 | 0.010 | 0.441 |
| no-rigidity | 8.2 | 0.522 | 0.758 | 19.1 | 47.3 | 0.745 | 0.010 | 0.441 |
| no-depth-correction | 7.3 | 0.532 | 0.795 | 19.0 | 41.9 | 0.739 | 0.010 | 0.440 |
| depth-prior-ba | 5.9 | 0.604 | 0.847 | 18.1 | 38.0 | 0.743 | 0.010 | 0.441 |
| depth-anything | 8.4 | 0.497 | 0.757 | 16.9 | 41.6 | 0.742 | 0.011 | 0.440 |
| raft | 8.2 | 0.518 | 0.755 | 21.9 | 38.5 | 0.753 | 0.005 | 0.442 |
| colmap | 24.5 | 0.224 | 0.413 | 17.2 | 42.2 | 0.676 | 0.019 | 0.433 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.0063 | 0.0060 | 0.0041 | 1.011 | 1.520 | 28.54 |
| static-only | 0.0063 | 0.0060 | 0.0041 | 1.054 | 1.520 | 25.65 |
| oracle-depth | 0.0001 | 0.0033 | 0.0036 | 0.863 | 1.520 | 30.12 |
| oracle-tracks | 0.0063 | 0.0037 | 0.0031 | 0.937 | 1.520 | 28.87 |
| oracle-poses | 0.0063 | 0.0058 | 0.0032 | 0.786 | 1.520 | 29.30 |
| oracle-all | 0.0001 | 0.0002 | 0.0020 | 0.679 | 1.520 | 32.62 |
| no-depth-loss | 0.0063 | 0.0060 | 0.0041 | 0.924 | 1.520 | 28.42 |
| no-track-loss | 0.0063 | 0.0060 | 0.0044 | 1.108 | 1.520 | 27.98 |
| no-rigidity | 0.0063 | 0.0060 | 0.0041 | 0.876 | 1.520 | 28.63 |
| no-depth-correction | 0.0063 | 0.0060 | 0.0037 | 0.906 | 1.520 | 28.68 |
| depth-prior-ba | 0.0063 | 0.0059 | 0.0038 | 0.861 | 1.520 | 28.91 |
| depth-anything | 0.0107 | 0.0088 | 0.0042 | 0.827 | 1.520 | 27.78 |
| raft | 0.0063 | 0.0060 | 0.0037 | 0.839 | 1.520 | 28.46 |
| colmap | 0.0063 | 0.0088 | 0.0070 | 0.930 | 1.520 | 26.99 |

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
- `depth-anything`: zero-shot Depth Anything V2 (small) instead of the default depth back-end
- `raft`: RAFT optical flow instead of DIS
- `colmap`: camera poses from COLMAP (pycolmap) instead of the track-based SfM
