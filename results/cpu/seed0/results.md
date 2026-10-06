# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.55 | 0.43 | 0.083 | 0.038 | 0.042 | 0.036 |
| rolling | 0.49 | 0.38 | 0.066 | 0.038 | 0.032 | 0.030 |
| sliding | 0.48 | 0.53 | 0.076 | 0.038 | 0.016 | 0.019 |
| squash | 0.74 | 0.55 | 0.072 | 0.041 | 0.025 | 0.024 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| still | 27.50 | 0.881 | 0.053 | 22.80 | 0.717 | 0.100 | - |
| rolling | 24.55 | 0.851 | 0.103 | 21.59 | 0.718 | 0.124 | 16.97 |
| sliding | 25.51 | 0.876 | 0.089 | 23.47 | 0.800 | 0.120 | 18.17 |
| squash | 26.32 | 0.901 | 0.062 | 22.32 | 0.769 | 0.100 | 18.46 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 11.7 | 0.381 | 0.651 | - | - | - | 0.003 | 0.492 |
| rolling | 8.9 | 0.483 | 0.711 | 12.5 | 28.6 | 0.718 | 0.013 | 0.303 |
| sliding | 4.9 | 0.692 | 0.916 | 9.3 | 27.9 | 0.779 | 0.013 | 0.441 |
| squash | 7.6 | 0.524 | 0.723 | 8.5 | 21.4 | 0.733 | 0.015 | 0.517 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.2259 | 0.0106 | 0.0023 | 0.271 | 1.465 | - |
| rolling | 0.2266 | 0.0112 | 0.0039 | 0.898 | 1.932 | 30.28 |
| sliding | 0.2265 | 0.0113 | 0.0036 | 0.942 | 1.289 | 30.40 |
| squash | 0.2263 | 0.0112 | 0.0037 | 0.875 | 1.395 | 29.66 |

## Variants, averaged over the scenes

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.57 | 0.47 | 0.074 | 0.039 | 0.029 | 0.027 |
| static-only | 0.57 | 0.47 | 0.074 | 0.039 | 0.029 | 0.040 |
| oracle-depth | 0.57 | 0.47 | 0.074 | 0.000 | 0.023 | 0.026 |
| oracle-tracks | 0.00 | 0.00 | 0.000 | 0.039 | 0.011 | 0.013 |
| oracle-poses | 0.00 | 0.00 | 0.000 | 0.039 | 0.015 | 0.018 |
| oracle-all | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.010 |
| no-depth-loss | 0.57 | 0.47 | 0.074 | 0.039 | 0.029 | 0.028 |
| no-track-loss | 0.57 | 0.47 | 0.074 | 0.039 | 0.029 | 0.028 |
| no-rigidity | 0.57 | 0.47 | 0.074 | 0.039 | 0.029 | 0.027 |
| no-depth-correction | 0.57 | 0.47 | 0.074 | 0.039 | 0.044 | 0.031 |
| depth-prior-ba | 0.84 | 0.65 | 0.099 | 0.039 | 0.042 | 0.048 |
| depth-anything | 0.57 | 0.47 | 0.074 | 0.023 | 0.032 | 0.030 |
| cotracker | 0.57 | 0.47 | 0.074 | 0.039 | 0.029 | 0.028 |
| raft | 0.57 | 0.47 | 0.074 | 0.039 | 0.029 | 0.030 |
| colmap | 2.93 | 1.63 | 0.246 | 0.039 | 0.075 | 0.080 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 25.97 | 0.877 | 0.077 | 22.54 | 0.751 | 0.111 | 17.87 |
| static-only | 23.15 | 0.827 | 0.141 | 20.14 | 0.701 | 0.179 | 11.20 |
| oracle-depth | 25.99 | 0.883 | 0.071 | 22.50 | 0.757 | 0.107 | 18.12 |
| oracle-tracks | 25.94 | 0.883 | 0.066 | 24.21 | 0.823 | 0.090 | 19.08 |
| oracle-poses | 25.94 | 0.879 | 0.073 | 23.63 | 0.809 | 0.098 | 17.61 |
| oracle-all | 26.57 | 0.897 | 0.056 | 24.58 | 0.836 | 0.078 | 21.01 |
| no-depth-loss | 25.95 | 0.876 | 0.077 | 22.45 | 0.749 | 0.114 | 17.54 |
| no-track-loss | 24.45 | 0.847 | 0.083 | 22.27 | 0.752 | 0.111 | 16.08 |
| no-rigidity | 25.80 | 0.875 | 0.077 | 22.51 | 0.750 | 0.112 | 18.01 |
| no-depth-correction | 25.40 | 0.863 | 0.087 | 21.98 | 0.720 | 0.122 | 16.57 |
| depth-prior-ba | 25.65 | 0.874 | 0.082 | 20.58 | 0.639 | 0.122 | 16.22 |
| depth-anything | 25.59 | 0.872 | 0.086 | 21.64 | 0.727 | 0.122 | 15.85 |
| cotracker | 25.55 | 0.878 | 0.073 | 22.44 | 0.751 | 0.105 | 18.19 |
| raft | 25.22 | 0.866 | 0.086 | 22.18 | 0.744 | 0.117 | 16.97 |
| colmap | 25.15 | 0.851 | 0.088 | 19.75 | 0.519 | 0.145 | 15.03 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8.3 | 0.520 | 0.750 | 10.1 | 26.0 | 0.743 | 0.011 | 0.438 |
| static-only | 8.6 | 0.501 | 0.737 | - | 73.9 | 0.000 | 0.000 | 0.488 |
| oracle-depth | 8.0 | 0.540 | 0.764 | 10.8 | 29.6 | 0.763 | 0.010 | 0.439 |
| oracle-tracks | 4.4 | 0.750 | 0.928 | 4.3 | 7.8 | 0.949 | 0.001 | 1.000 |
| oracle-poses | 5.2 | 0.684 | 0.894 | 9.1 | 24.4 | 0.745 | 0.011 | 0.438 |
| oracle-all | 3.4 | 0.844 | 0.976 | 2.6 | 2.2 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 8.5 | 0.510 | 0.738 | 10.4 | 26.5 | 0.743 | 0.011 | 0.438 |
| no-track-loss | 7.9 | 0.553 | 0.763 | 13.9 | 26.7 | 0.743 | 0.011 | 0.438 |
| no-rigidity | 8.2 | 0.520 | 0.751 | 10.7 | 29.2 | 0.743 | 0.011 | 0.438 |
| no-depth-correction | 7.9 | 0.459 | 0.761 | 11.4 | 32.4 | 0.672 | 0.023 | 0.425 |
| depth-prior-ba | 16.2 | 0.197 | 0.471 | 11.5 | 35.5 | 0.744 | 0.011 | 0.438 |
| depth-anything | 8.4 | 0.497 | 0.757 | 16.9 | 41.6 | 0.742 | 0.011 | 0.440 |
| cotracker | 8.2 | 0.511 | 0.758 | 7.4 | 13.0 | 0.743 | 0.011 | 0.841 |
| raft | 8.3 | 0.511 | 0.750 | 20.6 | 57.6 | 0.764 | 0.006 | 0.439 |
| colmap | 24.3 | 0.221 | 0.410 | 16.4 | 40.9 | 0.673 | 0.020 | 0.431 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.2263 | 0.0111 | 0.0034 | 0.746 | 1.520 | 30.11 |
| static-only | 0.2263 | 0.0111 | 0.0043 | 1.062 | 1.520 | 25.65 |
| oracle-depth | 0.0001 | 0.0033 | 0.0036 | 0.863 | 1.520 | 30.12 |
| oracle-tracks | 0.2263 | 0.0087 | 0.0026 | 0.866 | 1.520 | 31.05 |
| oracle-poses | 0.2263 | 0.0109 | 0.0032 | 0.785 | 1.520 | 29.81 |
| oracle-all | 0.0001 | 0.0002 | 0.0020 | 0.679 | 1.520 | 32.62 |
| no-depth-loss | 0.2263 | 0.0111 | 0.0035 | 0.752 | 1.520 | 29.86 |
| no-track-loss | 0.2263 | 0.0111 | 0.0042 | 1.184 | 1.520 | 28.59 |
| no-rigidity | 0.2263 | 0.0111 | 0.0036 | 0.796 | 1.520 | 30.03 |
| no-depth-correction | 0.2263 | 0.0314 | 0.0044 | 0.814 | 1.520 | 28.96 |
| depth-prior-ba | 0.2263 | 0.0112 | 0.0046 | 0.900 | 1.520 | 28.35 |
| depth-anything | 0.0107 | 0.0088 | 0.0042 | 0.827 | 1.520 | 27.78 |
| cotracker | 0.2263 | 0.0111 | 0.0039 | 0.924 | 1.520 | 29.96 |
| raft | 0.2263 | 0.0111 | 0.0044 | 0.904 | 1.520 | 29.10 |
| colmap | 0.2263 | 0.0132 | 0.0071 | 1.012 | 1.520 | 27.12 |

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
- `cotracker`: CoTracker3 as the dense tracker
- `raft`: RAFT optical flow instead of DIS
- `colmap`: camera poses from COLMAP (pycolmap) instead of the track-based SfM
