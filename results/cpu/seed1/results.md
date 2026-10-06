# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.47 | 0.37 | 0.052 | 0.032 | 0.017 | 0.016 |
| rolling | 0.43 | 0.44 | 0.068 | 0.034 | 0.027 | 0.025 |
| sliding | 0.67 | 0.58 | 0.093 | 0.034 | 0.046 | 0.040 |
| squash | 0.59 | 0.62 | 0.090 | 0.034 | 0.037 | 0.035 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| still | 26.20 | 0.880 | 0.074 | 24.60 | 0.823 | 0.089 | - |
| rolling | 25.27 | 0.866 | 0.094 | 22.33 | 0.761 | 0.124 | 16.67 |
| sliding | 24.60 | 0.860 | 0.098 | 20.39 | 0.653 | 0.127 | 17.61 |
| squash | 25.76 | 0.884 | 0.067 | 21.30 | 0.694 | 0.128 | 17.12 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 5.4 | 0.624 | 0.898 | - | - | - | 0.002 | 0.463 |
| rolling | 6.5 | 0.544 | 0.811 | 8.7 | 27.1 | 0.748 | 0.019 | 0.260 |
| sliding | 11.1 | 0.382 | 0.658 | 17.2 | 40.5 | 0.754 | 0.017 | 0.482 |
| squash | 10.4 | 0.390 | 0.648 | 9.0 | 21.0 | 0.714 | 0.012 | 0.439 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.2547 | 0.0108 | 0.0018 | 0.315 | 1.848 | - |
| rolling | 0.2560 | 0.0109 | 0.0057 | 1.047 | 1.586 | 29.99 |
| sliding | 0.2562 | 0.0134 | 0.0055 | 1.139 | 1.579 | 29.52 |
| squash | 0.2560 | 0.0121 | 0.0048 | 1.133 | 1.440 | 28.16 |

## Variants, averaged over the scenes

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.54 | 0.50 | 0.076 | 0.033 | 0.032 | 0.029 |
| static-only | 0.54 | 0.50 | 0.076 | 0.033 | 0.032 | 0.042 |
| oracle-depth | 0.54 | 0.50 | 0.076 | 0.000 | 0.027 | 0.028 |
| oracle-tracks | 0.00 | 0.00 | 0.000 | 0.033 | 0.011 | 0.013 |
| oracle-poses | 0.00 | 0.00 | 0.000 | 0.033 | 0.015 | 0.017 |
| oracle-all | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.010 |
| no-depth-loss | 0.54 | 0.50 | 0.076 | 0.033 | 0.032 | 0.029 |
| no-track-loss | 0.54 | 0.50 | 0.076 | 0.033 | 0.032 | 0.029 |
| no-rigidity | 0.54 | 0.50 | 0.076 | 0.033 | 0.032 | 0.029 |
| no-depth-correction | 0.54 | 0.50 | 0.076 | 0.033 | 0.043 | 0.030 |
| depth-prior-ba | 0.55 | 0.49 | 0.063 | 0.033 | 0.018 | 0.022 |
| depth-anything | 0.54 | 0.50 | 0.076 | 0.023 | 0.034 | 0.030 |
| cotracker | 0.54 | 0.50 | 0.076 | 0.033 | 0.032 | 0.029 |
| raft | 0.54 | 0.50 | 0.076 | 0.033 | 0.032 | 0.035 |
| colmap | 1.63 | 1.49 | 0.213 | 0.033 | 0.030 | 0.035 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 25.46 | 0.872 | 0.083 | 22.15 | 0.733 | 0.117 | 17.14 |
| static-only | 22.76 | 0.832 | 0.146 | 20.11 | 0.694 | 0.186 | 10.69 |
| oracle-depth | 25.55 | 0.878 | 0.076 | 22.32 | 0.741 | 0.114 | 17.04 |
| oracle-tracks | 25.84 | 0.886 | 0.065 | 24.02 | 0.821 | 0.091 | 19.07 |
| oracle-poses | 25.95 | 0.879 | 0.075 | 23.71 | 0.812 | 0.103 | 18.00 |
| oracle-all | 26.94 | 0.901 | 0.054 | 24.67 | 0.840 | 0.077 | 21.19 |
| no-depth-loss | 25.54 | 0.873 | 0.082 | 22.11 | 0.732 | 0.119 | 16.89 |
| no-track-loss | 24.45 | 0.855 | 0.087 | 22.16 | 0.735 | 0.117 | 16.09 |
| no-rigidity | 25.30 | 0.873 | 0.085 | 22.16 | 0.733 | 0.121 | 17.37 |
| no-depth-correction | 24.30 | 0.864 | 0.084 | 21.83 | 0.716 | 0.117 | 16.37 |
| depth-prior-ba | 25.49 | 0.873 | 0.084 | 23.15 | 0.793 | 0.113 | 17.14 |
| depth-anything | 25.79 | 0.874 | 0.081 | 21.71 | 0.721 | 0.122 | 16.22 |
| cotracker | 26.06 | 0.885 | 0.073 | 22.42 | 0.741 | 0.108 | 18.59 |
| raft | 24.13 | 0.862 | 0.088 | 21.63 | 0.726 | 0.124 | 16.33 |
| colmap | 25.25 | 0.864 | 0.084 | 21.49 | 0.705 | 0.119 | 16.61 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8.3 | 0.485 | 0.754 | 11.6 | 29.5 | 0.738 | 0.012 | 0.411 |
| static-only | 8.7 | 0.464 | 0.733 | - | 74.3 | 0.000 | 0.000 | 0.472 |
| oracle-depth | 8.5 | 0.486 | 0.749 | 11.6 | 26.9 | 0.759 | 0.011 | 0.412 |
| oracle-tracks | 4.4 | 0.755 | 0.929 | 3.9 | 6.3 | 0.951 | 0.001 | 1.000 |
| oracle-poses | 5.2 | 0.694 | 0.895 | 10.0 | 23.2 | 0.743 | 0.012 | 0.411 |
| oracle-all | 3.4 | 0.847 | 0.977 | 2.6 | 2.3 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 8.4 | 0.487 | 0.749 | 13.7 | 30.4 | 0.738 | 0.012 | 0.411 |
| no-track-loss | 8.0 | 0.508 | 0.769 | 15.4 | 29.2 | 0.738 | 0.012 | 0.411 |
| no-rigidity | 8.4 | 0.478 | 0.752 | 13.3 | 32.6 | 0.738 | 0.012 | 0.411 |
| no-depth-correction | 7.9 | 0.445 | 0.762 | 12.0 | 27.5 | 0.693 | 0.018 | 0.407 |
| depth-prior-ba | 6.6 | 0.555 | 0.804 | 12.0 | 28.6 | 0.741 | 0.012 | 0.411 |
| depth-anything | 8.7 | 0.458 | 0.742 | 15.1 | 31.0 | 0.752 | 0.013 | 0.412 |
| cotracker | 8.6 | 0.461 | 0.745 | 6.4 | 10.7 | 0.738 | 0.012 | 0.821 |
| raft | 8.5 | 0.466 | 0.740 | 16.9 | 38.9 | 0.796 | 0.006 | 0.413 |
| colmap | 10.9 | 0.368 | 0.654 | 13.3 | 27.5 | 0.710 | 0.016 | 0.408 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.2557 | 0.0118 | 0.0044 | 0.909 | 1.613 | 29.22 |
| static-only | 0.2557 | 0.0118 | 0.0043 | 1.146 | 1.613 | 25.20 |
| oracle-depth | 0.0001 | 0.0035 | 0.0035 | 0.828 | 1.613 | 29.15 |
| oracle-tracks | 0.2557 | 0.0092 | 0.0028 | 0.990 | 1.613 | 30.51 |
| oracle-poses | 0.2557 | 0.0116 | 0.0033 | 0.773 | 1.613 | 29.68 |
| oracle-all | 0.0001 | 0.0002 | 0.0020 | 0.690 | 1.613 | 32.76 |
| no-depth-loss | 0.2557 | 0.0118 | 0.0044 | 0.868 | 1.613 | 29.10 |
| no-track-loss | 0.2557 | 0.0118 | 0.0044 | 1.096 | 1.613 | 28.47 |
| no-rigidity | 0.2557 | 0.0118 | 0.0044 | 0.880 | 1.613 | 29.26 |
| no-depth-correction | 0.2557 | 0.0367 | 0.0045 | 0.894 | 1.613 | 28.56 |
| depth-prior-ba | 0.2557 | 0.0116 | 0.0041 | 0.865 | 1.613 | 29.22 |
| depth-anything | 0.0113 | 0.0099 | 0.0040 | 0.799 | 1.613 | 28.27 |
| cotracker | 0.2557 | 0.0118 | 0.0036 | 0.845 | 1.613 | 30.05 |
| raft | 0.2557 | 0.0118 | 0.0062 | 1.192 | 1.613 | 28.36 |
| colmap | 0.2557 | 0.0121 | 0.0052 | 0.925 | 1.613 | 28.66 |

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
