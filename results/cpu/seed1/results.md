# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.47 | 0.37 | 0.052 | 0.032 | 0.017 | 0.016 |
| rolling | 0.43 | 0.44 | 0.068 | 0.034 | 0.027 | 0.025 |
| sliding | 0.67 | 0.58 | 0.093 | 0.034 | 0.046 | 0.040 |
| squash | 0.59 | 0.62 | 0.090 | 0.034 | 0.037 | 0.036 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| still | 26.20 | 0.880 | 24.60 | 0.823 | - |
| rolling | 25.27 | 0.866 | 22.33 | 0.761 | 16.67 |
| sliding | 24.65 | 0.856 | 20.40 | 0.652 | 17.67 |
| squash | 25.91 | 0.889 | 21.39 | 0.698 | 17.00 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 5.4 | 0.624 | 0.898 | - | - | - | 0.002 | 0.463 |
| rolling | 6.5 | 0.544 | 0.811 | 8.7 | 27.1 | 0.748 | 0.019 | 0.260 |
| sliding | 11.3 | 0.357 | 0.653 | 22.5 | 39.7 | 0.754 | 0.017 | 0.482 |
| squash | 10.3 | 0.385 | 0.652 | 8.7 | 19.9 | 0.714 | 0.012 | 0.439 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.2547 | 0.0108 | 0.0018 | 0.315 | 1.848 | - |
| rolling | 0.2560 | 0.0109 | 0.0057 | 1.047 | 1.586 | 29.99 |
| sliding | 0.2562 | 0.0134 | 0.0052 | 1.146 | 1.579 | 29.59 |
| squash | 0.2560 | 0.0121 | 0.0045 | 0.896 | 1.440 | 28.15 |

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
| colmap | 1.56 | 1.43 | 0.204 | 0.033 | 0.030 | 0.035 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| full | 25.51 | 0.873 | 22.18 | 0.733 | 17.11 |
| static-only | 22.75 | 0.831 | 20.09 | 0.693 | 10.71 |
| oracle-depth | 25.36 | 0.875 | 22.34 | 0.741 | 17.09 |
| oracle-tracks | 25.84 | 0.886 | 24.02 | 0.821 | 19.07 |
| oracle-poses | 25.95 | 0.879 | 23.71 | 0.812 | 18.00 |
| oracle-all | 26.94 | 0.901 | 24.67 | 0.840 | 21.19 |
| no-depth-loss | 25.51 | 0.873 | 22.10 | 0.733 | 16.77 |
| no-track-loss | 24.31 | 0.852 | 22.09 | 0.733 | 16.12 |
| no-rigidity | 25.21 | 0.870 | 22.15 | 0.732 | 17.27 |
| no-depth-correction | 24.21 | 0.861 | 21.79 | 0.714 | 16.35 |
| depth-prior-ba | 25.52 | 0.874 | 23.20 | 0.794 | 17.29 |
| colmap | 25.23 | 0.865 | 21.57 | 0.707 | 16.56 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8.4 | 0.477 | 0.753 | 13.3 | 28.9 | 0.738 | 0.012 | 0.411 |
| static-only | 8.7 | 0.463 | 0.733 | - | 74.3 | 0.000 | 0.000 | 0.472 |
| oracle-depth | 8.5 | 0.484 | 0.748 | 11.7 | 26.7 | 0.759 | 0.011 | 0.412 |
| oracle-tracks | 4.4 | 0.755 | 0.929 | 3.9 | 6.3 | 0.951 | 0.001 | 1.000 |
| oracle-poses | 5.2 | 0.694 | 0.895 | 10.0 | 23.2 | 0.743 | 0.012 | 0.411 |
| oracle-all | 3.4 | 0.847 | 0.977 | 2.6 | 2.3 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 8.3 | 0.489 | 0.752 | 13.8 | 30.5 | 0.738 | 0.012 | 0.411 |
| no-track-loss | 7.9 | 0.510 | 0.768 | 13.5 | 29.1 | 0.738 | 0.012 | 0.411 |
| no-rigidity | 8.3 | 0.481 | 0.754 | 13.4 | 32.6 | 0.738 | 0.012 | 0.411 |
| no-depth-correction | 7.9 | 0.446 | 0.762 | 12.3 | 27.6 | 0.693 | 0.018 | 0.407 |
| depth-prior-ba | 6.6 | 0.553 | 0.803 | 12.2 | 28.8 | 0.741 | 0.012 | 0.411 |
| colmap | 10.9 | 0.365 | 0.653 | 12.5 | 25.8 | 0.712 | 0.016 | 0.408 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.2557 | 0.0118 | 0.0043 | 0.851 | 1.613 | 29.24 |
| static-only | 0.2557 | 0.0118 | 0.0043 | 1.138 | 1.613 | 25.20 |
| oracle-depth | 0.0001 | 0.0035 | 0.0035 | 0.896 | 1.613 | 29.23 |
| oracle-tracks | 0.2557 | 0.0092 | 0.0028 | 0.990 | 1.613 | 30.51 |
| oracle-poses | 0.2557 | 0.0116 | 0.0033 | 0.773 | 1.613 | 29.68 |
| oracle-all | 0.0001 | 0.0002 | 0.0020 | 0.690 | 1.613 | 32.76 |
| no-depth-loss | 0.2557 | 0.0118 | 0.0044 | 0.901 | 1.613 | 28.94 |
| no-track-loss | 0.2557 | 0.0118 | 0.0045 | 1.150 | 1.613 | 28.30 |
| no-rigidity | 0.2557 | 0.0118 | 0.0044 | 0.914 | 1.613 | 29.19 |
| no-depth-correction | 0.2557 | 0.0367 | 0.0045 | 0.911 | 1.613 | 28.54 |
| depth-prior-ba | 0.2557 | 0.0116 | 0.0041 | 0.823 | 1.613 | 29.41 |
| colmap | 0.2557 | 0.0120 | 0.0051 | 0.921 | 1.613 | 28.73 |

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
