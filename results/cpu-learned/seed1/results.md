# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.47 | 0.37 | 0.052 | 0.020 | 0.016 | 0.016 |
| rolling | 0.43 | 0.44 | 0.068 | 0.020 | 0.026 | 0.026 |
| sliding | 0.67 | 0.58 | 0.093 | 0.025 | 0.047 | 0.041 |
| squash | 0.59 | 0.62 | 0.090 | 0.018 | 0.037 | 0.037 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| still | 26.29 | 0.879 | 24.66 | 0.827 | - |
| rolling | 25.04 | 0.860 | 22.37 | 0.757 | 16.64 |
| sliding | 24.98 | 0.862 | 20.12 | 0.650 | 16.59 |
| squash | 25.54 | 0.891 | 21.19 | 0.695 | 16.32 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 5.5 | 0.638 | 0.869 | - | - | - | 0.001 | 0.463 |
| rolling | 7.3 | 0.521 | 0.771 | 10.2 | 25.2 | 0.762 | 0.017 | 0.260 |
| sliding | 11.3 | 0.377 | 0.654 | 35.0 | 52.4 | 0.754 | 0.015 | 0.486 |
| squash | 10.7 | 0.395 | 0.646 | 8.8 | 17.5 | 0.728 | 0.012 | 0.443 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.0067 | 0.0045 | 0.0017 | 0.305 | 1.848 | - |
| rolling | 0.0054 | 0.0055 | 0.0034 | 0.954 | 1.586 | 29.94 |
| sliding | 0.0069 | 0.0077 | 0.0044 | 0.994 | 1.579 | 28.55 |
| squash | 0.0064 | 0.0065 | 0.0045 | 0.947 | 1.440 | 27.79 |

## Variants, averaged over the scenes

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.54 | 0.50 | 0.076 | 0.021 | 0.032 | 0.030 |
| static-only | 0.54 | 0.50 | 0.076 | 0.021 | 0.032 | 0.041 |
| oracle-depth | 0.54 | 0.50 | 0.076 | 0.000 | 0.027 | 0.028 |
| oracle-tracks | 0.00 | 0.00 | 0.000 | 0.021 | 0.011 | 0.015 |
| oracle-poses | 0.00 | 0.00 | 0.000 | 0.021 | 0.015 | 0.018 |
| oracle-all | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.010 |
| no-depth-loss | 0.54 | 0.50 | 0.076 | 0.021 | 0.032 | 0.030 |
| no-track-loss | 0.54 | 0.50 | 0.076 | 0.021 | 0.032 | 0.029 |
| no-rigidity | 0.54 | 0.50 | 0.076 | 0.021 | 0.032 | 0.030 |
| no-depth-correction | 0.54 | 0.50 | 0.076 | 0.021 | 0.034 | 0.030 |
| depth-prior-ba | 0.58 | 0.47 | 0.061 | 0.021 | 0.020 | 0.022 |
| colmap | 1.62 | 1.48 | 0.212 | 0.021 | 0.030 | 0.035 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| full | 25.46 | 0.873 | 22.09 | 0.732 | 16.52 |
| static-only | 22.76 | 0.831 | 20.08 | 0.693 | 10.67 |
| oracle-depth | 25.36 | 0.875 | 22.34 | 0.741 | 17.09 |
| oracle-tracks | 25.78 | 0.878 | 23.49 | 0.809 | 17.33 |
| oracle-poses | 25.80 | 0.879 | 23.19 | 0.807 | 17.15 |
| oracle-all | 26.94 | 0.901 | 24.67 | 0.840 | 21.19 |
| no-depth-loss | 25.29 | 0.871 | 21.99 | 0.732 | 16.40 |
| no-track-loss | 25.09 | 0.865 | 22.15 | 0.739 | 15.98 |
| no-rigidity | 25.31 | 0.872 | 22.10 | 0.732 | 16.62 |
| no-depth-correction | 25.14 | 0.864 | 21.81 | 0.722 | 16.23 |
| depth-prior-ba | 25.65 | 0.872 | 22.78 | 0.781 | 16.89 |
| colmap | 25.00 | 0.861 | 21.35 | 0.698 | 16.30 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8.7 | 0.483 | 0.735 | 18.0 | 31.7 | 0.748 | 0.011 | 0.413 |
| static-only | 8.8 | 0.472 | 0.728 | - | 74.3 | 0.000 | 0.000 | 0.472 |
| oracle-depth | 8.5 | 0.484 | 0.748 | 11.7 | 26.7 | 0.759 | 0.011 | 0.412 |
| oracle-tracks | 4.7 | 0.715 | 0.911 | 10.8 | 19.4 | 0.954 | 0.000 | 1.000 |
| oracle-poses | 5.5 | 0.650 | 0.875 | 12.0 | 28.2 | 0.753 | 0.011 | 0.413 |
| oracle-all | 3.4 | 0.847 | 0.977 | 2.6 | 2.3 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 8.6 | 0.488 | 0.732 | 18.5 | 32.3 | 0.748 | 0.011 | 0.413 |
| no-track-loss | 8.3 | 0.510 | 0.747 | 18.6 | 31.7 | 0.748 | 0.011 | 0.413 |
| no-rigidity | 8.6 | 0.483 | 0.736 | 18.7 | 32.6 | 0.748 | 0.011 | 0.413 |
| no-depth-correction | 8.2 | 0.461 | 0.745 | 19.3 | 34.1 | 0.737 | 0.013 | 0.413 |
| depth-prior-ba | 6.3 | 0.581 | 0.826 | 16.9 | 31.5 | 0.751 | 0.011 | 0.413 |
| colmap | 11.0 | 0.367 | 0.656 | 16.0 | 34.3 | 0.721 | 0.015 | 0.410 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.0064 | 0.0060 | 0.0035 | 0.800 | 1.613 | 28.76 |
| static-only | 0.0064 | 0.0060 | 0.0042 | 1.164 | 1.613 | 25.20 |
| oracle-depth | 0.0001 | 0.0035 | 0.0035 | 0.896 | 1.613 | 29.23 |
| oracle-tracks | 0.0064 | 0.0037 | 0.0029 | 0.930 | 1.613 | 29.40 |
| oracle-poses | 0.0064 | 0.0057 | 0.0028 | 0.727 | 1.613 | 29.17 |
| oracle-all | 0.0001 | 0.0002 | 0.0020 | 0.690 | 1.613 | 32.76 |
| no-depth-loss | 0.0064 | 0.0060 | 0.0036 | 0.828 | 1.613 | 28.59 |
| no-track-loss | 0.0064 | 0.0060 | 0.0038 | 0.977 | 1.613 | 28.31 |
| no-rigidity | 0.0064 | 0.0060 | 0.0036 | 0.816 | 1.613 | 28.82 |
| no-depth-correction | 0.0064 | 0.0060 | 0.0037 | 0.868 | 1.613 | 28.38 |
| depth-prior-ba | 0.0064 | 0.0058 | 0.0036 | 0.857 | 1.613 | 29.10 |
| colmap | 0.0064 | 0.0067 | 0.0051 | 1.001 | 1.613 | 28.42 |

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
