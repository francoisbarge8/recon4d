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

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| still | 26.29 | 0.879 | 0.077 | 24.66 | 0.827 | 0.093 | - |
| rolling | 25.04 | 0.860 | 0.088 | 22.37 | 0.757 | 0.124 | 16.64 |
| sliding | 24.70 | 0.859 | 0.104 | 20.20 | 0.650 | 0.131 | 16.78 |
| squash | 25.07 | 0.888 | 0.066 | 21.07 | 0.697 | 0.133 | 16.41 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 5.5 | 0.638 | 0.869 | - | - | - | 0.001 | 0.463 |
| rolling | 7.3 | 0.521 | 0.771 | 10.2 | 25.2 | 0.762 | 0.017 | 0.260 |
| sliding | 11.3 | 0.371 | 0.653 | 33.5 | 52.0 | 0.754 | 0.015 | 0.486 |
| squash | 10.7 | 0.392 | 0.644 | 9.0 | 18.0 | 0.728 | 0.012 | 0.443 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.0067 | 0.0045 | 0.0017 | 0.305 | 1.848 | - |
| rolling | 0.0054 | 0.0055 | 0.0034 | 0.954 | 1.586 | 29.94 |
| sliding | 0.0069 | 0.0077 | 0.0046 | 1.094 | 1.579 | 28.80 |
| squash | 0.0064 | 0.0065 | 0.0047 | 0.989 | 1.440 | 27.70 |

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
| depth-anything | 0.54 | 0.50 | 0.076 | 0.023 | 0.034 | 0.030 |
| raft | 0.54 | 0.50 | 0.076 | 0.021 | 0.032 | 0.032 |
| colmap | 1.63 | 1.50 | 0.214 | 0.021 | 0.030 | 0.035 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 25.28 | 0.872 | 0.084 | 22.08 | 0.733 | 0.120 | 16.61 |
| static-only | 22.75 | 0.832 | 0.143 | 20.08 | 0.694 | 0.182 | 10.65 |
| oracle-depth | 25.55 | 0.878 | 0.076 | 22.32 | 0.741 | 0.114 | 17.04 |
| oracle-tracks | 25.78 | 0.878 | 0.069 | 23.49 | 0.809 | 0.097 | 17.33 |
| oracle-poses | 25.80 | 0.879 | 0.076 | 23.19 | 0.807 | 0.102 | 17.15 |
| oracle-all | 26.94 | 0.901 | 0.054 | 24.67 | 0.840 | 0.077 | 21.19 |
| no-depth-loss | 25.27 | 0.871 | 0.081 | 21.93 | 0.731 | 0.121 | 16.33 |
| no-track-loss | 24.98 | 0.864 | 0.081 | 22.13 | 0.738 | 0.117 | 15.99 |
| no-rigidity | 25.30 | 0.872 | 0.085 | 22.10 | 0.732 | 0.122 | 16.60 |
| no-depth-correction | 25.09 | 0.864 | 0.088 | 21.78 | 0.721 | 0.123 | 16.14 |
| depth-prior-ba | 25.73 | 0.874 | 0.080 | 22.74 | 0.780 | 0.114 | 16.73 |
| depth-anything | 25.79 | 0.874 | 0.081 | 21.71 | 0.721 | 0.122 | 16.22 |
| raft | 25.19 | 0.865 | 0.086 | 21.75 | 0.725 | 0.121 | 16.02 |
| colmap | 25.12 | 0.863 | 0.084 | 21.34 | 0.699 | 0.121 | 16.14 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8.7 | 0.480 | 0.734 | 17.6 | 31.7 | 0.748 | 0.011 | 0.413 |
| static-only | 8.8 | 0.474 | 0.730 | - | 74.3 | 0.000 | 0.000 | 0.472 |
| oracle-depth | 8.5 | 0.486 | 0.749 | 11.6 | 26.9 | 0.759 | 0.011 | 0.412 |
| oracle-tracks | 4.7 | 0.715 | 0.911 | 10.8 | 19.4 | 0.954 | 0.000 | 1.000 |
| oracle-poses | 5.5 | 0.650 | 0.875 | 12.0 | 28.2 | 0.753 | 0.011 | 0.413 |
| oracle-all | 3.4 | 0.847 | 0.977 | 2.6 | 2.3 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 8.6 | 0.485 | 0.733 | 18.6 | 32.6 | 0.748 | 0.011 | 0.413 |
| no-track-loss | 8.3 | 0.509 | 0.748 | 18.6 | 31.5 | 0.748 | 0.011 | 0.413 |
| no-rigidity | 8.7 | 0.482 | 0.736 | 18.8 | 33.4 | 0.748 | 0.011 | 0.413 |
| no-depth-correction | 8.2 | 0.460 | 0.743 | 19.0 | 34.4 | 0.737 | 0.013 | 0.413 |
| depth-prior-ba | 6.3 | 0.582 | 0.826 | 16.5 | 31.4 | 0.751 | 0.011 | 0.413 |
| depth-anything | 8.7 | 0.458 | 0.742 | 15.1 | 31.0 | 0.752 | 0.013 | 0.412 |
| raft | 8.9 | 0.474 | 0.728 | 17.9 | 39.6 | 0.799 | 0.005 | 0.413 |
| colmap | 11.0 | 0.375 | 0.661 | 16.9 | 33.7 | 0.720 | 0.015 | 0.409 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.0064 | 0.0060 | 0.0036 | 0.836 | 1.613 | 28.81 |
| static-only | 0.0064 | 0.0060 | 0.0041 | 1.151 | 1.613 | 25.20 |
| oracle-depth | 0.0001 | 0.0035 | 0.0035 | 0.828 | 1.613 | 29.15 |
| oracle-tracks | 0.0064 | 0.0037 | 0.0029 | 0.930 | 1.613 | 29.40 |
| oracle-poses | 0.0064 | 0.0057 | 0.0028 | 0.727 | 1.613 | 29.17 |
| oracle-all | 0.0001 | 0.0002 | 0.0020 | 0.690 | 1.613 | 32.76 |
| no-depth-loss | 0.0064 | 0.0060 | 0.0037 | 0.849 | 1.613 | 28.49 |
| no-track-loss | 0.0064 | 0.0060 | 0.0037 | 0.920 | 1.613 | 28.30 |
| no-rigidity | 0.0064 | 0.0060 | 0.0036 | 0.842 | 1.613 | 28.84 |
| no-depth-correction | 0.0064 | 0.0060 | 0.0038 | 0.912 | 1.613 | 28.25 |
| depth-prior-ba | 0.0064 | 0.0058 | 0.0034 | 0.778 | 1.613 | 29.01 |
| depth-anything | 0.0113 | 0.0099 | 0.0040 | 0.799 | 1.613 | 28.27 |
| raft | 0.0064 | 0.0060 | 0.0041 | 1.056 | 1.613 | 28.20 |
| colmap | 0.0064 | 0.0067 | 0.0050 | 0.926 | 1.613 | 28.29 |

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
