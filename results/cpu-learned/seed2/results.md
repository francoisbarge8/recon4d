# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.53 | 0.39 | 0.059 | 0.015 | 0.017 | 0.018 |
| rolling | 0.50 | 0.43 | 0.057 | 0.018 | 0.013 | 0.017 |
| sliding | 0.66 | 0.56 | 0.088 | 0.034 | 0.040 | 0.037 |
| squash | 1.00 | 0.94 | 0.119 | 0.024 | 0.025 | 0.034 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| still | 26.16 | 0.871 | 0.076 | 22.69 | 0.780 | 0.100 | - |
| rolling | 25.64 | 0.865 | 0.080 | 23.14 | 0.789 | 0.119 | 17.66 |
| sliding | 24.81 | 0.866 | 0.086 | 20.61 | 0.675 | 0.120 | 15.75 |
| squash | 26.74 | 0.882 | 0.070 | 22.42 | 0.755 | 0.117 | 16.63 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 5.7 | 0.601 | 0.855 | - | - | - | 0.004 | 0.455 |
| rolling | 4.5 | 0.722 | 0.926 | 27.9 | 55.4 | 0.722 | 0.020 | 0.284 |
| sliding | 10.3 | 0.418 | 0.651 | 33.7 | 61.6 | 0.764 | 0.013 | 0.515 |
| squash | 9.5 | 0.371 | 0.668 | 10.9 | 12.4 | 0.732 | 0.013 | 0.511 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.0045 | 0.0046 | 0.0021 | 0.280 | 1.786 | - |
| rolling | 0.0071 | 0.0054 | 0.0043 | 0.854 | 1.782 | 30.66 |
| sliding | 0.0073 | 0.0069 | 0.0048 | 1.223 | 1.587 | 27.55 |
| squash | 0.0087 | 0.0072 | 0.0048 | 0.792 | 1.289 | 28.71 |

## Variants, averaged over the scenes

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.026 |
| static-only | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.038 |
| oracle-depth | 0.67 | 0.58 | 0.081 | 0.000 | 0.017 | 0.023 |
| oracle-tracks | 0.00 | 0.00 | 0.000 | 0.023 | 0.012 | 0.015 |
| oracle-poses | 0.00 | 0.00 | 0.000 | 0.023 | 0.016 | 0.022 |
| oracle-all | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.010 |
| no-depth-loss | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.028 |
| no-track-loss | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.029 |
| no-rigidity | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.027 |
| no-depth-correction | 0.67 | 0.58 | 0.081 | 0.023 | 0.025 | 0.027 |
| depth-prior-ba | 0.58 | 0.47 | 0.067 | 0.023 | 0.023 | 0.023 |
| depth-anything | 0.67 | 0.58 | 0.081 | 0.022 | 0.025 | 0.026 |
| raft | 0.67 | 0.58 | 0.081 | 0.023 | 0.024 | 0.027 |
| colmap | 1.55 | 1.48 | 0.192 | 0.023 | 0.026 | 0.031 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 25.84 | 0.871 | 0.078 | 22.21 | 0.750 | 0.114 | 16.68 |
| static-only | 23.24 | 0.829 | 0.130 | 20.48 | 0.718 | 0.168 | 11.74 |
| oracle-depth | 26.23 | 0.876 | 0.072 | 22.67 | 0.764 | 0.109 | 18.39 |
| oracle-tracks | 26.29 | 0.882 | 0.065 | 23.51 | 0.804 | 0.098 | 17.08 |
| oracle-poses | 25.93 | 0.874 | 0.077 | 23.13 | 0.794 | 0.109 | 16.77 |
| oracle-all | 26.62 | 0.893 | 0.054 | 24.45 | 0.833 | 0.083 | 20.90 |
| no-depth-loss | 25.75 | 0.868 | 0.081 | 22.15 | 0.748 | 0.120 | 16.83 |
| no-track-loss | 24.22 | 0.841 | 0.083 | 21.84 | 0.744 | 0.119 | 15.64 |
| no-rigidity | 25.72 | 0.869 | 0.081 | 22.22 | 0.749 | 0.116 | 16.90 |
| no-depth-correction | 25.51 | 0.867 | 0.079 | 22.17 | 0.746 | 0.114 | 16.95 |
| depth-prior-ba | 25.75 | 0.870 | 0.079 | 22.23 | 0.759 | 0.115 | 16.77 |
| depth-anything | 25.94 | 0.873 | 0.077 | 22.09 | 0.746 | 0.119 | 17.12 |
| raft | 25.56 | 0.866 | 0.075 | 22.13 | 0.750 | 0.116 | 16.32 |
| colmap | 25.48 | 0.861 | 0.085 | 22.15 | 0.730 | 0.119 | 16.43 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 7.5 | 0.528 | 0.775 | 24.2 | 43.1 | 0.739 | 0.013 | 0.441 |
| static-only | 7.6 | 0.518 | 0.769 | - | 74.2 | 0.000 | 0.000 | 0.488 |
| oracle-depth | 6.7 | 0.563 | 0.790 | 14.0 | 29.2 | 0.758 | 0.012 | 0.440 |
| oracle-tracks | 4.6 | 0.737 | 0.915 | 13.9 | 24.4 | 0.958 | 0.000 | 1.000 |
| oracle-poses | 5.6 | 0.643 | 0.873 | 18.2 | 34.9 | 0.740 | 0.013 | 0.443 |
| oracle-all | 3.4 | 0.843 | 0.975 | 3.1 | 3.7 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 7.8 | 0.513 | 0.760 | 23.9 | 43.1 | 0.739 | 0.013 | 0.441 |
| no-track-loss | 7.3 | 0.539 | 0.786 | 25.9 | 43.2 | 0.739 | 0.013 | 0.441 |
| no-rigidity | 7.4 | 0.533 | 0.777 | 24.5 | 41.5 | 0.739 | 0.013 | 0.441 |
| no-depth-correction | 7.4 | 0.508 | 0.795 | 21.9 | 36.7 | 0.734 | 0.013 | 0.442 |
| depth-prior-ba | 6.5 | 0.571 | 0.797 | 20.0 | 40.4 | 0.741 | 0.012 | 0.441 |
| depth-anything | 7.3 | 0.529 | 0.774 | 14.1 | 32.3 | 0.726 | 0.016 | 0.441 |
| raft | 7.4 | 0.524 | 0.778 | 21.8 | 47.2 | 0.747 | 0.006 | 0.446 |
| colmap | 8.8 | 0.449 | 0.728 | 19.3 | 37.6 | 0.676 | 0.021 | 0.433 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.0069 | 0.0060 | 0.0040 | 0.787 | 1.611 | 28.97 |
| static-only | 0.0069 | 0.0060 | 0.0042 | 1.103 | 1.611 | 26.32 |
| oracle-depth | 0.0001 | 0.0034 | 0.0038 | 0.743 | 1.611 | 30.33 |
| oracle-tracks | 0.0069 | 0.0039 | 0.0028 | 0.690 | 1.611 | 29.44 |
| oracle-poses | 0.0069 | 0.0058 | 0.0036 | 0.705 | 1.611 | 28.91 |
| oracle-all | 0.0001 | 0.0002 | 0.0021 | 0.662 | 1.611 | 32.45 |
| no-depth-loss | 0.0069 | 0.0060 | 0.0040 | 0.706 | 1.611 | 28.97 |
| no-track-loss | 0.0069 | 0.0060 | 0.0048 | 1.199 | 1.611 | 27.78 |
| no-rigidity | 0.0069 | 0.0060 | 0.0040 | 0.786 | 1.611 | 29.05 |
| no-depth-correction | 0.0069 | 0.0065 | 0.0041 | 0.794 | 1.611 | 29.11 |
| depth-prior-ba | 0.0069 | 0.0060 | 0.0038 | 0.770 | 1.611 | 28.93 |
| depth-anything | 0.0097 | 0.0087 | 0.0042 | 0.741 | 1.611 | 28.72 |
| raft | 0.0069 | 0.0060 | 0.0037 | 0.804 | 1.611 | 28.91 |
| colmap | 0.0069 | 0.0067 | 0.0049 | 0.793 | 1.611 | 28.50 |

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
