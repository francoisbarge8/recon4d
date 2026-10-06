# Benchmark results (cpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.53 | 0.39 | 0.059 | 0.034 | 0.018 | 0.018 |
| rolling | 0.50 | 0.43 | 0.057 | 0.033 | 0.014 | 0.017 |
| sliding | 0.66 | 0.56 | 0.088 | 0.035 | 0.036 | 0.034 |
| squash | 1.00 | 0.94 | 0.119 | 0.034 | 0.023 | 0.030 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| still | 26.25 | 0.872 | 0.080 | 22.89 | 0.785 | 0.090 | - |
| rolling | 25.24 | 0.855 | 0.071 | 23.17 | 0.788 | 0.109 | 18.79 |
| sliding | 25.44 | 0.874 | 0.082 | 21.33 | 0.693 | 0.117 | 18.04 |
| squash | 26.41 | 0.886 | 0.069 | 23.12 | 0.776 | 0.107 | 18.36 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 5.6 | 0.614 | 0.869 | - | - | - | 0.004 | 0.454 |
| rolling | 4.4 | 0.755 | 0.930 | 15.8 | 52.9 | 0.712 | 0.021 | 0.283 |
| sliding | 9.8 | 0.405 | 0.650 | 21.6 | 56.5 | 0.773 | 0.015 | 0.508 |
| squash | 9.3 | 0.375 | 0.637 | 6.2 | 11.0 | 0.730 | 0.014 | 0.507 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.2496 | 0.0103 | 0.0021 | 0.273 | 1.786 | - |
| rolling | 0.2495 | 0.0104 | 0.0042 | 0.802 | 1.782 | 30.44 |
| sliding | 0.2501 | 0.0111 | 0.0044 | 0.974 | 1.587 | 29.85 |
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
| depth-anything | 0.67 | 0.58 | 0.081 | 0.022 | 0.025 | 0.026 |
| cotracker | 0.67 | 0.58 | 0.081 | 0.034 | 0.023 | 0.022 |
| raft | 0.67 | 0.58 | 0.081 | 0.034 | 0.023 | 0.026 |
| colmap | 1.53 | 1.48 | 0.193 | 0.034 | 0.025 | 0.031 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 25.83 | 0.872 | 0.076 | 22.63 | 0.761 | 0.106 | 18.40 |
| static-only | 23.26 | 0.832 | 0.129 | 20.56 | 0.723 | 0.161 | 11.77 |
| oracle-depth | 26.23 | 0.876 | 0.072 | 22.67 | 0.764 | 0.109 | 18.39 |
| oracle-tracks | 25.95 | 0.881 | 0.062 | 24.25 | 0.821 | 0.091 | 19.80 |
| oracle-poses | 26.10 | 0.876 | 0.071 | 23.90 | 0.812 | 0.096 | 18.83 |
| oracle-all | 26.62 | 0.893 | 0.054 | 24.45 | 0.833 | 0.083 | 20.90 |
| no-depth-loss | 25.85 | 0.871 | 0.075 | 22.57 | 0.757 | 0.112 | 18.37 |
| no-track-loss | 24.68 | 0.849 | 0.079 | 22.34 | 0.754 | 0.110 | 16.89 |
| no-rigidity | 25.68 | 0.870 | 0.076 | 22.61 | 0.761 | 0.111 | 18.50 |
| no-depth-correction | 25.39 | 0.863 | 0.085 | 22.18 | 0.739 | 0.116 | 17.64 |
| depth-prior-ba | 25.99 | 0.875 | 0.075 | 23.39 | 0.791 | 0.105 | 18.48 |
| depth-anything | 25.94 | 0.873 | 0.077 | 22.09 | 0.746 | 0.119 | 17.12 |
| cotracker | 25.76 | 0.876 | 0.069 | 22.73 | 0.764 | 0.102 | 18.59 |
| raft | 25.73 | 0.870 | 0.073 | 22.50 | 0.757 | 0.110 | 17.26 |
| colmap | 25.60 | 0.863 | 0.082 | 22.45 | 0.735 | 0.113 | 17.86 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 7.3 | 0.537 | 0.771 | 14.5 | 40.1 | 0.739 | 0.014 | 0.438 |
| static-only | 7.5 | 0.527 | 0.761 | - | 74.2 | 0.000 | 0.000 | 0.488 |
| oracle-depth | 6.7 | 0.563 | 0.790 | 14.0 | 29.2 | 0.758 | 0.012 | 0.440 |
| oracle-tracks | 4.4 | 0.750 | 0.931 | 3.8 | 6.1 | 0.951 | 0.001 | 1.000 |
| oracle-poses | 5.3 | 0.671 | 0.888 | 12.3 | 36.3 | 0.742 | 0.013 | 0.440 |
| oracle-all | 3.4 | 0.843 | 0.975 | 3.1 | 3.7 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 7.7 | 0.515 | 0.754 | 14.5 | 40.1 | 0.739 | 0.014 | 0.438 |
| no-track-loss | 7.0 | 0.547 | 0.782 | 15.4 | 38.1 | 0.739 | 0.014 | 0.438 |
| no-rigidity | 7.3 | 0.539 | 0.772 | 15.3 | 39.8 | 0.739 | 0.014 | 0.438 |
| no-depth-correction | 7.4 | 0.504 | 0.781 | 15.9 | 47.8 | 0.726 | 0.016 | 0.433 |
| depth-prior-ba | 7.5 | 0.469 | 0.738 | 13.4 | 39.9 | 0.741 | 0.013 | 0.439 |
| depth-anything | 7.3 | 0.529 | 0.774 | 14.1 | 32.3 | 0.726 | 0.016 | 0.441 |
| cotracker | 7.0 | 0.553 | 0.785 | 5.4 | 9.5 | 0.739 | 0.014 | 0.835 |
| raft | 7.3 | 0.528 | 0.772 | 16.3 | 41.2 | 0.770 | 0.006 | 0.441 |
| colmap | 9.1 | 0.401 | 0.686 | 14.6 | 42.9 | 0.678 | 0.021 | 0.430 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.2499 | 0.0108 | 0.0038 | 0.710 | 1.611 | 30.02 |
| static-only | 0.2499 | 0.0108 | 0.0042 | 1.095 | 1.611 | 26.32 |
| oracle-depth | 0.0001 | 0.0034 | 0.0038 | 0.743 | 1.611 | 30.33 |
| oracle-tracks | 0.2499 | 0.0086 | 0.0027 | 0.752 | 1.611 | 31.55 |
| oracle-poses | 0.2499 | 0.0106 | 0.0031 | 0.629 | 1.611 | 30.68 |
| oracle-all | 0.0001 | 0.0002 | 0.0021 | 0.662 | 1.611 | 32.45 |
| no-depth-loss | 0.2499 | 0.0108 | 0.0041 | 0.784 | 1.611 | 30.06 |
| no-track-loss | 0.2499 | 0.0108 | 0.0046 | 1.150 | 1.611 | 28.83 |
| no-rigidity | 0.2499 | 0.0108 | 0.0039 | 0.737 | 1.611 | 29.99 |
| no-depth-correction | 0.2499 | 0.0341 | 0.0049 | 0.775 | 1.611 | 29.51 |
| depth-prior-ba | 0.2499 | 0.0106 | 0.0038 | 0.689 | 1.611 | 30.29 |
| depth-anything | 0.0097 | 0.0087 | 0.0042 | 0.741 | 1.611 | 28.72 |
| cotracker | 0.2499 | 0.0108 | 0.0034 | 0.716 | 1.611 | 30.51 |
| raft | 0.2499 | 0.0108 | 0.0042 | 0.793 | 1.611 | 29.90 |
| colmap | 0.2499 | 0.0113 | 0.0054 | 0.828 | 1.611 | 29.47 |

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
