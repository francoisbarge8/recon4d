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

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| still | 27.50 | 0.881 | 22.80 | 0.717 | - |
| rolling | 24.63 | 0.851 | 21.65 | 0.718 | 17.12 |
| sliding | 25.51 | 0.876 | 23.47 | 0.800 | 18.17 |
| squash | 25.98 | 0.898 | 22.05 | 0.765 | 18.26 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 11.7 | 0.381 | 0.651 | - | - | - | 0.003 | 0.492 |
| rolling | 9.0 | 0.475 | 0.706 | 12.2 | 27.7 | 0.718 | 0.013 | 0.303 |
| sliding | 4.9 | 0.692 | 0.916 | 9.3 | 27.9 | 0.779 | 0.013 | 0.441 |
| squash | 7.6 | 0.528 | 0.722 | 8.4 | 21.3 | 0.733 | 0.015 | 0.517 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.2259 | 0.0106 | 0.0023 | 0.271 | 1.465 | - |
| rolling | 0.2266 | 0.0112 | 0.0039 | 0.922 | 1.932 | 30.44 |
| sliding | 0.2265 | 0.0113 | 0.0036 | 0.942 | 1.289 | 30.40 |
| squash | 0.2263 | 0.0112 | 0.0039 | 0.994 | 1.395 | 28.85 |

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
| depth-prior-ba | 0.84 | 0.65 | 0.099 | 0.039 | 0.042 | 0.047 |
| colmap | 3.09 | 1.71 | 0.249 | 0.039 | 0.076 | 0.082 |

**Novel view synthesis**

| | PSNR | SSIM | val PSNR | val SSIM | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|
| full | 25.90 | 0.876 | 22.49 | 0.750 | 17.85 |
| static-only | 23.15 | 0.828 | 20.11 | 0.701 | 11.22 |
| oracle-depth | 26.08 | 0.883 | 22.57 | 0.759 | 18.24 |
| oracle-tracks | 25.94 | 0.883 | 24.21 | 0.823 | 19.08 |
| oracle-poses | 25.94 | 0.879 | 23.63 | 0.809 | 17.61 |
| oracle-all | 26.57 | 0.897 | 24.58 | 0.836 | 21.01 |
| no-depth-loss | 25.92 | 0.875 | 22.49 | 0.750 | 17.67 |
| no-track-loss | 24.52 | 0.848 | 22.22 | 0.749 | 16.07 |
| no-rigidity | 25.76 | 0.875 | 22.49 | 0.750 | 18.10 |
| no-depth-correction | 25.45 | 0.864 | 22.06 | 0.723 | 16.67 |
| depth-prior-ba | 25.81 | 0.876 | 20.64 | 0.643 | 16.63 |
| colmap | 25.10 | 0.852 | 19.60 | 0.503 | 15.28 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 8.3 | 0.519 | 0.749 | 10.0 | 25.6 | 0.743 | 0.011 | 0.438 |
| static-only | 8.6 | 0.502 | 0.737 | - | 73.9 | 0.000 | 0.000 | 0.488 |
| oracle-depth | 8.1 | 0.542 | 0.764 | 10.6 | 29.4 | 0.763 | 0.010 | 0.439 |
| oracle-tracks | 4.4 | 0.750 | 0.928 | 4.3 | 7.8 | 0.949 | 0.001 | 1.000 |
| oracle-poses | 5.2 | 0.684 | 0.894 | 9.1 | 24.4 | 0.745 | 0.011 | 0.438 |
| oracle-all | 3.4 | 0.844 | 0.976 | 2.6 | 2.2 | 0.958 | 0.000 | 1.000 |
| no-depth-loss | 8.4 | 0.513 | 0.742 | 10.3 | 26.0 | 0.743 | 0.011 | 0.438 |
| no-track-loss | 7.9 | 0.554 | 0.764 | 13.4 | 26.6 | 0.743 | 0.011 | 0.438 |
| no-rigidity | 8.3 | 0.522 | 0.749 | 10.5 | 29.3 | 0.743 | 0.011 | 0.438 |
| no-depth-correction | 7.9 | 0.458 | 0.759 | 11.5 | 32.0 | 0.672 | 0.023 | 0.425 |
| depth-prior-ba | 16.2 | 0.199 | 0.472 | 11.3 | 35.8 | 0.744 | 0.011 | 0.438 |
| colmap | 24.4 | 0.213 | 0.422 | 15.6 | 32.2 | 0.671 | 0.021 | 0.430 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.2263 | 0.0111 | 0.0035 | 0.782 | 1.520 | 29.90 |
| static-only | 0.2263 | 0.0111 | 0.0044 | 1.077 | 1.520 | 25.65 |
| oracle-depth | 0.0001 | 0.0033 | 0.0036 | 0.843 | 1.520 | 30.22 |
| oracle-tracks | 0.2263 | 0.0087 | 0.0026 | 0.866 | 1.520 | 31.05 |
| oracle-poses | 0.2263 | 0.0109 | 0.0032 | 0.785 | 1.520 | 29.81 |
| oracle-all | 0.0001 | 0.0002 | 0.0020 | 0.679 | 1.520 | 32.62 |
| no-depth-loss | 0.2263 | 0.0111 | 0.0035 | 0.750 | 1.520 | 29.81 |
| no-track-loss | 0.2263 | 0.0111 | 0.0041 | 1.084 | 1.520 | 28.50 |
| no-rigidity | 0.2263 | 0.0111 | 0.0035 | 0.765 | 1.520 | 30.07 |
| no-depth-correction | 0.2263 | 0.0314 | 0.0043 | 0.848 | 1.520 | 29.10 |
| depth-prior-ba | 0.2263 | 0.0112 | 0.0046 | 0.845 | 1.520 | 28.68 |
| colmap | 0.2263 | 0.0133 | 0.0075 | 0.961 | 1.520 | 27.28 |

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
