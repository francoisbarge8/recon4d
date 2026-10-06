# Benchmark results (gpu profile)

## Per scene, `full` pipeline

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.11 | 0.08 | 0.011 | 0.011 | 0.009 | 0.008 |
| rolling | 0.10 | 0.08 | 0.011 | 0.010 | 0.008 | 0.007 |
| sliding | 0.12 | 0.08 | 0.012 | 0.019 | 0.013 | 0.014 |
| squash | 0.11 | 0.12 | 0.016 | 0.010 | 0.008 | 0.008 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| still | 31.12 | 0.880 | 0.081 | 28.55 | 0.809 | 0.142 | - |
| rolling | 27.29 | 0.863 | 0.089 | 24.13 | 0.783 | 0.152 | 16.82 |
| sliding | 26.83 | 0.864 | 0.098 | 23.38 | 0.773 | 0.159 | 15.98 |
| squash | 27.61 | 0.862 | 0.093 | 23.91 | 0.802 | 0.133 | 17.23 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| still | 3.2 | 0.844 | 0.980 | - | - | - | 0.038 | 0.236 |
| rolling | 2.8 | 0.902 | 0.972 | 5.4 | 14.0 | 0.569 | 0.041 | 0.335 |
| sliding | 4.0 | 0.704 | 0.964 | 18.8 | 30.8 | 0.704 | 0.036 | 0.572 |
| squash | 2.7 | 0.881 | 0.989 | 5.5 | 11.8 | 0.645 | 0.041 | 0.527 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| still | 0.0136 | 0.0022 | 0.0007 | 0.126 | 1.272 | - |
| rolling | 0.0083 | 0.0025 | 0.0018 | 0.934 | 1.515 | 30.76 |
| sliding | 0.0107 | 0.0026 | 0.0024 | 1.519 | 0.861 | 28.70 |
| squash | 0.0092 | 0.0024 | 0.0024 | 1.054 | 1.155 | 29.28 |

## Variants, averaged over the scenes

**Camera poses and depth**

| | ATE (cm) | RPE-t (cm) | RPE-r (deg) | AbsRel raw | AbsRel aligned | AbsRel rendered |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.11 | 0.09 | 0.012 | 0.012 | 0.010 | 0.009 |
| static-only | 0.11 | 0.09 | 0.012 | 0.012 | 0.010 | 0.034 |
| oracle-depth | 0.11 | 0.09 | 0.012 | 0.000 | 0.005 | 0.007 |
| oracle-tracks | 0.00 | 0.00 | 0.000 | 0.012 | 0.006 | 0.005 |
| oracle-poses | 0.00 | 0.00 | 0.000 | 0.012 | 0.008 | 0.007 |
| oracle-all | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 0.003 |
| no-depth-loss | 0.11 | 0.09 | 0.012 | 0.012 | 0.010 | 0.013 |
| no-track-loss | 0.11 | 0.09 | 0.012 | 0.012 | 0.010 | 0.011 |
| no-rigidity | 0.11 | 0.09 | 0.012 | 0.012 | 0.010 | 0.010 |
| no-depth-correction | 0.11 | 0.09 | 0.012 | 0.012 | 0.013 | 0.012 |
| depth-prior-ba | 0.15 | 0.09 | 0.013 | 0.012 | 0.012 | 0.011 |
| depth-anything | 0.11 | 0.09 | 0.012 | 0.027 | 0.019 | 0.014 |
| cotracker | 0.11 | 0.09 | 0.012 | 0.012 | 0.010 | 0.009 |
| raft | 0.11 | 0.09 | 0.012 | 0.012 | 0.010 | 0.013 |
| colmap | 0.60 | 0.47 | 0.065 | 0.012 | 0.012 | 0.011 |

**Novel view synthesis**

| | PSNR | SSIM | LPIPS | val PSNR | val SSIM | val LPIPS | val PSNR (moving) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 28.21 | 0.867 | 0.090 | 24.99 | 0.792 | 0.147 | 16.67 |
| static-only | 25.88 | 0.843 | 0.130 | 21.49 | 0.738 | 0.233 | 11.08 |
| oracle-depth | 28.27 | 0.868 | 0.088 | 25.67 | 0.801 | 0.140 | 18.05 |
| oracle-tracks | 27.60 | 0.863 | 0.085 | 25.37 | 0.800 | 0.143 | 16.56 |
| oracle-poses | 28.45 | 0.871 | 0.087 | 25.62 | 0.809 | 0.136 | 16.95 |
| oracle-all | 28.07 | 0.867 | 0.079 | 26.22 | 0.809 | 0.131 | 18.52 |
| no-depth-loss | 28.14 | 0.867 | 0.092 | 24.41 | 0.783 | 0.153 | 15.69 |
| no-track-loss | 27.11 | 0.857 | 0.101 | 24.47 | 0.787 | 0.160 | 15.48 |
| no-rigidity | 27.70 | 0.861 | 0.092 | 24.70 | 0.788 | 0.147 | 16.00 |
| no-depth-correction | 27.84 | 0.864 | 0.091 | 24.89 | 0.790 | 0.150 | 16.53 |
| depth-prior-ba | 28.10 | 0.866 | 0.092 | 24.48 | 0.780 | 0.149 | 16.48 |
| depth-anything | 27.75 | 0.862 | 0.094 | 23.82 | 0.774 | 0.162 | 15.20 |
| cotracker | 27.82 | 0.862 | 0.086 | 24.80 | 0.785 | 0.150 | 16.49 |
| raft | 27.76 | 0.864 | 0.096 | 24.54 | 0.779 | 0.162 | 16.28 |
| colmap | 27.74 | 0.861 | 0.092 | 24.61 | 0.773 | 0.154 | 16.57 |

**Geometry and motion**

| | Chamfer (cm) | F@5cm | F@10cm | Chamfer moving (cm) | 3D EPE (cm) | Mask IoU | Mask FPR | Track d_avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 3.2 | 0.833 | 0.976 | 9.9 | 18.9 | 0.639 | 0.039 | 0.417 |
| static-only | 4.5 | 0.787 | 0.939 | - | 74.2 | 0.000 | 0.000 | 0.328 |
| oracle-depth | 2.8 | 0.868 | 0.995 | 3.5 | 7.0 | 0.644 | 0.038 | 0.418 |
| oracle-tracks | 2.2 | 0.942 | 0.990 | 9.7 | 16.1 | 0.981 | 0.000 | 1.000 |
| oracle-poses | 2.6 | 0.912 | 0.986 | 10.3 | 14.6 | 0.640 | 0.039 | 0.418 |
| oracle-all | 1.8 | 0.969 | 0.998 | 1.8 | 2.8 | 0.984 | 0.000 | 1.000 |
| no-depth-loss | 3.9 | 0.789 | 0.945 | 10.8 | 19.0 | 0.639 | 0.039 | 0.417 |
| no-track-loss | 3.2 | 0.828 | 0.978 | 11.1 | 19.3 | 0.639 | 0.039 | 0.417 |
| no-rigidity | 3.2 | 0.832 | 0.977 | 10.5 | 28.1 | 0.639 | 0.039 | 0.417 |
| no-depth-correction | 3.2 | 0.859 | 0.979 | 10.2 | 16.3 | 0.640 | 0.039 | 0.418 |
| depth-prior-ba | 3.7 | 0.744 | 0.961 | 10.7 | 16.2 | 0.639 | 0.039 | 0.417 |
| depth-anything | 4.1 | 0.759 | 0.924 | 11.1 | 20.9 | 0.629 | 0.041 | 0.416 |
| cotracker | 3.1 | 0.845 | 0.981 | 10.3 | 15.5 | 0.639 | 0.039 | 0.885 |
| raft | 3.8 | 0.782 | 0.941 | 15.1 | 34.9 | 0.763 | 0.010 | 0.292 |
| colmap | 3.8 | 0.783 | 0.931 | 10.5 | 15.5 | 0.599 | 0.045 | 0.410 |

**Temporal consistency**

| | Depth TE raw | Depth TE aligned | Depth TE rendered | Warp error (x1e-3) | Warp error GT video | td-PSNR val |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.0104 | 0.0025 | 0.0018 | 0.908 | 1.201 | 29.58 |
| static-only | 0.0104 | 0.0025 | 0.0035 | 0.559 | 1.201 | 27.57 |
| oracle-depth | 0.0000 | 0.0010 | 0.0017 | 1.005 | 1.201 | 30.78 |
| oracle-tracks | 0.0104 | 0.0016 | 0.0019 | 1.042 | 1.201 | 29.40 |
| oracle-poses | 0.0104 | 0.0025 | 0.0016 | 0.892 | 1.201 | 29.84 |
| oracle-all | 0.0000 | 0.0000 | 0.0014 | 0.927 | 1.201 | 31.21 |
| no-depth-loss | 0.0104 | 0.0025 | 0.0019 | 0.957 | 1.201 | 28.64 |
| no-track-loss | 0.0104 | 0.0025 | 0.0024 | 1.082 | 1.201 | 28.18 |
| no-rigidity | 0.0104 | 0.0025 | 0.0020 | 0.973 | 1.201 | 28.80 |
| no-depth-correction | 0.0104 | 0.0038 | 0.0020 | 1.043 | 1.201 | 29.42 |
| depth-prior-ba | 0.0104 | 0.0024 | 0.0019 | 0.966 | 1.201 | 29.28 |
| depth-anything | 0.0078 | 0.0066 | 0.0027 | 1.036 | 1.201 | 27.48 |
| cotracker | 0.0104 | 0.0025 | 0.0019 | 0.897 | 1.201 | 29.26 |
| raft | 0.0104 | 0.0025 | 0.0019 | 0.802 | 1.201 | 29.47 |
| colmap | 0.0104 | 0.0026 | 0.0023 | 0.993 | 1.201 | 29.29 |

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
