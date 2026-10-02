"""recon4d: 3D/4D reconstruction from monocular video.

The package is organised as a pipeline:

``data``      video / synthetic sequences with exact ground truth
``frontend``  depth prediction, point tracking, camera pose estimation
``fusion``    depth alignment, back-projection and TSDF fusion
``gaussians`` differentiable Gaussian splatting (static and dynamic)
``nerf``      volume-rendered radiance field baseline
``metrics``   Chamfer, PSNR/SSIM/LPIPS, pose error, temporal consistency

Conventions shared by every module are documented in :mod:`recon4d.geometry`.
"""

__version__ = "0.1.0"
