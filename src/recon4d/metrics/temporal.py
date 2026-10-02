"""Temporal consistency metrics for rendered videos and depth sequences.

Three complementary notions are measured:

* :func:`warping_error` - *appearance* stability along motion trajectories: the next frame,
  warped back with the optical flow, should look like the current one (Lai et al.,
  "Learning Blind Video Temporal Consistency", ECCV 2018).
* :func:`temporal_difference_psnr` - fidelity of the *changes* between consecutive frames
  with respect to a reference video; it isolates what moves.
* :func:`depth_temporal_error` - *geometric* stability: two consecutive depth maps, lifted
  to 3D, should describe the same surface (the temporal alignment error used to evaluate
  video depth estimators such as ChronoDepth / DepthCrafter, generalised here to moving
  objects through ground-truth correspondences).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from recon4d.geometry.camera import pixel_centers, sample_depth, transform_points, unproject


def warp_backward(image: Tensor, flow: Tensor) -> Tensor:
    """Warp ``image (H, W, C)`` of frame ``t+1`` back to frame ``t``.

    ``flow (H, W, 2)`` is the forward flow from ``t`` to ``t+1``: pixel ``x`` of the result
    is ``image`` sampled at ``x + flow(x)``.
    """
    height, width = flow.shape[:2]
    target = pixel_centers(height, width, dtype=flow.dtype) + flow
    grid = target / torch.tensor([width, height], dtype=flow.dtype) * 2.0 - 1.0
    warped = F.grid_sample(
        image.permute(2, 0, 1)[None],
        grid[None].to(image.dtype),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped[0].permute(1, 2, 0)


def warping_error(frames: Tensor, flows: Tensor, valid: Tensor) -> Tensor:
    """Mean squared difference between each frame and its flow-warped successor.

    Args:
        frames: ``(T, H, W, 3)`` video.
        flows: ``(T - 1, H, W, 2)`` forward flow between consecutive frames.
        valid: ``(T - 1, H, W)`` True where the correspondence exists (not occluded, not
            leaving the image).

    Returns:
        The warping error averaged over valid pixels and frame pairs. It is not zero for a
        perfect video (shading and resampling change appearance), so it is best read
        relative to the error of the reference video computed with the same flow.
    """
    total = frames.new_zeros(())
    count = frames.new_zeros(())
    for t in range(frames.shape[0] - 1):
        warped = warp_backward(frames[t + 1], flows[t])
        squared = ((frames[t] - warped) ** 2).mean(dim=-1)
        total = total + (squared * valid[t]).sum()
        count = count + valid[t].sum()
    return total / count.clamp_min(1)


def temporal_difference_psnr(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """PSNR between the frame-to-frame differences of two videos ``(T, H, W, 3)``.

    A video that reproduces every frame of the reference reaches infinity; one that
    flickers, lags behind the motion or freezes moving objects is penalised even if each
    frame alone looks plausible. ``mask (T - 1, H, W)`` restricts the comparison.
    """
    diff_pred = pred[1:] - pred[:-1]
    diff_target = target[1:] - target[:-1]
    squared = (diff_pred - diff_target) ** 2
    if mask is None:
        mse = squared.mean()
    else:
        mse = (squared * mask[..., None]).sum() / (mask.sum() * 3).clamp_min(1)
    # Differences of [0, 1] images span [-1, 1]: the peak-to-peak range is 2.
    return 10.0 * torch.log10(4.0 / mse.clamp_min(1e-12))


def depth_temporal_error(
    depth: Tensor,
    K: Tensor,
    c2w: Tensor,
    flows: Tensor,
    valid: Tensor,
    scene_flow: Tensor | None = None,
) -> Tensor:
    """Relative 3D inconsistency between consecutive depth maps.

    For every valid correspondence ``x -> x + flow(x)`` between frames ``t`` and ``t+1``,
    both pixels are lifted to world space with their own depth. For a static point the two
    lifts must coincide; for a moving point they must differ by its true displacement
    ``scene_flow``. The metric is the mean of

        ``|| (P_{t+1} - P_t) - scene_flow || / depth_t``

    i.e. the 3D jitter relative to the distance to the camera. A per-frame depth estimator
    that flickers scores high even when each map is individually accurate. Correspondences
    landing on a depth discontinuity of frame ``t+1`` are skipped
    (see :func:`recon4d.geometry.camera.sample_depth`).

    Args:
        depth: ``(T, H, W)`` z-depth maps in the frame of the poses ``c2w (T, 4, 4)``.
        flows: ``(T - 1, H, W, 2)`` forward optical flow.
        valid: ``(T - 1, H, W)`` valid correspondences.
        scene_flow: optional ``(T - 1, H, W, 3)`` true world displacement of each pixel's
            surface point (zero for static pixels). ``None`` assumes a static scene, so
            ``valid`` should then exclude moving objects.
    """
    n_frames, height, width = depth.shape
    centers = pixel_centers(height, width, dtype=depth.dtype)
    total = depth.new_zeros(())
    count = depth.new_zeros(())
    for t in range(n_frames - 1):
        mask = valid[t] & (depth[t] > 0)
        if not mask.any():
            continue
        uv = centers[mask]
        uv_next = uv + flows[t][mask]
        z = depth[t][mask]
        z_next, ok = sample_depth(depth[t + 1], uv_next)
        point = transform_points(c2w[t], unproject(K, uv, z))
        point_next = transform_points(c2w[t + 1], unproject(K, uv_next, z_next))
        displacement = point_next - point
        if scene_flow is not None:
            displacement = displacement - scene_flow[t][mask]
        error = displacement.norm(dim=-1) / z
        total = total + (error * ok).sum()
        count = count + ok.sum()
    return total / count.clamp_min(1)
