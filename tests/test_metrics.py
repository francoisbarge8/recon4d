"""Every metric is checked against a value derived independently (closed form or SciPy)."""

import math

import numpy as np
import pytest
import torch
from scipy.signal import correlate2d

from recon4d.geometry import (
    Sim3,
    look_at,
    pixel_centers,
    quat_to_rotmat,
    sample_depth,
    so3_exp,
)
from recon4d.metrics import (
    align_depth,
    chamfer_distance,
    depth_metrics,
    depth_temporal_error,
    f_score,
    geometry_metrics,
    masked_ssim_map,
    nearest_distances,
    nearest_distances_torch,
    pose_metrics,
    psnr,
    relative_pose_error,
    ssim,
    ssim_map,
    tapvid_metrics,
    temporal_difference_psnr,
    trajectory_error_3d,
    warp_backward,
    warping_error,
)

# ----------------------------------------------------------------------------- image


def test_psnr_closed_form_and_mask():
    target = torch.rand(2, 16, 16, 3)
    assert psnr(target + 0.1, target).item() == pytest.approx(20.0, abs=1e-4)
    assert psnr(target + 0.01, target).item() == pytest.approx(40.0, abs=1e-3)
    # Errors outside the mask are ignored.
    pred = target.clone()
    pred[:, :8] += 0.1
    mask = torch.zeros(2, 16, 16, dtype=torch.bool)
    mask[:, :8] = True
    assert psnr(pred, target, mask).item() == pytest.approx(20.0, abs=1e-4)
    assert psnr(pred, target, ~mask).item() > 100.0


def _ssim_reference(x: np.ndarray, y: np.ndarray, size: int = 11, sigma: float = 1.5) -> float:
    """SSIM of two grey images written with SciPy, straight from Wang et al. (2004)."""
    coords = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(coords**2) / (2 * sigma**2))
    window = np.outer(g, g) / np.outer(g, g).sum()

    def blur(z):
        return correlate2d(z, window, mode="valid")

    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    index = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    return float(index.mean())


def test_ssim_matches_scipy_reference():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(40, 56, 3, generator=generator, dtype=torch.float64)
    y = (x + 0.15 * torch.randn(40, 56, 3, generator=generator, dtype=torch.float64)).clamp(0, 1)
    expected = np.mean([_ssim_reference(x[..., c].numpy(), y[..., c].numpy()) for c in range(3)])
    assert ssim(x, y).item() == pytest.approx(expected, abs=1e-10)
    assert 0.2 < expected < 0.95


def test_ssim_properties():
    x = torch.rand(3, 32, 32, 3)
    assert ssim(x, x).item() == pytest.approx(1.0, abs=1e-6)
    noisy = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    noisier = (x + 0.3 * torch.randn_like(x)).clamp(0, 1)
    assert 1.0 > ssim(x, noisy) > ssim(x, noisier)
    assert ssim(x, noisy).item() == pytest.approx(ssim(noisy, x).item(), abs=1e-6)


def test_ssim_is_differentiable():
    x = torch.rand(1, 24, 24, 3, requires_grad=True)
    y = torch.rand(1, 24, 24, 3)
    (1.0 - ssim(x, y)).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_masked_ssim_agrees_with_plain_ssim_and_ignores_outside():
    generator = torch.Generator().manual_seed(1)
    x = torch.rand(1, 3, 40, 40, generator=generator, dtype=torch.float64)
    y = (x + 0.1 * torch.randn(1, 3, 40, 40, generator=generator, dtype=torch.float64)).clamp(0, 1)
    full = torch.ones(1, 1, 40, 40, dtype=torch.float64)
    # With a full mask the two definitions coincide wherever the window fits in the image.
    assert torch.allclose(masked_ssim_map(x, y, full)[..., 5:-5, 5:-5], ssim_map(x, y), atol=1e-10)

    mask = torch.zeros(40, 40, dtype=torch.bool)
    mask[10:22, 8:30] = True
    a = x[0].permute(1, 2, 0)
    b = y[0].permute(1, 2, 0)
    corrupted = a.clone()
    corrupted[~mask] = torch.rand(int((~mask).sum()), 3, generator=generator, dtype=torch.float64)
    assert ssim(corrupted, b, mask).item() == pytest.approx(ssim(a, b, mask).item(), abs=1e-12)
    assert ssim(corrupted, b).item() < ssim(a, b).item()
    with pytest.warns(UserWarning):
        assert math.isnan(ssim(a, b, torch.zeros(40, 40, dtype=torch.bool)).item())


# -------------------------------------------------------------------------- geometry


def _plane(n: int, z: float, generator) -> torch.Tensor:
    xy = torch.rand(n, 2, generator=generator, dtype=torch.float64)
    return torch.cat([xy, torch.full((n, 1), z, dtype=torch.float64)], dim=1)


def test_chamfer_between_parallel_planes():
    generator = torch.Generator().manual_seed(0)
    a, b = _plane(20000, 0.0, generator), _plane(20000, 0.05, generator)
    result = chamfer_distance(a, b)
    for key in ("accuracy", "completeness", "chamfer"):
        assert result[key].item() == pytest.approx(0.05, rel=0.02)
    assert chamfer_distance(a, a)["chamfer"].item() == 0.0


def test_accuracy_and_completeness_are_not_symmetric():
    generator = torch.Generator().manual_seed(0)
    gt = _plane(5000, 0.0, generator)
    partial = gt[gt[:, 0] < 0.5]  # a correct but incomplete reconstruction
    result = chamfer_distance(partial, gt)
    assert result["accuracy"].item() == 0.0
    assert result["completeness"].item() > 0.05


def test_kdtree_and_brute_force_nearest_neighbours_agree():
    a = torch.rand(700, 3, dtype=torch.float64)
    b = torch.rand(900, 3, dtype=torch.float64)
    assert torch.allclose(
        nearest_distances(a, b), nearest_distances_torch(a, b, chunk=128), atol=1e-12
    )
    brute = torch.cdist(a, b).amin(dim=1)
    assert torch.allclose(nearest_distances(a, b), brute, atol=1e-12)
    assert torch.isinf(nearest_distances(a, torch.zeros(0, 3, dtype=torch.float64))).all()


def test_differentiable_chamfer_pulls_points_together():
    a = torch.rand(50, 3, requires_grad=True)
    b = torch.rand(60, 3)
    loss = chamfer_distance(a, b, differentiable=True)["chamfer"]
    loss.backward()
    with torch.no_grad():
        moved = a - 0.05 * a.grad / a.grad.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    assert chamfer_distance(moved, b)["chamfer"] < loss


def test_f_score_thresholds():
    generator = torch.Generator().manual_seed(0)
    gt = _plane(4000, 0.0, generator)
    shifted = gt + torch.tensor([0.0, 0.0, 0.03], dtype=torch.float64)
    assert f_score(shifted, gt, 0.05)["fscore"].item() == pytest.approx(1.0)
    assert f_score(shifted, gt, 0.02)["fscore"].item() == 0.0
    metrics = geometry_metrics(shifted, gt, thresholds=(0.02, 0.05))
    assert metrics["chamfer"] == pytest.approx(0.03, rel=1e-6)
    assert metrics["fscore@0.05"] == 1.0 and metrics["fscore@0.02"] == 0.0
    # Half of the prediction is far away: precision 0.5, recall 1.
    outliers = torch.cat([gt, gt + 10.0])
    half = geometry_metrics(outliers, gt, thresholds=(0.05,))
    assert half["precision@0.05"] == pytest.approx(0.5)
    assert half["recall@0.05"] == pytest.approx(1.0)
    assert half["fscore@0.05"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        geometry_metrics(torch.zeros(0, 3), gt)


# ------------------------------------------------------------------------------ pose


def _trajectory(n: int = 20) -> torch.Tensor:
    angles = torch.linspace(0.2, 1.6, n, dtype=torch.float64)
    eyes = torch.stack(
        [3 * torch.cos(angles), 1.0 + 0.3 * torch.sin(3 * angles), 3 * torch.sin(angles)], 1
    )
    return torch.stack([look_at(e, torch.zeros(3, dtype=torch.float64)) for e in eyes])


def test_pose_metrics_are_invariant_to_similarity():
    gt = _trajectory()
    sim = Sim3(
        torch.tensor(0.37, dtype=torch.float64),
        quat_to_rotmat(torch.tensor([0.3, -0.5, 0.2, 0.8], dtype=torch.float64)),
        torch.tensor([4.0, -2.0, 1.0], dtype=torch.float64),
    )
    est = sim.apply_to_c2w(gt)
    metrics = pose_metrics(est, gt)
    assert metrics["ate"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rpe_trans"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rpe_rot_deg"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["rot_deg"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["scale"] == pytest.approx(1 / 0.37)


def test_ate_measures_position_noise():
    gt = _trajectory(200)
    est = gt.clone()
    noise = 0.02 * torch.randn(
        200, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    est[:, :3, 3] += noise
    # The alignment absorbs a little of the noise, hence the slightly smaller value.
    assert pose_metrics(est, gt, with_scale=False)["ate"] == pytest.approx(
        0.02 * math.sqrt(3), rel=0.08
    )


def test_orientation_error_is_measured_after_rotation_alignment():
    gt = _trajectory()
    roll = so3_exp(torch.tensor([0.0, 0.0, math.radians(5.0)], dtype=torch.float64))
    # One camera out of 20 is rolled by 5 degrees about its optical axis. The best global
    # rotation absorbs 1/20 of it, leaving 19/20 on that camera and 1/20 on the others.
    est = gt.clone()
    est[7, :3, :3] = gt[7, :3, :3] @ roll
    metrics = pose_metrics(est, gt)
    assert metrics["ate"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["rot_deg"] == pytest.approx(2 * 5.0 * 19 / 400, rel=0.01)


def test_rpe_of_a_constant_camera_roll_is_second_order():
    gt = _trajectory()
    roll = so3_exp(torch.tensor([0.0, 0.0, math.radians(5.0)], dtype=torch.float64))
    est = gt.clone()
    est[:, :3, :3] = gt[:, :3, :3] @ roll  # every camera rolled by 5 degrees about its axis
    metrics = pose_metrics(est, gt)
    # No rotation of the world can undo a roll of each camera about its own axis ...
    assert 1.0 < metrics["rot_deg"] <= 5.0
    # ... but it only conjugates the relative rotations: the relative error is the
    # commutator of the roll with the frame-to-frame rotation, a second-order term.
    assert 0.0 < metrics["rpe_rot_deg"] < 0.1 * 5.0


def test_rpe_detects_a_single_bad_frame():
    gt = _trajectory(11)
    est = gt.clone()
    # From frame 5 on, the whole trajectory is rotated by 3 degrees about the world origin:
    # relative poses are unchanged inside each half and wrong only across the break.
    kick = torch.eye(4, dtype=torch.float64)
    kick[:3, :3] = so3_exp(torch.tensor([0.0, math.radians(3.0), 0.0], dtype=torch.float64))
    est[5:] = kick @ gt[5:]
    trans, rot = relative_pose_error(est, gt)
    assert rot.item() == pytest.approx(3.0 / math.sqrt(10), rel=1e-6)  # one of 10 pairs is off
    assert trans.item() > 0.0


# ----------------------------------------------------------------------------- depth


def test_depth_alignment_modes_recover_their_transform():
    gt = torch.rand(30, 40, dtype=torch.float64) * 4.0 + 1.0
    assert torch.allclose(align_depth(gt * 3.0, gt, mode="scale"), gt, atol=1e-10)
    assert torch.allclose(align_depth(gt * 0.5 + 2.0, gt, mode="scale_shift"), gt, atol=1e-8)
    affine_disparity = 2.0 / gt - 0.1  # what an affine-invariant network would output
    assert torch.allclose(align_depth(affine_disparity, gt, mode="disparity"), gt, atol=1e-8)
    assert torch.equal(align_depth(gt * 3.0, gt, mode="none"), gt * 3.0)
    with pytest.raises(ValueError):
        align_depth(gt, gt, mode="affine")


def test_depth_metrics_closed_form():
    gt = torch.rand(20, 20, dtype=torch.float64) * 4.0 + 1.0
    perfect = depth_metrics(gt, gt)
    assert perfect["abs_rel"] == 0.0 and perfect["rmse"] == 0.0 and perfect["delta1"] == 1.0
    ten_percent = depth_metrics(gt * 1.1, gt)
    assert ten_percent["abs_rel"] == pytest.approx(0.1)
    assert ten_percent["rmse_log"] == pytest.approx(math.log(1.1))
    assert ten_percent["delta1"] == 1.0
    doubled = depth_metrics(gt * 2.0, gt)
    assert doubled["delta1"] == 0.0 and doubled["delta2"] == 0.0 and doubled["delta3"] == 0.0
    assert depth_metrics(gt * 1.5, gt)["delta2"] == 1.0
    mask = torch.zeros(20, 20, dtype=torch.bool)
    mask[:5] = True
    corrupted = gt.clone()
    corrupted[5:] *= 3.0
    assert depth_metrics(corrupted, gt, mask)["abs_rel"] == 0.0


def test_sample_depth_is_exact_on_planes_and_rejects_edges():
    # A slanted plane seen by a pinhole camera: inverse depth is affine in pixel coordinates.
    height, width = 20, 30
    centers = pixel_centers(height, width, dtype=torch.float64)
    inverse = 0.2 + 0.004 * centers[..., 0] + 0.003 * centers[..., 1]
    depth = 1.0 / inverse
    uv = torch.rand(200, 2, dtype=torch.float64) * torch.tensor([width - 2.0, height - 2.0]) + 1.0
    z, ok = sample_depth(depth, uv, max_ratio=1.2)
    expected = 1.0 / (0.2 + 0.004 * uv[:, 0] + 0.003 * uv[:, 1])
    assert ok.all()
    assert torch.allclose(z, expected, atol=1e-12)

    step = torch.ones(height, width, dtype=torch.float64)
    step[:, 15:] = 3.0  # a depth discontinuity between columns 14 and 15
    z, ok = sample_depth(step, torch.tensor([[15.0, 10.0], [5.0, 10.0], [25.0, 10.0], [-1.0, 3.0]]))
    assert ok.tolist() == [False, True, True, False]
    assert z.tolist() == [0.0, 1.0, 3.0, 0.0]
    holes = step.clone()
    holes[10, 5] = 0.0
    assert not sample_depth(holes, torch.tensor([[5.5, 10.5]]))[1][0]
    valid = torch.ones(height, width, dtype=torch.bool)
    valid[10, 25] = False
    assert not sample_depth(step, torch.tensor([[25.5, 10.5]]), valid=valid)[1][0]


# -------------------------------------------------------------------------- tracking


def test_tapvid_perfect_and_offset_predictions():
    n, t = 50, 12
    gt_uv = torch.rand(n, t, 2) * torch.tensor([192.0, 144.0])
    gt_vis = torch.rand(n, t) > 0.3
    query = torch.zeros(n, dtype=torch.int64)
    perfect = tapvid_metrics(gt_uv, gt_vis, gt_uv, gt_vis, query, 192, 144)
    assert perfect["delta_avg"] == 1.0 and perfect["average_jaccard"] == 1.0
    assert perfect["occlusion_accuracy"] == 1.0 and perfect["epe"] == 0.0

    # 3 px at 256 x 256 corresponds to 3 * 192 / 256 px horizontally in the original image.
    shifted = gt_uv + torch.tensor([3.0 * 192 / 256, 0.0])
    metrics = tapvid_metrics(shifted, gt_vis, gt_uv, gt_vis, query, 192, 144)
    assert metrics["pts_within_4"] == 1.0 and metrics["pts_within_2"] == 0.0
    assert metrics["delta_avg"] == pytest.approx(3 / 5)
    assert metrics["epe"] == pytest.approx(3.0 * 192 / 256, rel=1e-5)


def test_tapvid_visibility_and_jaccard_counts():
    # One track, 5 frames, frame 0 is the query (excluded).
    gt_uv = torch.zeros(1, 5, 2)
    gt_vis = torch.tensor([[True, True, True, False, False]])
    pred_vis = torch.tensor([[True, True, False, True, False]])
    query = torch.zeros(1, dtype=torch.int64)
    metrics = tapvid_metrics(gt_uv, pred_vis, gt_uv, gt_vis, query, 256, 256)
    assert metrics["occlusion_accuracy"] == pytest.approx(2 / 4)
    assert metrics["pts_within_1"] == 1.0
    # TP = 1 (frame 1), FP = 1 (frame 3), GT positives = 2  ->  1 / (2 + 1).
    assert metrics["jaccard_1"] == pytest.approx(1 / 3)


def test_trajectory_error_3d():
    gt = torch.rand(10, 6, 3)
    pred = gt + torch.tensor([0.06, 0.0, 0.0])
    valid = torch.ones(10, 6, dtype=torch.bool)
    valid[0] = False
    metrics = trajectory_error_3d(pred, gt, valid)
    assert metrics["epe_3d"] == pytest.approx(0.06, rel=1e-5)
    assert metrics["delta_3d_0.05"] == 0.0 and metrics["delta_3d_0.1"] == 1.0


# -------------------------------------------------------------------------- temporal


def _translating_video(n_frames: int = 5, shift: int = 2):
    """A textured image translating by ``shift`` pixels per frame, with its flow."""
    generator = torch.Generator().manual_seed(0)
    base = torch.rand(40, 80, 3, generator=generator)
    base = torch.nn.functional.avg_pool2d(base.permute(2, 0, 1)[None], 5, 1, 2)[0].permute(1, 2, 0)
    frames = torch.stack([torch.roll(base, shifts=shift * t, dims=1) for t in range(n_frames)])
    flows = torch.zeros(n_frames - 1, 40, 80, 2)
    flows[..., 0] = shift
    valid = torch.ones(n_frames - 1, 40, 80, dtype=torch.bool)
    valid[:, :, -shift - 1 :] = False  # pixels leaving the image
    return frames, flows, valid


def test_warp_backward_inverts_a_translation():
    frames, flows, valid = _translating_video()
    warped = warp_backward(frames[1], flows[0])
    assert torch.allclose(warped[valid[0]], frames[0][valid[0]], atol=1e-5)


def test_warping_error_is_zero_for_consistent_video_and_detects_flicker():
    frames, flows, valid = _translating_video()
    assert warping_error(frames, flows, valid).item() < 1e-9
    flicker = frames.clone()
    flicker[1::2] += 0.1  # every other frame is brighter
    assert warping_error(flicker, flows, valid).item() == pytest.approx(0.01, rel=1e-3)


def test_temporal_difference_psnr():
    frames, _, _ = _translating_video()
    assert temporal_difference_psnr(frames, frames).item() > 100.0
    frozen = frames[:1].expand_as(frames)  # a still video cannot reproduce the motion
    lagging = torch.roll(frames, shifts=1, dims=2)
    assert temporal_difference_psnr(frozen, frames) < temporal_difference_psnr(lagging, frames)
    mask = torch.zeros(4, 40, 80, dtype=torch.bool)
    mask[:, :10] = True
    flicker = frames.clone()
    flicker[2, 10:] += 0.2  # only outside the mask
    assert temporal_difference_psnr(flicker, frames, mask).item() > 100.0
    # Constant brightness offset on one frame: difference error 0.2 on two transitions.
    expected = 10 * math.log10(4.0 / (0.2**2 * 2 / 4 * (30 / 40)))
    assert temporal_difference_psnr(flicker, frames).item() == pytest.approx(expected, abs=1e-3)


def test_depth_temporal_error_on_ground_truth(tiny_rolling):
    """Exact depth, poses, flow and scene flow are perfectly consistent; noisy depth is not."""
    seq = tiny_rolling
    gt = seq.gt
    pairs = [gt.oracle.scene_flow(t, t + 1) for t in range(seq.num_frames - 1)]
    flows = torch.stack([p[0] for p in pairs])
    valid = torch.stack([p[1] for p in pairs])
    scene_flow = torch.stack([p[2] for p in pairs])
    K = seq.K()
    exact = depth_temporal_error(gt.depth, K, gt.c2w, flows, valid, scene_flow)
    assert exact.item() < 2e-3
    # Ignoring the object motion is wrong on the moving balls ...
    static_assumption = depth_temporal_error(gt.depth, K, gt.c2w, flows, valid)
    assert static_assumption > 3 * exact
    # ... but fine when they are masked out.
    static_only = depth_temporal_error(gt.depth, K, gt.c2w, flows, valid & ~gt.dynamic_mask[:-1])
    assert static_only.item() < 2e-3
    # Per-frame scale flicker (what a monocular network does) is detected.
    flicker = 1.0 + 0.05 * torch.tensor([1.0, -1.0] * (seq.num_frames // 2))[:, None, None]
    flickering = depth_temporal_error(gt.depth * flicker, K, gt.c2w, flows, valid, scene_flow)
    assert flickering.item() == pytest.approx(0.1, rel=0.15)
