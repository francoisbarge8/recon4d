"""Bundle adjustment and track-based structure-from-motion."""

import math

import pytest
import torch

from recon4d.frontend.pose import (
    BAProblem,
    SfMConfig,
    bundle_adjust,
    projection_jacobians,
    reconstruct,
    reprojection_residuals,
)
from recon4d.frontend.pose.bundle_adjustment import DepthPrior, depth_jacobians, depth_residuals
from recon4d.frontend.pose.sfm import sample_track_depths
from recon4d.geometry import Intrinsics, invert_se3, look_at, project, so3_exp, transform_points
from recon4d.metrics import pose_metrics
from recon4d.types import Tracks

DT = torch.float64
W, H = 160, 120


def synthetic_problem(
    n_cams: int = 8,
    n_points: int = 120,
    seed: int = 0,
    visibility: float = 0.8,
    arc: float = 0.45,
):
    """Cameras on an arc of half-angle ``arc`` looking at a random point cloud."""
    generator = torch.Generator().manual_seed(seed)
    K = Intrinsics.from_fov(W, H, 60.0).matrix(DT)
    angles = torch.linspace(-arc, arc, n_cams, dtype=DT)
    eyes = torch.stack(
        [3.0 * torch.sin(angles), 0.4 * torch.cos(3 * angles), -3.0 * torch.cos(angles)], 1
    )
    w2c = invert_se3(torch.stack([look_at(e, torch.zeros(3, dtype=DT)) for e in eyes]))
    points = (torch.rand(n_points, 3, generator=generator, dtype=DT) - 0.5) * torch.tensor(
        [2.4, 1.6, 2.0], dtype=DT
    )
    uv, z = project(K, transform_points(w2c[None], points[:, None, None, :])[:, :, 0])
    visible = torch.rand(n_points, n_cams, generator=generator) < visibility
    visible[:, :2] = True
    assert (z > 0.5).all()
    return K, w2c, points, uv, visible


def as_problem(K, w2c, points, uv, visible) -> BAProblem:
    point_index, cam_index = torch.nonzero(visible, as_tuple=True)
    fixed = torch.zeros(w2c.shape[0], dtype=torch.bool)
    fixed[0] = True
    return BAProblem(w2c, points, cam_index, point_index, uv[point_index, cam_index], K, fixed)


# ----------------------------------------------------------------- bundle adjustment


def test_analytic_jacobians_match_autograd():
    K, w2c, points, uv, visible = synthetic_problem(n_cams=3, n_points=6)
    problem = as_problem(K, w2c, points, uv, visible)
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    ci, pi = problem.cam_index, problem.point_index
    _, cam = reprojection_residuals(R, t, points, K, ci, pi, problem.uv)
    J_pose, J_point, J_focal = projection_jacobians(R, cam, K, ci)

    def residual_of(pose_delta, point_delta, focal_delta):
        dR = so3_exp(pose_delta[:, :3])
        R_new = dR @ R
        t_new = (dR @ t[..., None])[..., 0] + pose_delta[:, 3:]
        K_new = K.clone()
        K_new[0, 0] = K[0, 0] + focal_delta
        K_new[1, 1] = K[1, 1] + focal_delta
        res, _ = reprojection_residuals(
            R_new, t_new, points + point_delta, K_new, ci, pi, problem.uv
        )
        return res

    zeros = (torch.zeros(3, 6, dtype=DT), torch.zeros(6, 3, dtype=DT), torch.zeros((), dtype=DT))
    auto_pose, auto_point, auto_focal = torch.autograd.functional.jacobian(residual_of, zeros)
    m = ci.shape[0]
    obs = torch.arange(m)
    # Each residual only depends on its own camera and point.
    assert torch.allclose(auto_pose[obs, :, ci], J_pose, atol=1e-9)
    assert torch.allclose(auto_point[obs, :, pi], J_point, atol=1e-9)
    assert torch.allclose(auto_focal, J_focal, atol=1e-9)
    others = auto_pose.clone()
    others[obs, :, ci] = 0.0
    assert others.abs().max() == 0.0


def perturb(w2c, points, pose_noise=0.02, point_noise=0.05, seed=1):
    generator = torch.Generator().manual_seed(seed)
    n = w2c.shape[0]
    delta = torch.randn(n, 6, generator=generator, dtype=DT) * pose_noise
    delta[0] = 0.0  # the gauge camera stays put
    noisy = w2c.clone()
    dR = so3_exp(delta[:, :3])
    noisy[:, :3, :3] = dR @ w2c[:, :3, :3]
    noisy[:, :3, 3] = (dR @ w2c[:, :3, 3, None])[..., 0] + delta[:, 3:]
    noisy_points = points + torch.randn(points.shape, generator=generator, dtype=DT) * point_noise
    return noisy, noisy_points


def test_bundle_adjustment_converges_to_zero_reprojection_error():
    K, w2c, points, uv, visible = synthetic_problem()
    noisy_w2c, noisy_points = perturb(w2c, points)
    result = bundle_adjust(as_problem(K, noisy_w2c, noisy_points, uv, visible), huber_delta=None)
    assert result.initial_cost > 1e3
    assert result.final_cost < 1e-12
    assert result.residuals.max() < 1e-6
    # The solution equals the ground truth up to the remaining gauge freedom (scale).
    metrics = pose_metrics(invert_se3(result.w2c), invert_se3(w2c))
    assert metrics["ate"] < 1e-7 and metrics["rot_deg"] < 1e-6
    assert torch.equal(result.w2c[0], noisy_w2c[0]), "the fixed camera must not move"


def test_bundle_adjustment_recovers_the_focal_length():
    K, w2c, points, uv, visible = synthetic_problem(n_cams=10, n_points=200)
    noisy_w2c, noisy_points = perturb(w2c, points, 0.01, 0.03)
    K_wrong = K.clone()
    K_wrong[0, 0] *= 1.15
    K_wrong[1, 1] *= 1.15
    problem = as_problem(K_wrong, noisy_w2c, noisy_points, uv, visible)
    fixed_focal = bundle_adjust(problem, huber_delta=None)
    free_focal = bundle_adjust(problem, huber_delta=None, optimize_focal=True, max_iterations=60)
    assert fixed_focal.final_cost > 1.0  # a wrong focal cannot explain the observations
    assert free_focal.final_cost < 1e-10
    assert free_focal.K[0, 0].item() == pytest.approx(K[0, 0].item(), rel=1e-6)


def test_huber_loss_resists_outliers():
    K, w2c, points, uv, visible = synthetic_problem(n_cams=10, n_points=200)
    generator = torch.Generator().manual_seed(3)
    corrupted = uv.clone()
    bad = torch.rand(uv.shape[:2], generator=generator) < 0.08
    corrupted[bad] += torch.randn(int(bad.sum()), 2, generator=generator, dtype=DT) * 25.0
    noisy_w2c, noisy_points = perturb(w2c, points, 0.01, 0.03)
    problem = as_problem(K, noisy_w2c, noisy_points, corrupted, visible)
    robust = bundle_adjust(problem, huber_delta=1.0, max_iterations=50)
    plain = bundle_adjust(problem, huber_delta=None, max_iterations=50)
    gt_c2w = invert_se3(w2c)
    ate_robust = pose_metrics(invert_se3(robust.w2c), gt_c2w)["ate"]
    ate_plain = pose_metrics(invert_se3(plain.w2c), gt_c2w)["ate"]
    assert ate_robust < 0.2 * ate_plain
    assert ate_robust < 5e-3


# ----------------------------------------------------------------------- depth prior


def depth_prior_for(
    problem: BAProblem, w2c, points, scales, affine=False, shifts=None
) -> DepthPrior:
    """Exact inverse depths, expressed in each camera's own arbitrary scale (and shift)."""
    ci, pi = problem.cam_index, problem.point_index
    z = transform_points(w2c[ci], points[pi][:, None])[:, 0, 2]
    shift = torch.zeros_like(scales) if shifts is None else shifts
    # The network's output q satisfies 1 / z = a q + b.
    inverse = (1.0 / z - shift[ci]) / scales[ci]
    params = torch.stack([scales, shift], dim=1) * 1.0
    valid = torch.ones_like(z, dtype=torch.bool)
    return DepthPrior(inverse, valid, params, affine)


def test_depth_prior_jacobians_match_autograd():
    K, w2c, points, uv, visible = synthetic_problem(n_cams=3, n_points=6)
    problem = as_problem(K, w2c, points, uv, visible)
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    ci, pi = problem.cam_index, problem.point_index
    scales = torch.tensor([0.7, 1.3, 2.1], dtype=DT)
    shifts = torch.tensor([0.02, -0.01, 0.03], dtype=DT)
    prior = depth_prior_for(problem, w2c, points, scales, True, shifts)
    prior.valid[2] = False
    params = prior.params + 0.05  # away from the exact solution
    _, cam = reprojection_residuals(R, t, points, K, ci, pi, problem.uv)
    J_pose, J_point, J_params = depth_jacobians(R, cam, params, prior, ci)

    def residual_of(pose_delta, point_delta, param_delta):
        dR = so3_exp(pose_delta[:, :3])
        R_new = dR @ R
        t_new = (dR @ t[..., None])[..., 0] + pose_delta[:, 3:]
        _, cam_new = reprojection_residuals(
            R_new, t_new, points + point_delta, K, ci, pi, problem.uv
        )
        return depth_residuals(cam_new, params + param_delta, prior, ci)

    zeros = (torch.zeros(3, 6, dtype=DT), torch.zeros(6, 3, dtype=DT), torch.zeros(3, 2, dtype=DT))
    auto_pose, auto_point, auto_params = torch.autograd.functional.jacobian(residual_of, zeros)
    obs = torch.arange(ci.shape[0])
    assert torch.allclose(auto_pose[obs, ci], J_pose, atol=1e-9)
    assert torch.allclose(auto_point[obs, pi], J_point, atol=1e-9)
    assert torch.allclose(auto_params[obs, ci], J_params, atol=1e-9)
    assert J_pose[2].abs().sum() == 0 and J_params[2].abs().sum() == 0  # the invalid sample


def test_bundle_adjustment_with_depth_prior_recovers_the_per_frame_scales():
    K, w2c, points, uv, visible = synthetic_problem(n_cams=8, n_points=150)
    noisy_w2c, noisy_points = perturb(w2c, points)
    problem = as_problem(K, noisy_w2c, noisy_points, uv, visible)
    scales = torch.linspace(0.4, 2.5, 8, dtype=DT)
    prior = depth_prior_for(problem, w2c, points, scales)
    prior.params[:, 0] *= 1.3  # wrong initial alignment
    problem.depth = prior
    result = bundle_adjust(problem, huber_delta=None, max_iterations=60)
    assert result.final_cost < 1e-10
    # The gauge scale is free: scales are recovered up to one common factor.
    ratio = result.depth_params[:, 0] / scales
    assert torch.allclose(ratio, ratio[0].expand(8), rtol=1e-5)
    assert result.depth_params[:, 1].abs().max() == 0  # the shift is not optimised
    metrics = pose_metrics(invert_se3(result.w2c), invert_se3(w2c))
    assert metrics["ate"] < 1e-6


def test_affine_depth_prior_recovers_scale_and_shift():
    K, w2c, points, uv, visible = synthetic_problem(n_cams=6, n_points=150)
    noisy_w2c, noisy_points = perturb(w2c, points, 0.01, 0.03)
    problem = as_problem(K, noisy_w2c, noisy_points, uv, visible)
    scales = torch.linspace(0.6, 1.8, 6, dtype=DT)
    shifts = torch.linspace(-0.03, 0.05, 6, dtype=DT)
    prior = depth_prior_for(problem, w2c, points, scales, True, shifts)
    prior.params[:, 0] *= 0.8
    prior.params[:, 1] += 0.02
    problem.depth = prior
    result = bundle_adjust(problem, huber_delta=None, max_iterations=80)
    assert result.final_cost < 1e-9
    assert pose_metrics(invert_se3(result.w2c), invert_se3(w2c))["ate"] < 1e-5


def test_depth_prior_sharpens_the_structure_of_a_short_baseline():
    """With a short baseline and noisy tracks, point depths are barely constrained.

    Relative depths from a monocular network constrain exactly that direction: the depth
    relief of the reconstruction becomes several times more accurate, even when the prior
    is weighted for a realistic network (10% depth error ~ 1 pixel).
    """
    K, w2c, points, uv, visible = synthetic_problem(
        n_cams=8, n_points=250, visibility=0.9, arc=0.04
    )
    generator = torch.Generator().manual_seed(7)
    noisy_uv = uv + 0.5 * torch.randn(uv.shape, generator=generator, dtype=DT)
    problem = as_problem(K, w2c, points, noisy_uv, visible)

    def relief_error(result) -> float:
        """Median relative depth error in the first camera, up to a global scale."""
        z = transform_points(result.w2c[0], result.points)[:, 2]
        z_true = transform_points(w2c[0], points)[:, 2]
        ratio = z / z_true
        return float((ratio / ratio.median() - 1.0).abs().median())

    plain = bundle_adjust(problem, max_iterations=50)
    problem.depth = depth_prior_for(problem, w2c, points, torch.ones(8, dtype=DT))
    guided = bundle_adjust(problem, max_iterations=50)
    assert relief_error(plain) > 0.03
    assert relief_error(guided) < 0.3 * relief_error(plain)


def test_sample_track_depths(tiny_still):
    seq = tiny_still
    gt = seq.gt.oracle.random_tracks(300, seed=0)
    tracks = Tracks(gt.uv, gt.visible)
    inverse, valid = sample_track_depths(tracks, seq.gt.depth, "scale")
    assert inverse.shape == valid.shape == (300, seq.num_frames)
    assert not (valid & ~tracks.visible).any()
    # Exact up to the interpolation of the depth map between pixel centres.
    relative = (inverse[valid].float() * gt.depth[valid] - 1.0).abs()
    assert relative.median() < 1e-3 and relative.max() < 0.06
    # An inverse-depth input gives the same samples.
    from_disparity, valid2 = sample_track_depths(tracks, 1.0 / seq.gt.depth, "disparity")
    assert torch.equal(valid, valid2)
    assert torch.allclose(from_disparity[valid], inverse[valid], rtol=1e-6)
    with pytest.raises(ValueError):
        sample_track_depths(tracks, seq.gt.depth, "metric")


# --------------------------------------------------------------------- full SfM


def tracks_from(uv, visible) -> Tracks:
    return Tracks(uv.to(torch.float32), visible)


def test_sfm_recovers_poses_and_structure_from_exact_tracks():
    K, w2c, points, uv, visible = synthetic_problem(n_cams=12, n_points=300, visibility=0.9)
    result = reconstruct(tracks_from(uv, visible), K.float(), SfMConfig())
    assert result.registered.all()
    assert result.valid.float().mean() > 0.9
    metrics = pose_metrics(result.c2w, invert_se3(w2c))
    assert metrics["ate"] < 2e-3 and metrics["rot_deg"] < 0.05
    # Structure agrees with the ground truth after the same similarity.
    sim_scale = metrics["scale"]
    depths = transform_points(result.w2c[0], result.points[result.valid])[:, 2]
    gt_depths = transform_points(w2c[0], points[result.valid])[:, 2]
    assert torch.allclose(depths * sim_scale, gt_depths, rtol=5e-3)
    assert result.reproj_error[result.valid].median() < 0.05
    # Scale gauge: the median depth of the points is 1.
    all_depths = transform_points(result.w2c[:, None], result.points[result.valid][None])[..., 2]
    assert all_depths.median().item() == pytest.approx(1.0, abs=0.1)


def test_sfm_flags_independently_moving_points_as_outliers():
    K, w2c, points, uv, visible = synthetic_problem(n_cams=12, n_points=300, visibility=1.0)
    n_moving = 60
    # The last points rise over time: their tracks are not rigid. (The motion is chosen
    # across the direction of the camera translation: a point moving *along* it is the
    # classical degenerate case, indistinguishable from a static point at another depth.)
    shift = torch.linspace(0.0, 0.8, w2c.shape[0], dtype=DT)
    direction = torch.tensor([0.2, 1.0, 0.0], dtype=DT)
    moving = points[-n_moving:, None, :] + shift[None, :, None] * direction
    cam = (w2c[None, :, :3, :3] @ moving[..., None])[..., 0] + w2c[None, :, :3, 3]
    uv = uv.clone()
    uv[-n_moving:], _ = project(K, cam)
    result = reconstruct(tracks_from(uv, visible), K.float(), SfMConfig())
    metrics = pose_metrics(result.c2w, invert_se3(w2c))
    assert metrics["ate"] < 5e-3 and metrics["rot_deg"] < 0.1
    static_valid = result.valid[:-n_moving].float().mean()
    moving_valid = result.valid[-n_moving:].float().mean()
    assert static_valid > 0.9
    assert moving_valid < 0.15
    assert result.reproj_error[-n_moving:].median() > 5 * result.reproj_error[:-n_moving].median()


def test_sfm_with_noisy_tracks_and_unknown_focal():
    K, w2c, _, uv, visible = synthetic_problem(n_cams=14, n_points=400, visibility=0.9)
    generator = torch.Generator().manual_seed(5)
    noisy_uv = uv + 0.3 * torch.randn(uv.shape, generator=generator, dtype=DT)
    K_guess = K.clone()
    K_guess[0, 0] *= 1.2
    K_guess[1, 1] *= 1.2
    result = reconstruct(
        tracks_from(noisy_uv, visible), K_guess.float(), SfMConfig(refine_focal=True)
    )
    assert result.K[0, 0].item() == pytest.approx(K[0, 0].item(), rel=0.03)
    metrics = pose_metrics(result.c2w, invert_se3(w2c))
    assert metrics["ate"] < 0.03 and metrics["rot_deg"] < 0.5


def test_sfm_fails_loudly_without_parallax():
    K, _, _, uv, visible = synthetic_problem(n_cams=6, n_points=100)
    static_uv = uv[:, :1].expand(-1, 6, -1)  # the camera never moves
    with pytest.raises(RuntimeError, match="initial frame pair"):
        reconstruct(tracks_from(static_uv, visible), K.float())


def test_sfm_on_ground_truth_tracks_of_the_benchmark(tiny_still):
    """End-to-end on a rendered sequence, using exact tracks from the oracle."""
    seq = tiny_still
    gt = seq.gt.oracle.random_tracks(600, seed=0)
    result = reconstruct(Tracks(gt.uv, gt.visible), seq.K(), SfMConfig())
    assert result.registered.all()
    metrics = pose_metrics(result.c2w.float(), seq.gt.c2w)
    assert metrics["ate"] < 0.01, metrics  # metres, in a 6 m room
    assert metrics["rot_deg"] < 0.2, metrics
    assert math.isfinite(metrics["scale"])
