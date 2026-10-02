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
from recon4d.geometry import Intrinsics, invert_se3, look_at, project, so3_exp, transform_points
from recon4d.metrics import pose_metrics
from recon4d.types import Tracks

DT = torch.float64
W, H = 160, 120


def synthetic_problem(n_cams: int = 8, n_points: int = 120, seed: int = 0, visibility: float = 0.8):
    """Cameras on an arc looking at a random point cloud, with exact observations."""
    generator = torch.Generator().manual_seed(seed)
    K = Intrinsics.from_fov(W, H, 60.0).matrix(DT)
    angles = torch.linspace(-0.45, 0.45, n_cams, dtype=DT)
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
