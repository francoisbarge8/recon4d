import math

import pytest
import torch

from recon4d.geometry import (
    Intrinsics,
    Sim3,
    backproject_depth,
    batched_kabsch,
    camera_centers,
    invert_se3,
    look_at,
    make_se3,
    pixel_centers,
    pixel_rays,
    project,
    quat_multiply,
    quat_to_rotmat,
    reprojection_errors,
    rot6d_to_rotmat,
    rotation_angle,
    rotmat_to_quat,
    rotmat_to_rot6d,
    sample_bilinear,
    skew,
    so3_exp,
    so3_log,
    transform_points,
    triangulate_dlt,
    triangulation_angles,
    umeyama,
    unproject,
)

DT = torch.float64


def random_rotations(n: int) -> torch.Tensor:
    return quat_to_rotmat(torch.randn(n, 4, dtype=DT))


# ------------------------------------------------------------------------- rotations


def test_quat_to_rotmat_is_orthonormal():
    R = random_rotations(64)
    eye = torch.eye(3, dtype=DT).expand(64, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-12)
    assert torch.allclose(torch.det(R), torch.ones(64, dtype=DT), atol=1e-12)


def test_quat_rotmat_roundtrip():
    R = random_rotations(256)
    assert torch.allclose(quat_to_rotmat(rotmat_to_quat(R)), R, atol=1e-10)


def test_rotmat_to_quat_near_180_degrees():
    # Rotations by pi are the hard case for trace-based conversions.
    axes = torch.nn.functional.normalize(torch.randn(32, 3, dtype=DT), dim=-1)
    R = so3_exp(axes * (math.pi - 1e-7))
    assert torch.allclose(quat_to_rotmat(rotmat_to_quat(R)), R, atol=1e-9)


def test_quat_multiply_matches_matrix_product():
    a, b = torch.randn(16, 4, dtype=DT), torch.randn(16, 4, dtype=DT)
    a, b = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
    assert torch.allclose(
        quat_to_rotmat(quat_multiply(a, b)), quat_to_rotmat(a) @ quat_to_rotmat(b), atol=1e-12
    )


def test_so3_exp_log_roundtrip():
    omega = torch.randn(128, 3, dtype=DT)
    omega = omega / omega.norm(dim=-1, keepdim=True) * torch.rand(128, 1, dtype=DT) * 3.0
    assert torch.allclose(so3_log(so3_exp(omega)), omega, atol=1e-9)


def test_so3_exp_small_angle_is_accurate_and_differentiable():
    omega = torch.tensor([[1e-9, -2e-9, 5e-10], [0.0, 0.0, 0.0]], dtype=DT, requires_grad=True)
    R = so3_exp(omega)
    assert torch.allclose(R, torch.eye(3, dtype=DT) + skew(omega), atol=1e-15)
    R.sum().backward()
    assert torch.isfinite(omega.grad).all()


def test_so3_log_gradient_is_finite_at_identity():
    R = torch.eye(3, dtype=DT).requires_grad_(True)
    so3_log(R).sum().backward()
    assert torch.isfinite(R.grad).all()


def test_so3_exp_matches_matrix_exponential():
    omega = torch.randn(8, 3, dtype=DT)
    assert torch.allclose(so3_exp(omega), torch.linalg.matrix_exp(skew(omega)), atol=1e-12)


def test_rot6d_roundtrip():
    R = random_rotations(32)
    assert torch.allclose(rot6d_to_rotmat(rotmat_to_rot6d(R)), R, atol=1e-12)


def test_rotation_angle():
    angles = torch.tensor([0.0, 1e-6, 0.3, 1.5, math.pi - 1e-6], dtype=DT)
    axis = torch.tensor([0.36, 0.48, 0.8], dtype=DT)
    assert torch.allclose(rotation_angle(so3_exp(angles[:, None] * axis)), angles, atol=1e-9)


# ---------------------------------------------------------------------------- camera


def test_project_unproject_roundtrip():
    K = Intrinsics.from_fov(64, 48, 70.0).matrix(DT)
    uv = torch.rand(100, 2, dtype=DT) * torch.tensor([64.0, 48.0], dtype=DT)
    depth = torch.rand(100, dtype=DT) * 5.0 + 0.5
    uv_back, z = project(K, unproject(K, uv, depth))
    assert torch.allclose(uv_back, uv, atol=1e-10)
    assert torch.allclose(z, depth, atol=1e-12)


def test_intrinsics_from_fov_and_scaling():
    intr = Intrinsics.from_fov(200, 100, 90.0)
    assert intr.fx == pytest.approx(100.0)
    assert (intr.cx, intr.cy) == (100.0, 50.0)
    half = intr.scaled(0.5)
    assert (half.width, half.height, half.fx, half.cx) == (100, 50, pytest.approx(50.0), 50.0)


def test_invert_se3():
    T = make_se3(random_rotations(8), torch.randn(8, 3, dtype=DT))
    eye = torch.eye(4, dtype=DT).expand(8, 4, 4)
    assert torch.allclose(T @ invert_se3(T), eye, atol=1e-12)


def test_look_at_points_optical_axis_at_target_with_y_down():
    eye = torch.tensor([2.0, 1.5, 3.0], dtype=DT)
    target = torch.tensor([0.0, 0.5, 0.0], dtype=DT)
    c2w = look_at(eye, target)
    R = c2w[:3, :3]
    assert torch.allclose(R.T @ R, torch.eye(3, dtype=DT), atol=1e-12)
    assert torch.det(R) == pytest.approx(1.0)
    # The target projects onto the principal point ...
    K = Intrinsics.from_fov(64, 48, 60.0).matrix(DT)
    uv, z = project(K, transform_points(invert_se3(c2w), target[None]))
    assert torch.allclose(uv[0], torch.tensor([32.0, 24.0], dtype=DT), atol=1e-9)
    assert z[0] > 0
    # ... and a point above the target appears higher in the image (smaller v).
    above, _ = project(K, transform_points(invert_se3(c2w), (target + torch.tensor([0, 0.2, 0]))[None]))
    assert above[0, 1] < uv[0, 1]
    assert torch.allclose(camera_centers(invert_se3(c2w)), eye, atol=1e-12)


def test_pixel_rays_reach_backprojected_points():
    intr = Intrinsics.from_fov(16, 12, 60.0)
    K = intr.matrix(DT)
    c2w = look_at(torch.tensor([1.0, 2.0, 3.0], dtype=DT), torch.zeros(3, dtype=DT))
    depth = torch.rand(12, 16, dtype=DT) + 1.0
    points = backproject_depth(K, c2w, depth)
    origins, dirs = pixel_rays(K, c2w, pixel_centers(12, 16, dtype=DT))
    assert torch.allclose(origins + depth[..., None] * dirs, points, atol=1e-12)
    # Re-projecting recovers the pixel centres and the depth.
    uv, z = project(K, transform_points(invert_se3(c2w), points.reshape(-1, 3)))
    assert torch.allclose(uv.reshape(12, 16, 2), pixel_centers(12, 16, dtype=DT), atol=1e-9)
    assert torch.allclose(z.reshape(12, 16), depth, atol=1e-12)


def test_sample_bilinear_uses_half_integer_pixel_centres():
    image = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    centers = pixel_centers(3, 4).reshape(-1, 2)
    assert torch.allclose(sample_bilinear(image, centers)[:, 0], image.reshape(-1), atol=1e-5)
    # Halfway between two horizontally adjacent pixels: their mean.
    mid = sample_bilinear(image, torch.tensor([[1.0, 0.5]]))
    assert mid.item() == pytest.approx(0.5)


# ------------------------------------------------------------------------- alignment


def test_umeyama_recovers_similarity():
    src = torch.randn(50, 3, dtype=DT)
    R = random_rotations(1)[0]
    t = torch.randn(3, dtype=DT)
    dst = 2.5 * src @ R.T + t
    sim = umeyama(src, dst)
    assert sim.scale.item() == pytest.approx(2.5)
    assert torch.allclose(sim.R, R, atol=1e-10)
    assert torch.allclose(sim.t, t, atol=1e-10)
    assert torch.allclose(sim.apply(src), dst, atol=1e-10)
    assert torch.allclose(sim.inverse().apply(dst), src, atol=1e-10)


def test_umeyama_never_returns_a_reflection():
    src = torch.randn(30, 3, dtype=DT)
    dst = src * torch.tensor([1.0, 1.0, -1.0], dtype=DT)  # mirrored copy
    sim = umeyama(src, dst)
    assert torch.det(sim.R).item() == pytest.approx(1.0)


def test_umeyama_weights_ignore_outliers():
    src = torch.randn(40, 3, dtype=DT)
    R = random_rotations(1)[0]
    dst = src @ R.T
    dst[:5] += 10.0
    weights = torch.ones(40, dtype=DT)
    weights[:5] = 0.0
    sim = umeyama(src, dst, weights=weights, with_scale=False)
    assert torch.allclose(sim.R, R, atol=1e-10)


def test_sim3_apply_to_c2w_keeps_poses_rigid():
    c2w = make_se3(random_rotations(6), torch.randn(6, 3, dtype=DT))
    sim = Sim3(torch.tensor(1.7, dtype=DT), random_rotations(1)[0], torch.randn(3, dtype=DT))
    moved = sim.apply_to_c2w(c2w)
    R = moved[:, :3, :3]
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3, dtype=DT).expand(6, 3, 3), atol=1e-12)
    assert torch.allclose(moved[:, :3, 3], sim.apply(c2w[:, :3, 3]), atol=1e-12)


def test_batched_kabsch():
    src = torch.randn(4, 20, 3, dtype=DT)
    R = random_rotations(4)
    t = torch.randn(4, 3, dtype=DT)
    dst = src @ R.transpose(1, 2) + t[:, None]
    weights = torch.rand(4, 20, dtype=DT)
    weights[3] = 0.0  # an empty batch falls back to the identity
    R_est, t_est = batched_kabsch(src, dst, weights)
    assert torch.allclose(R_est[:3], R[:3], atol=1e-9)
    assert torch.allclose(t_est[:3], t[:3], atol=1e-9)
    assert torch.allclose(R_est[3], torch.eye(3, dtype=DT))
    assert torch.allclose(t_est[3], torch.zeros(3, dtype=DT))


# --------------------------------------------------------------------- triangulation


def _toy_views(n_views: int = 5):
    K = Intrinsics.from_fov(64, 48, 60.0).matrix(DT)
    eyes = torch.stack(
        [torch.tensor([math.sin(a) * 3.0, 0.3 * a, math.cos(a) * 3.0], dtype=DT)
         for a in torch.linspace(-0.5, 0.5, n_views).tolist()]
    )
    c2w = torch.stack([look_at(e, torch.zeros(3, dtype=DT)) for e in eyes])
    return K, invert_se3(c2w)


def test_triangulate_dlt_exact_and_respects_visibility():
    K, w2c = _toy_views()
    points = torch.rand(40, 3, dtype=DT) - 0.5
    uv, _ = project(K, transform_points(w2c[None], points[:, None, None, :])[:, :, 0])
    visible = torch.rand(40, 5) > 0.3
    visible[:, :2] = True
    visible[0] = False
    visible[0, 0] = True  # a single observation cannot be triangulated
    # Corrupt the observations that are flagged invisible: they must be ignored.
    uv_noisy = torch.where(visible[..., None], uv, uv + 50.0)
    est, valid = triangulate_dlt(K, w2c, uv_noisy, visible)
    assert not valid[0]
    assert valid[1:].all()
    assert torch.allclose(est[1:], points[1:], atol=1e-8)
    errors, depths = reprojection_errors(K, w2c, est[1:], uv[1:])
    assert errors.max() < 1e-6
    assert (depths > 0).all()


def test_triangulation_angles():
    K, w2c = _toy_views(3)
    del K
    points = torch.zeros(1, 3, dtype=DT)
    visible = torch.ones(1, 3, dtype=torch.bool)
    centers = camera_centers(w2c)
    expected = torch.acos(
        torch.nn.functional.normalize(centers[0], dim=0)
        @ torch.nn.functional.normalize(centers[2], dim=0)
    )
    assert triangulation_angles(w2c, points, visible).item() == pytest.approx(expected.item())
