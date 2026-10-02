"""The rasterizer is checked against a brute-force per-pixel reference and by gradcheck."""

import math

import pytest
import torch

from recon4d.gaussians import (
    RasterSettings,
    covariance_3d,
    eval_sh,
    project_gaussians,
    rasterize,
    rgb_to_sh,
    sh_basis,
    sh_to_rgb,
)
from recon4d.geometry import Intrinsics, invert_se3, look_at, project, transform_points

DT = torch.float64
W, H = 40, 28


def camera(dtype=DT):
    K = Intrinsics.from_fov(W, H, 60.0).matrix(dtype)
    eye = torch.tensor([0.3, 0.4, -3.0], dtype=dtype)
    return K, invert_se3(look_at(eye, torch.zeros(3, dtype=dtype)))


def random_gaussians(n: int, seed: int = 0, dtype=DT):
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(n, 3, generator=g, dtype=dtype) - 0.5) * torch.tensor(
        [3.0, 2.0, 2.0], dtype=dtype
    )
    quats = torch.randn(n, 4, generator=g, dtype=dtype)
    scales = torch.exp(torch.rand(n, 3, generator=g, dtype=dtype) * 2.2 - 3.2)
    opacities = torch.rand(n, generator=g, dtype=dtype) * 0.95 + 0.02
    colors = torch.rand(n, 3, generator=g, dtype=dtype)
    return means, quats, scales, opacities, colors


def reference_rasterize(projection, opacities, features, background, settings):
    """Brute force: every Gaussian is evaluated on every pixel, front to back, no tiles.

    This is the compositing equation written down directly, sharing nothing with the
    binning / bucketing machinery of the real implementation.
    """
    rows = torch.arange(H, dtype=DT)[:, None] + 0.5
    cols = torch.arange(W, dtype=DT)[None, :] + 0.5
    image = torch.zeros(H, W, features.shape[1], dtype=DT)
    alpha_map = torch.zeros(H, W, dtype=DT)
    transmittance = torch.ones(H, W, dtype=DT)
    for g in torch.argsort(projection.depths).tolist():
        if projection.radii[g] <= 0:
            continue
        dx = cols - projection.means2d[g, 0]
        dy = rows - projection.means2d[g, 1]
        a, b, c = projection.conics[g]
        power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        alpha = (opacities[g] * torch.exp(power)).clamp_max(settings.alpha_max)
        alpha = torch.where((power <= 0) & (alpha >= settings.alpha_min), alpha, 0.0)
        weight = alpha * transmittance
        image += weight[..., None] * features[g]
        alpha_map += weight
        transmittance = transmittance * (1.0 - alpha)
    return image + (1.0 - alpha_map)[..., None] * background, alpha_map


def render(means, quats, scales, opacities, colors, K, w2c, settings=None, background=None):
    settings = settings or RasterSettings()
    projection = project_gaussians(means, quats, scales, K, w2c, W, H, settings, opacities)
    return rasterize(projection, opacities, colors, W, H, settings, background), projection


# ------------------------------------------------------------------------ projection


def test_projection_matches_pinhole_and_monte_carlo_covariance():
    K, w2c = camera()
    means, quats, scales, _, _ = random_gaussians(6, seed=1)
    scales = scales * 0.05  # small Gaussians: the affine approximation is then accurate
    settings = RasterSettings(cov_blur=0.0)
    projection = project_gaussians(means, quats, scales, K, w2c, W, H, settings)
    uv, z = project(K, transform_points(w2c, means))
    assert torch.allclose(projection.means2d, uv, atol=1e-10)
    assert torch.allclose(projection.depths, z, atol=1e-12)

    cov3d = covariance_3d(quats, scales)
    generator = torch.Generator().manual_seed(0)
    for i in range(6):
        chol = torch.linalg.cholesky(cov3d[i])
        samples = means[i] + torch.randn(400000, 3, generator=generator, dtype=DT) @ chol.T
        uv_samples, _ = project(K, transform_points(w2c, samples))
        empirical = torch.cov(uv_samples.T)
        a, b, c = projection.conics[i]
        predicted = torch.linalg.inv(torch.stack([torch.stack([a, b]), torch.stack([b, c])]))
        # Monte Carlo noise is about sigma^2 / sqrt(n): compare relative to the matrix scale.
        assert (empirical - predicted).abs().max() < 0.02 * predicted.abs().max()


def test_gaussians_behind_or_outside_the_camera_are_culled():
    K, w2c = camera()
    means = torch.tensor([[0.0, 0.0, -6.0], [50.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=DT)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 3, dtype=DT)
    scales = torch.full((3, 3), 0.05, dtype=DT)
    projection = project_gaussians(means, quats, scales, K, w2c, W, H)
    assert projection.visible.tolist() == [False, False, True]


def test_opacity_aware_radius_culls_invisible_gaussians():
    K, w2c = camera()
    means = torch.zeros(3, 3, dtype=DT)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 3, dtype=DT)
    scales = torch.full((3, 3), 0.2, dtype=DT)
    opacities = torch.tensor([1.0, 0.05, 0.003], dtype=DT)
    radii = project_gaussians(means, quats, scales, K, w2c, W, H, opacities=opacities).radii
    assert radii[0] > radii[1] > 0
    assert radii[2] == 0, "opacity below alpha_min can never contribute"


# --------------------------------------------------------------------- rasterization


@pytest.mark.parametrize("tile_size", [4, 8, 16])
def test_matches_brute_force_reference(tile_size):
    K, w2c = camera()
    means, quats, scales, opacities, colors = random_gaussians(60)
    settings = RasterSettings(tile_size=tile_size)
    background = torch.tensor([0.1, 0.2, 0.3], dtype=DT)
    out, projection = render(means, quats, scales, opacities, colors, K, w2c, settings, background)
    assert projection.visible.sum() > 40
    expected, expected_alpha = reference_rasterize(
        projection, opacities, colors, background, settings
    )
    assert torch.allclose(out.image, expected, atol=1e-10)
    assert torch.allclose(out.alpha, expected_alpha, atol=1e-10)
    assert out.alpha.max() > 0.9 and out.alpha.min() < 0.5


def test_image_does_not_depend_on_tile_size_or_chunking():
    K, w2c = camera()
    params = random_gaussians(300, seed=3)
    images = [
        render(*params, K, w2c, RasterSettings(tile_size=ts, chunk_elements=chunk))[0].image
        for ts, chunk in ((4, 6_000_000), (8, 6_000_000), (16, 6_000_000), (8, 2_000))
    ]
    for image in images[1:]:
        assert torch.allclose(image, images[0], atol=1e-10)


def test_float32_agrees_with_float64():
    K, w2c = camera()
    params = random_gaussians(150, seed=5)
    out64 = render(*params, K, w2c)[0].image
    K32, w2c32 = K.float(), w2c.float()
    out32 = render(*[p.float() for p in params], K32, w2c32)[0].image
    assert out32.dtype == torch.float32
    assert (out32.double() - out64).abs().max() < 2e-3


def test_single_gaussian_peak_and_mass():
    K, w2c = camera()
    means = torch.tensor([[0.0, 0.0, 0.0]], dtype=DT)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=DT)
    scales = torch.full((1, 3), 0.25, dtype=DT)
    opacities = torch.tensor([0.6], dtype=DT)
    colors = torch.tensor([[1.0, 0.5, 0.25]], dtype=DT)
    out, projection = render(means, quats, scales, opacities, colors, K, w2c)
    # Value at the pixel containing the centre: the Gaussian evaluated at that pixel centre.
    a, b, c = projection.conics[0]
    col, row = int(projection.means2d[0, 0]), int(projection.means2d[0, 1])
    dx, dy = col + 0.5 - projection.means2d[0, 0], row + 0.5 - projection.means2d[0, 1]
    expected = 0.6 * torch.exp(-0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy)
    assert 0.5 < expected <= 0.6
    assert out.alpha[row, col].item() == pytest.approx(expected.item(), rel=1e-12)
    assert out.alpha.max() == out.alpha[row, col]
    assert torch.allclose(out.image[row, col], colors[0] * out.alpha[row, col])
    # Total mass of a 2D Gaussian: opacity * 2 pi sqrt(det Sigma), up to the 1/255 cut-off.
    det_sigma = 1.0 / (a * c - b * b)
    assert out.alpha.sum().item() == pytest.approx(
        0.6 * 2 * math.pi * math.sqrt(det_sigma), rel=0.02
    )


def test_front_gaussian_occludes_back_gaussian():
    K, w2c = camera()
    means = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=DT)  # camera sits at z = -3
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=DT)
    scales = torch.full((2, 3), 0.4, dtype=DT)
    opacities = torch.tensor([0.99, 0.99], dtype=DT)
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=DT)
    out, projection = render(means, quats, scales, opacities, colors, K, w2c)
    assert projection.depths[1] < projection.depths[0]
    centre = out.image[H // 2, W // 2]
    assert centre[2] > 0.9 and centre[0] < 0.1  # blue (near) hides red (far)
    # Swapping the depths swaps the outcome.
    swapped = render(means.flip(0), quats, scales, opacities, colors, K, w2c)[0]
    assert swapped.image[H // 2, W // 2, 0] > 0.9


def test_background_and_empty_scene():
    K, w2c = camera()
    empty = [torch.zeros(0, k, dtype=DT) for k in (3, 4, 3)]
    background = torch.tensor([0.2, 0.4, 0.6], dtype=DT)
    out, _ = render(
        *empty, torch.zeros(0, dtype=DT), torch.zeros(0, 3, dtype=DT), K, w2c, None, background
    )
    assert out.image.shape == (H, W, 3)
    assert torch.allclose(out.image, background.expand(H, W, 3))
    assert (out.alpha == 0).all()


def test_non_multiple_image_size_and_arbitrary_feature_channels():
    K, w2c = camera()
    means, quats, scales, opacities, _ = random_gaussians(80, seed=2)
    features = torch.rand(80, 7, dtype=DT)
    settings = RasterSettings(tile_size=16)  # 40 x 28 is not a multiple of 16
    projection = project_gaussians(means, quats, scales, K, w2c, W, H, settings, opacities)
    out = rasterize(projection, opacities, features, W, H, settings)
    assert out.image.shape == (H, W, 7)
    reference = rasterize(projection, opacities, features, W, H, RasterSettings(tile_size=4))
    assert torch.allclose(out.image, reference.image, atol=1e-10)


def test_max_per_tile_keeps_the_front_most_gaussians():
    K, w2c = camera()
    params = random_gaussians(400, seed=7)
    exact = render(*params, K, w2c)[0].image
    capped = render(*params, K, w2c, RasterSettings(max_per_tile=4))[0].image
    generous = render(*params, K, w2c, RasterSettings(max_per_tile=10_000))[0].image
    assert torch.allclose(generous, exact, atol=1e-12)
    assert not torch.allclose(capped, exact, atol=1e-3)


def test_antialias_dims_subpixel_gaussians():
    K, w2c = camera()
    means = torch.zeros(1, 3, dtype=DT)
    quats = torch.tensor([[1.0, 0, 0, 0]], dtype=DT)
    scales = torch.full((1, 3), 0.002, dtype=DT)  # far smaller than a pixel
    opacities = torch.tensor([0.9], dtype=DT)
    colors = torch.ones(1, 3, dtype=DT)
    plain = render(means, quats, scales, opacities, colors, K, w2c)[0].alpha.sum()
    filtered = render(
        means, quats, scales, opacities, colors, K, w2c, RasterSettings(antialias=True)
    )[0].alpha.sum()
    assert filtered < 0.05 * plain


# ------------------------------------------------------------------------- gradients


def test_gradcheck_all_inputs():
    K, w2c = camera()
    means, quats, scales, opacities, colors = random_gaussians(12, seed=4)
    # Keep opacities away from the 0.99 clamp, whose kink finite differences cannot see.
    opacities = opacities * 0.8
    log_scales = scales.log()
    target = torch.rand(H, W, 3, dtype=DT, generator=torch.Generator().manual_seed(1))
    inputs = [t.clone().requires_grad_(True) for t in (means, quats, log_scales, opacities, colors)]

    def loss(means, quats, log_scales, opacities, colors):
        settings = RasterSettings()
        projection = project_gaussians(
            means, quats, log_scales.exp(), K, w2c, W, H, settings, opacities
        )
        out = rasterize(projection, opacities, colors, W, H, settings)
        return ((out.image - target) ** 2).sum() + out.alpha.sum()

    assert torch.autograd.gradcheck(loss, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)


def test_gradient_flows_to_camera_pose_and_intrinsics():
    K, w2c = camera()
    params = random_gaussians(20, seed=6)
    K = K.clone().requires_grad_(True)
    w2c = w2c.clone().requires_grad_(True)
    out = render(*params, K, w2c)[0]
    out.image.sum().backward()
    assert torch.isfinite(w2c.grad).all() and w2c.grad[:3].abs().sum() > 0
    assert torch.isfinite(K.grad).all() and K.grad[0, 0].abs() > 0


def test_culled_gaussians_receive_zero_gradient_not_nan():
    K, w2c = camera()
    means = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -9.0]], dtype=DT, requires_grad=True)
    quats = torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=DT, requires_grad=True)
    log_scales = torch.full((2, 3), -2.0, dtype=DT, requires_grad=True)
    opacities = torch.tensor([0.5, 0.5], dtype=DT, requires_grad=True)
    colors = torch.rand(2, 3, dtype=DT, requires_grad=True)
    out = render(means, quats, log_scales.exp(), opacities, colors, K, w2c)[0]
    out.image.sum().backward()
    for tensor in (means, quats, log_scales, opacities, colors):
        assert torch.isfinite(tensor.grad).all()
        assert (tensor.grad[1] == 0).all()
    assert means.grad[0].abs().sum() > 0


# ------------------------------------------------------------- spherical harmonics


def _fibonacci_sphere(n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=DT) + 0.5
    polar = torch.acos(1.0 - 2.0 * i / n)
    azimuth = math.pi * (1.0 + 5.0**0.5) * i
    return torch.stack(
        [
            torch.sin(polar) * torch.cos(azimuth),
            torch.sin(polar) * torch.sin(azimuth),
            torch.cos(polar),
        ],
        dim=-1,
    )


def test_sh_basis_is_orthonormal():
    dirs = _fibonacci_sphere(20000)
    basis = sh_basis(3, dirs)
    assert basis.shape == (20000, 16)
    gram = basis.T @ basis * (4.0 * math.pi / dirs.shape[0])
    assert torch.allclose(gram, torch.eye(16, dtype=DT), atol=2e-3)


def test_sh_degree_zero_is_view_independent_and_roundtrips_rgb():
    rgb = torch.rand(10, 3, dtype=DT)
    coeffs = torch.zeros(10, 16, 3, dtype=DT)
    coeffs[:, 0] = rgb_to_sh(rgb)
    dirs = _fibonacci_sphere(10)
    for degree in range(4):
        assert torch.allclose(eval_sh(degree, coeffs, dirs), rgb, atol=1e-12)
    assert torch.allclose(sh_to_rgb(rgb_to_sh(rgb)), rgb)


def test_sh_higher_degrees_add_view_dependence_and_respect_active_degree():
    coeffs = torch.randn(4, 16, 3, dtype=DT) * 0.2
    dirs = _fibonacci_sphere(4)
    low, high = eval_sh(1, coeffs, dirs), eval_sh(3, coeffs, dirs)
    assert not torch.allclose(low, high)
    truncated = coeffs.clone()
    truncated[:, 4:] = 0.0
    assert torch.allclose(eval_sh(3, truncated, dirs), low)
    with pytest.raises(ValueError):
        sh_basis(4, dirs)
