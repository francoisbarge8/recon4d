"""Point-cloud fusion and depth alignment."""

import pytest
import torch

from recon4d.frontend.tracking import grid_queries
from recon4d.fusion import (
    backproject_frames,
    downsample_to,
    multiview_consistency,
    voxel_downsample,
)
from recon4d.fusion.depth_align import (
    DepthAlignConfig,
    _bilinear_weights,
    align_depth_maps,
    fit_correction_field,
    fit_scale,
    fit_scale_shift,
)
from recon4d.geometry import invert_se3, pixel_centers
from recon4d.metrics import depth_metrics, nearest_distances
from recon4d.types import Tracks

# ----------------------------------------------------------------------- point clouds


def test_voxel_downsample_returns_centroids():
    points = torch.tensor(
        [[0.1, 0.1, 0.1], [0.3, 0.1, 0.1], [1.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], dtype=torch.float64
    )
    colors = torch.tensor(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float64
    )
    out, out_colors = voxel_downsample(points, 1.0, colors)
    assert out.shape == (3, 3)
    merged = out[(out - torch.tensor([0.2, 0.1, 0.1], dtype=torch.float64)).norm(dim=1) < 1e-9]
    assert merged.shape[0] == 1
    index = (out - torch.tensor([0.2, 0.1, 0.1], dtype=torch.float64)).norm(dim=1).argmin()
    assert torch.allclose(out_colors[index], torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64))
    empty, _ = voxel_downsample(torch.zeros(0, 3), 1.0)
    assert empty.shape == (0, 3)


def test_downsample_to_reaches_the_target_count():
    points = torch.rand(20000, 3)
    out, _, voxel = downsample_to(points, 1500)
    assert 0.6 * 1500 <= out.shape[0] <= 1500 and voxel > 0
    same, _, voxel = downsample_to(points[:100], 1500)
    assert same.shape[0] == 100 and voxel == 0.0


def test_backprojected_depth_lies_on_the_true_surface(tiny_still):
    seq = tiny_still
    gt = seq.gt
    points, colors = backproject_frames(gt.depth, seq.K(), gt.c2w, seq.images, stride=2)
    assert points.shape == colors.shape and points.shape[0] == 8 * 24 * 32
    surface = gt.oracle.surface_cloud(n_static=60000, min_views=1).static
    assert nearest_distances(points, surface).median() < 0.03  # sampling density of the cloud
    # A mask restricts the back-projection.
    mask = torch.zeros_like(gt.depth, dtype=torch.bool)
    mask[:, :10] = True
    assert backproject_frames(gt.depth, seq.K(), gt.c2w, mask=mask)[0].shape[0] == 8 * 10 * 64


def test_multiview_consistency_counts_agreeing_views(tiny_still):
    seq = tiny_still
    gt = seq.gt
    w2c = invert_se3(gt.c2w)
    count = multiview_consistency(gt.depth, seq.K(), w2c, 3, [2, 4], rel_tolerance=0.02)
    assert count.shape == (seq.height, seq.width)
    assert (count == 2).float().mean() > 0.6
    # A wrong depth is not confirmed by the other views.
    corrupted = gt.depth.clone()
    corrupted[3, 20:30, 20:40] *= 1.5
    bad = multiview_consistency(corrupted, seq.K(), w2c, 3, [2, 4], rel_tolerance=0.02)
    assert (bad[20:30, 20:40] == 0).float().mean() > 0.9
    # Views flagged invalid cannot confirm anything.
    none_valid = torch.zeros_like(gt.depth, dtype=torch.bool)
    assert multiview_consistency(gt.depth, seq.K(), w2c, 3, [2, 4], valid=none_valid).sum() == 0


# ------------------------------------------------------------------- depth alignment


def test_robust_scale_and_scale_shift_fits():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(500, generator=generator, dtype=torch.float64) + 0.5
    y = 2.5 * x
    y[:40] *= 3.0  # 8% gross outliers
    assert fit_scale(x, y, 0.05, 10).item() == pytest.approx(2.5, rel=1e-3)
    y2 = 1.7 * x - 0.2
    y2[:40] += 5.0
    a, b = fit_scale_shift(x, y2, 0.05, 20)
    assert a.item() == pytest.approx(1.7, rel=2e-2) and b.item() == pytest.approx(-0.2, abs=3e-2)


def static_tracks(seq, stride: int = 4) -> tuple[Tracks, torch.Tensor, torch.Tensor]:
    """Exact tracks of a pixel grid seeded in every frame, with their 3D points.

    Seeding every frame gives each image a uniform coverage of sparse depths, as a good
    tracker would.
    """
    grid = grid_queries(seq.height, seq.width, stride, margin=2)
    frames = torch.arange(seq.num_frames).repeat_interleave(len(grid))
    gt = seq.gt.oracle.tracks(frames, grid.repeat(seq.num_frames, 1))
    return Tracks(gt.uv, gt.visible), gt.xyz[:, 0], gt.visible


def test_alignment_recovers_per_frame_scales(tiny_still):
    seq = tiny_still
    gt = seq.gt
    tracks, points, inlier = static_tracks(seq)
    scales = torch.tensor([0.3, 2.0, 1.0, 0.7, 5.0, 1.3, 0.9, 1.1])
    pred = gt.depth * scales[:, None, None]
    w2c = invert_se3(gt.c2w)
    out = align_depth_maps(pred, tracks, points, inlier, w2c, DepthAlignConfig(grid=None))
    # Limited by the interpolation of the depth map under the sparse points.
    assert depth_metrics(out.depth, gt.depth)["abs_rel"] < 1e-3
    assert out.valid.all() and (out.num_points > 50).all()
    assert out.residual.max() < 1e-3


def test_alignment_of_affine_invariant_disparity(tiny_still):
    seq = tiny_still
    gt = seq.gt
    tracks, points, inlier = static_tracks(seq)
    a = torch.linspace(0.5, 2.0, 8)[:, None, None]
    b = torch.linspace(0.05, 0.3, 8)[:, None, None]
    disparity = a / gt.depth + b
    cfg = DepthAlignConfig(kind="disparity", grid=None)
    out = align_depth_maps(disparity, tracks, points, inlier, invert_se3(gt.c2w), cfg)
    assert depth_metrics(out.depth, gt.depth)["abs_rel"] < 2e-3
    with pytest.raises(ValueError):
        align_depth_maps(
            disparity, tracks, points, inlier, invert_se3(gt.c2w), DepthAlignConfig(kind="x")
        )


def test_correction_field_removes_smooth_distortion(tiny_still):
    seq = tiny_still
    gt = seq.gt
    tracks, points, inlier = static_tracks(seq, stride=3)
    # A different low-frequency multiplicative error in every frame, drawn from the model
    # the correction field can represent (bilinear interpolation of a coarse grid).
    generator = torch.Generator().manual_seed(0)
    centers = pixel_centers(seq.height, seq.width, dtype=torch.float64).reshape(-1, 2)
    weights = _bilinear_weights(centers, seq.height, seq.width, (4, 5))
    nodes = 0.15 * torch.randn(8, 20, generator=generator, dtype=torch.float64)
    distortion = (nodes @ weights.T).reshape(8, seq.height, seq.width).float()
    pred = gt.depth * torch.exp(distortion) * 1.8
    w2c = invert_se3(gt.c2w)
    plain = align_depth_maps(pred, tracks, points, inlier, w2c, DepthAlignConfig(grid=None))
    corrected = align_depth_maps(pred, tracks, points, inlier, w2c, DepthAlignConfig(grid=(4, 5)))
    error_plain = depth_metrics(plain.depth, gt.depth)["abs_rel"]
    error_corrected = depth_metrics(corrected.depth, gt.depth)["abs_rel"]
    assert error_plain > 0.05
    assert error_corrected < 0.25 * error_plain
    assert corrected.residual.max() < plain.residual.min()


def test_correction_field_is_flat_without_evidence():
    cfg = DepthAlignConfig(grid=(3, 4))
    uv = torch.rand(200, 2, dtype=torch.float64) * torch.tensor([64.0, 48.0], dtype=torch.float64)
    field = fit_correction_field(uv, torch.zeros(200, dtype=torch.float64), 48, 64, cfg)
    assert field.shape == (3, 4) and field.abs().max() < 1e-12
    # A constant offset is reproduced everywhere (up to the small prior towards zero).
    field = fit_correction_field(uv, torch.full((200,), 0.2, dtype=torch.float64), 48, 64, cfg)
    assert torch.allclose(field, torch.full((3, 4), 0.2, dtype=torch.float64), atol=0.02)


def test_under_constrained_frames_borrow_from_their_neighbours(tiny_still):
    seq = tiny_still
    gt = seq.gt
    tracks, points, inlier = static_tracks(seq)
    inlier = inlier.clone()
    inlier[:, 4] = False  # no sparse depth at all in frame 4
    pred = gt.depth * 2.0
    out = align_depth_maps(
        pred, tracks, points, inlier, invert_se3(gt.c2w), DepthAlignConfig(grid=None)
    )
    assert out.num_points[4] == 0
    assert torch.allclose(out.depth[4], gt.depth[4], rtol=1e-3)
    with pytest.raises(RuntimeError):
        align_depth_maps(pred, tracks, points, torch.zeros_like(inlier), invert_se3(gt.c2w))
