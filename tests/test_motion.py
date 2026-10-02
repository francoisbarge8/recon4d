"""Motion bases, their initialisation, and motion segmentation."""

import torch

from recon4d.frontend.flow import OracleFlow
from recon4d.frontend.motion_seg import (
    MotionSegConfig,
    dynamic_queries,
    flow_residuals,
    label_tracks,
    mask_fraction,
    motion_masks,
    rigid_flow,
)
from recon4d.gaussians.init import fit_motion_bases
from recon4d.gaussians.motion import MotionBases, init_motion_bases, kmeans
from recon4d.geometry import invert_se3, so3_exp
from recon4d.types import Tracks

# ---------------------------------------------------------------------- motion bases


def rigid_trajectory(
    n_frames: int, axis, rate: float, velocity
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotations ``(T, 3, 3)`` and translations ``(T, 3)`` of a body spinning and drifting."""
    times = torch.arange(n_frames, dtype=torch.float32)
    R = so3_exp(times[:, None] * rate * torch.tensor(axis, dtype=torch.float32))
    t = times[:, None] * torch.tensor(velocity, dtype=torch.float32)
    return R, t


def two_bodies(n_frames: int = 20, n_points: int = 80, seed: int = 0):
    """Two rigid point clouds with different motions; returns positions and body labels."""
    generator = torch.Generator().manual_seed(seed)
    local = torch.randn(2 * n_points, 3, generator=generator) * 0.2
    body = torch.arange(2 * n_points) >= n_points
    centers = torch.tensor([[-1.0, 0.0, 0.0], [1.5, 0.3, 0.0]])
    motions = (
        rigid_trajectory(n_frames, (0.0, 1.0, 0.0), 0.25, (0.05, 0.0, 0.02)),
        rigid_trajectory(n_frames, (1.0, 0.0, 1.0), -0.15, (0.0, 0.03, -0.04)),
    )
    xyz = torch.zeros(2 * n_points, n_frames, 3)
    for b, (R, t) in enumerate(motions):
        members = body == bool(b)
        rotated = torch.einsum("tij,nj->nti", R, local[members])
        xyz[members] = rotated + centers[b] + t[None]
    return xyz, body


def test_kmeans_separates_well_separated_blobs():
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(60, 3, generator=generator) * 0.1
    b = torch.randn(40, 3, generator=generator) * 0.1 + torch.tensor([3.0, 0.0, 0.0])
    labels = kmeans(torch.cat([a, b]), 2)
    assert labels[:60].unique().numel() == 1 and labels[60:].unique().numel() == 1
    assert labels[0] != labels[-1]
    assert kmeans(torch.zeros(3, 3), 5).max() < 3  # more clusters than points


def test_identity_bases_leave_points_in_place():
    bases = MotionBases.identity(num_bases=3, num_frames=5)
    points = torch.randn(10, 3)
    logits = torch.randn(10, 3)
    assert torch.allclose(bases.transform(logits, points, 2), points, atol=1e-6)
    assert bases.smoothness().item() == 0.0
    assert bases.num_bases == 3 and bases.num_frames == 5


def test_blend_and_trajectories_are_consistent():
    R, t = rigid_trajectory(6, (0.2, 1.0, 0.1), 0.3, (0.1, 0.0, 0.05))
    bases = MotionBases(torch.stack([R, R.transpose(1, 2)]), torch.stack([t, -t]))
    points = torch.randn(7, 3)
    hard = torch.tensor([[20.0, -20.0]]).expand(7, 2)
    # With one-hot coefficients a point follows its basis exactly.
    expected = torch.einsum("tij,nj->nti", R, points) + t[None]
    assert torch.allclose(bases.trajectories(hard, points), expected, atol=1e-5)
    soft = torch.randn(7, 2)
    trajectories = bases.trajectories(soft, points)
    for frame in range(6):
        assert torch.allclose(
            trajectories[:, frame], bases.transform(soft, points, frame), atol=1e-6
        )
        rotation, _ = bases.blend(soft, frame)
        eye = torch.eye(3).expand(7, 3, 3)
        assert torch.allclose(rotation @ rotation.transpose(1, 2), eye, atol=1e-5)
    assert bases.smoothness().item() > 0.0


def test_initialisation_chains_through_short_tracks():
    """No track spans the video: each is visible for 6 consecutive frames only."""
    xyz, body = two_bodies()
    n, n_frames, _ = xyz.shape
    generator = torch.Generator().manual_seed(1)
    start = torch.randint(0, n_frames - 5, (n,), generator=generator)
    frames = torch.arange(n_frames)[None, :]
    visible = (frames >= start[:, None]) & (frames < start[:, None] + 6)
    assert visible.sum(dim=1).max() == 6

    bases, logits, canonical, canonical_frame = init_motion_bases(xyz, visible, num_bases=4)
    assert 0 <= canonical_frame < n_frames
    predicted = bases.trajectories(logits, canonical)
    error = (predicted - xyz).norm(dim=-1)
    # Positions are reproduced on the observed frames ...
    assert error[visible].max() < 2e-2
    # ... and, thanks to the rigid model, extrapolated to the frames where a point was
    # not tracked at all.
    assert error[~visible].mean() < 5e-2
    # A basis only ever holds points of a single body.
    assignment = logits.argmax(dim=1)
    for b in assignment.unique():
        assert body[assignment == b].unique().numel() == 1


def test_initialisation_handles_degenerate_inputs():
    xyz = torch.rand(2, 5, 3)
    visible = torch.zeros(2, 5, dtype=torch.bool)
    visible[0, 0] = True
    bases, logits, canonical, _ = init_motion_bases(xyz, visible, num_bases=3)
    assert bases.num_bases == 1 and logits.shape == (2, 1)
    assert torch.isfinite(canonical).all()


def test_fitting_refines_noisy_bases():
    xyz, _ = two_bodies(n_frames=12, n_points=50)
    generator = torch.Generator().manual_seed(2)
    noisy = xyz + 0.01 * torch.randn(xyz.shape, generator=generator)
    visible = torch.rand(xyz.shape[:2], generator=generator) < 0.7
    visible[:, 5] = True
    bases, logits, canonical, _ = init_motion_bases(noisy, visible, num_bases=4, canonical_frame=5)
    # Perturb the initial solution, then let the fit repair it.
    with torch.no_grad():
        bases.trans.add_(0.05 * torch.randn(bases.trans.shape, generator=generator))
    before = (bases.trajectories(logits, canonical) - xyz).norm(dim=-1).mean().item()
    logits, canonical = fit_motion_bases(bases, logits, canonical, noisy, visible, 200, extent=1.0)
    after = (bases.trajectories(logits, canonical).detach() - xyz).norm(dim=-1).mean().item()
    assert after < 0.5 * before
    assert after < 0.02  # close to the noise floor of the observations


# ---------------------------------------------------------------- motion segmentation


def test_rigid_flow_equals_true_flow_on_a_static_scene(tiny_still):
    seq = tiny_still
    gt = seq.gt
    w2c = invert_se3(gt.c2w)
    flow, valid = rigid_flow(gt.depth[1], seq.K(), w2c[1], w2c[5])
    true_flow, true_valid = gt.oracle.flow(1, 5)
    assert (flow - true_flow)[valid & true_valid].abs().max() < 2e-3
    assert valid.float().mean() > 0.5


def test_motion_masks_find_the_moving_objects(tiny_rolling):
    seq = tiny_rolling
    gt = seq.gt
    cfg = MotionSegConfig(steps=(1, 2), residual_threshold=0.3, min_area=6, morph_radius=1)
    residuals = flow_residuals(
        seq.images, gt.depth, seq.K(), invert_se3(gt.c2w), OracleFlow(gt.oracle), cfg
    )
    assert residuals.shape == gt.depth.shape
    static_residual = residuals[~gt.dynamic_mask & torch.isfinite(residuals)]
    assert static_residual.median() < 0.05
    masks = motion_masks(residuals, cfg)
    intersection = (masks & gt.dynamic_mask).sum().item()
    union = (masks | gt.dynamic_mask).sum().item()
    assert intersection / union > 0.6
    # Almost nothing static is flagged.
    assert (masks & ~gt.dynamic_mask).sum().item() < 0.15 * gt.dynamic_mask.sum().item()


def test_small_components_are_removed():
    residuals = torch.zeros(1, 40, 40)
    residuals[0, 5:25, 5:25] = 5.0
    residuals[0, 32:34, 32:34] = 5.0  # a 4-pixel speck
    masks = motion_masks(residuals, MotionSegConfig(min_area=10, morph_radius=0))
    assert masks[0, 5:25, 5:25].all()
    assert not masks[0, 30:, 30:].any()
    assert not motion_masks(torch.full((1, 8, 8), float("nan"))).any()


def test_track_labels():
    uv = torch.tensor([[[5.5, 5.5]], [[5.5, 5.5]], [[20.5, 20.5]], [[20.5, 20.5]]]).expand(4, 3, 2)
    tracks = Tracks(uv.clone(), torch.ones(4, 3, dtype=torch.bool))
    masks = torch.zeros(3, 30, 30, dtype=torch.bool)
    masks[:, :10, :10] = True
    error = torch.tensor([10.0, 0.1, 10.0, 0.1])
    static = torch.tensor([False, True, False, True])
    cfg = MotionSegConfig(track_threshold=3.0)
    assert mask_fraction(tracks, masks).tolist() == [1.0, 1.0, 0.0, 0.0]
    # With masks, the mask decides; a large error off the mask is a tracking failure.
    assert label_tracks(tracks, error, static, masks, cfg).tolist() == [True, True, False, False]
    # Without masks, only tracks that no static point explains are dynamic.
    assert label_tracks(tracks, error, static, None, cfg).tolist() == [True, False, True, False]
    # A track spending a third of its life on the mask is static.
    masks[1:] = False
    assert not label_tracks(tracks, error, static, masks, cfg).any()


def test_dynamic_queries_cover_the_masks():
    masks = torch.zeros(5, 20, 30, dtype=torch.bool)
    masks[:, 4:12, 10:22] = True
    frames, uv = dynamic_queries(masks, stride=2, interval=2)
    assert set(frames.tolist()) == {0, 2, 4}
    assert len(frames) == 3 * 4 * 6
    assert masks[frames, uv[:, 1].long(), uv[:, 0].long()].all()
    assert torch.equal(uv - uv.floor(), torch.full_like(uv, 0.5)), "queries are pixel centres"
    empty_frames, empty_uv = dynamic_queries(torch.zeros(3, 8, 8, dtype=torch.bool), 2, 1)
    assert empty_frames.shape == (0,) and empty_uv.shape == (0, 2)
