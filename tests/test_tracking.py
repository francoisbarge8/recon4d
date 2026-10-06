"""Point trackers and optical flow."""

import pytest
import torch
import torch.nn.functional as F

from recon4d.frontend.flow import DISFlow, OracleFlow, forward_backward_error
from recon4d.frontend.tracking import KLTConfig, KLTTracker, grid_queries
from recon4d.frontend.tracking.flow_chain import FlowChainConfig, FlowChainTracker
from recon4d.geometry import pixel_centers
from recon4d.metrics import tapvid_metrics
from recon4d.types import Tracks

H, W = 80, 120


def translating_video(n_frames: int, shift: tuple[float, float], seed: int = 0) -> torch.Tensor:
    """A band-limited random texture moving by ``shift`` pixels per frame.

    Frame ``t`` shows ``texture(x - t * shift)``: a point at ``x0`` in frame 0 is at
    ``x0 + t * shift`` in frame ``t``. Sub-pixel shifts are exact up to bilinear resampling.
    """
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(1, 3, 30, 44, generator=generator)
    texture = F.interpolate(coarse, size=(H + 40, W + 40), mode="bicubic", align_corners=False)
    texture = texture.clamp(0.0, 1.0)
    centers = pixel_centers(H, W) + 20.0  # crop window inside the padded texture
    size = torch.tensor([W + 40.0, H + 40.0])
    frames = []
    for t in range(n_frames):
        grid = (centers - t * torch.tensor(shift)) / size * 2.0 - 1.0
        frames.append(F.grid_sample(texture, grid[None], mode="bilinear", align_corners=False)[0])
    return torch.stack(frames).permute(0, 2, 3, 1)


def interior_queries(margin: int = 20) -> torch.Tensor:
    return grid_queries(H, W, 12, margin=margin)


# ------------------------------------------------------------------------------- KLT


def test_klt_recovers_a_subpixel_translation():
    shift = (1.5, -0.75)
    video = translating_video(6, shift)
    queries = interior_queries()
    tracks = KLTTracker().track_queries(
        video, torch.zeros(len(queries), dtype=torch.int64), queries
    )
    assert tracks.visible.float().mean() > 0.9
    expected = queries[:, None, :] + torch.arange(6)[None, :, None] * torch.tensor(shift)
    error = (tracks.uv - expected).norm(dim=-1)[tracks.visible]
    # An offset of half a pixel (a pixel-centre convention mix-up) would show up here.
    assert error.mean() < 0.1 and error.max() < 0.5
    assert torch.allclose(tracks.uv[:, 0], queries)


def test_klt_tracks_backward_from_a_later_query_frame():
    shift = (-1.0, 0.5)
    video = translating_video(7, shift)
    queries = interior_queries()
    frames = torch.full((len(queries),), 4, dtype=torch.int64)
    tracks = KLTTracker().track_queries(video, frames, queries)
    offsets = (torch.arange(7) - 4)[None, :, None] * torch.tensor(shift)
    error = (tracks.uv - (queries[:, None, :] + offsets)).norm(dim=-1)[tracks.visible]
    assert tracks.visible[:, 0].float().mean() > 0.8 and tracks.visible[:, 6].float().mean() > 0.8
    assert error.mean() < 0.1


def test_klt_lost_points_are_invisible_and_keep_their_last_position():
    video = translating_video(10, (9.0, 0.0))
    # Points near the right border leave the image after one or two frames.
    queries = torch.tensor([[W - 12.5, 30.5], [W - 14.5, 50.5]])
    tracks = KLTTracker().track_queries(video, torch.zeros(2, dtype=torch.int64), queries)
    assert tracks.visible[:, 0].all() and not tracks.visible[:, -1].any()
    for i in range(2):
        last = int(tracks.visible[i].nonzero().max())
        assert torch.equal(tracks.uv[i, last + 1 :], tracks.uv[i, last].expand(9 - last, 2))


def test_klt_automatic_queries_cover_the_video():
    video = translating_video(12, (2.0, 0.5))
    tracker = KLTTracker(KLTConfig(keyframe_interval=4, max_corners=200))
    frames, uv = tracker.sample_queries(video)
    tracks = tracker.track(video)
    assert len(tracks) == len(frames) == len(uv) > 100
    assert set(frames.tolist()) == {0, 4, 8}
    # Every frame is covered by a healthy number of live tracks.
    assert tracks.visible.sum(dim=0).min() > 50
    # New corners avoid existing tracks: no two live tracks share a pixel neighbourhood.
    live = tracks.uv[tracks.visible[:, 8], 8]
    distances = torch.cdist(live, live) + 1e3 * torch.eye(len(live))
    assert (distances.amin(dim=1) > 2.0).float().mean() > 0.9


# ------------------------------------------------------------------------------ flow


def test_dis_flow_recovers_a_translation():
    shift = (2.5, -1.25)
    video = translating_video(2, shift)
    flow = DISFlow().estimate(video[0], video[1])
    assert flow.shape == (H, W, 2)
    interior = flow[10:-10, 10:-10]
    assert (interior - torch.tensor(shift)).norm(dim=-1).mean() < 0.2


def test_forward_backward_error():
    forward = torch.zeros(H, W, 2)
    forward[..., 0] = 3.0
    backward = -forward
    error = forward_backward_error(forward, backward)
    assert error[:, : W - 4].abs().max() < 1e-5
    assert torch.isinf(error[:, W - 2 :]).all()  # points leaving the image
    inconsistent = backward.clone()
    inconsistent[20:40, 30:60] = 0.0  # e.g. an occluder standing still there
    error = forward_backward_error(forward, inconsistent)
    assert error[25:35, 30:52].min() > 2.9
    assert error[50:, : W - 4].abs().max() < 1e-5


def test_oracle_flow_is_index_based(tiny_rolling):
    oracle = tiny_rolling.gt.oracle
    estimator = OracleFlow(oracle)
    flow = estimator.between(tiny_rolling.images, 1, 3)
    assert torch.equal(flow, oracle.flow(1, 3)[0])
    forward, backward = estimator.sequence(tiny_rolling.images)
    assert forward.shape == backward.shape == (7, 48, 64, 2)
    with pytest.raises(NotImplementedError):
        estimator.estimate(tiny_rolling.images[0], tiny_rolling.images[1])


# ------------------------------------------------------------------------ flow chain


def test_flow_chain_with_exact_flow_reproduces_ground_truth_tracks(tiny_rolling):
    seq = tiny_rolling
    oracle = seq.gt.oracle
    tracker = FlowChainTracker(OracleFlow(oracle))
    queries = grid_queries(seq.height, seq.width, 4, margin=3)
    frames = torch.full((len(queries),), 3, dtype=torch.int64)
    tracks = tracker.track_queries(seq.images, frames, queries)
    gt = oracle.tracks(frames, queries)
    both = tracks.visible & gt.visible
    error = (tracks.uv - gt.uv).norm(dim=-1)[both]
    # Chaining bilinearly interpolated flow is not exact near motion boundaries.
    assert error.median() < 0.05 and error.mean() < 0.5
    assert tracks.visible[torch.arange(len(queries)), frames].all()
    agreement = (tracks.visible == gt.visible).float().mean()
    assert agreement > 0.85
    # The tracker never claims a point that left the image.
    inside = (tracks.uv[..., 0] >= 0) & (tracks.uv[..., 0] <= seq.width)
    assert inside[tracks.visible].all()


def test_flow_chain_seeds_new_points_only_where_coverage_is_missing(tiny_rolling):
    seq = tiny_rolling
    tracker = FlowChainTracker(
        OracleFlow(seq.gt.oracle), FlowChainConfig(grid_stride=4, keyframe_interval=4)
    )
    frames, uv, tracks = tracker._detect_and_track(seq.images)
    first = int((frames == 0).sum())
    second = int((frames == 4).sum())
    assert first == len(grid_queries(seq.height, seq.width, 4, margin=2))
    assert 0 < second < 0.5 * first, "most of the image is still covered at the second keyframe"
    assert len(tracks) == first + second
    metrics = tapvid_metrics(
        tracks.uv,
        tracks.visible,
        *(lambda g: (g.uv, g.visible))(seq.gt.oracle.tracks(frames, uv)),
        frames,
        seq.width,
        seq.height,
    )
    assert metrics["delta_avg"] > 0.8 and metrics["epe_tracked"] < 0.5


# ---------------------------------------------------------------------- containers


def test_tracks_container():
    uv = torch.rand(5, 4, 2)
    visible = torch.tensor(
        [[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]], dtype=torch.bool
    )
    tracks = Tracks(uv, visible, torch.tensor([True, False, False, True, False]))
    assert len(tracks) == 5 and tracks.num_frames == 4
    assert tracks.first_visible().tolist() == [1, 0, 3, 0, 2]
    subset = tracks.subset(tracks.dynamic)
    assert len(subset) == 2 and subset.dynamic.all()
    merged = Tracks.concatenate([subset, tracks])
    assert len(merged) == 7 and merged.dynamic.sum() == 4
    assert Tracks.concatenate([Tracks(uv, visible), tracks]).dynamic is None
    with pytest.raises(ValueError):
        Tracks(uv, visible[:, :3])


def test_grid_queries():
    grid = grid_queries(48, 64, 8)
    assert grid.shape == (6 * 8, 2)
    assert torch.equal(grid[0], torch.tensor([4.5, 4.5]))
    inner = grid_queries(48, 64, 8, margin=6)
    assert len(inner) < len(grid) and inner.min() >= 6


def test_cotracker_loads_from_github_or_from_a_local_clone(tmp_path, monkeypatch):
    from recon4d.frontend.tracking.cotracker import CoTracker, CoTrackerConfig

    calls = []

    def fake_load(repo, model, source):
        calls.append((repo, model, source))
        return torch.nn.Identity()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    CoTracker()
    CoTracker(CoTrackerConfig(repo=str(tmp_path)))
    assert calls == [
        ("facebookresearch/co-tracker", "cotracker3_offline", "github"),
        (str(tmp_path), "cotracker3_offline", "local"),
    ]


def test_cotracker_bounds_points_times_frames_per_pass(monkeypatch):
    from recon4d.frontend.tracking.cotracker import CoTracker, CoTrackerConfig

    cfg = CoTrackerConfig()
    assert cfg.points_per_pass(24) == cfg.chunk  # the cpu profile is unchanged
    assert cfg.points_per_pass(60) * 60 <= cfg.max_point_frames
    assert CoTrackerConfig(max_point_frames=10).points_per_pass(60) == 1

    passes = []

    class FakeModel(torch.nn.Module):
        def forward(self, video, queries, backward_tracking):
            n_frames, n_points = video.shape[1], queries.shape[1]
            passes.append(n_points)
            uv = queries[:, None, :, 1:].expand(1, n_frames, n_points, 2)
            return uv, torch.ones(1, n_frames, n_points, dtype=torch.bool)

    monkeypatch.setattr(torch.hub, "load", lambda repo, model, source: FakeModel())
    tracker = CoTracker(CoTrackerConfig(max_point_frames=100))
    images = torch.zeros(10, 16, 16, 3)
    frames = torch.zeros(25, dtype=torch.long)
    tracks = tracker.track_queries(images, frames, torch.rand(25, 2) * 16)
    assert passes == [10, 10, 5]
    assert tracks.uv.shape == (25, 10, 2)
