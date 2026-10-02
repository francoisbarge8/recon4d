import math

import numpy as np
import pytest
import torch

from conftest import tiny_config
from recon4d.data.synthetic import (
    SCENE_NAMES,
    Texture,
    build_scene,
    build_synthetic_sequence,
    random_scene,
    render_frame,
)
from recon4d.data.synthetic.primitives import (
    DTYPE,
    INTERSECT,
    Bounce,
    RollOnCircle,
    SlideSpin,
    Squash,
    Static,
    sample_surface,
    surface_area,
)
from recon4d.data.synthetic.textures import fbm, value_noise
from recon4d.geometry import (
    Intrinsics,
    backproject_depth,
    invert_se3,
    pixel_centers,
    project,
    transform_points,
)

# -------------------------------------------------------------------------- textures


def test_value_noise_is_deterministic_bounded_and_continuous():
    p = torch.rand(2000, 3, dtype=DTYPE) * 20.0 - 10.0
    a = value_noise(p, seed=3)
    assert torch.equal(a, value_noise(p, seed=3))
    assert not torch.equal(a, value_noise(p, seed=4))
    assert a.min() >= 0.0 and a.max() <= 1.0
    # Lipschitz-continuity: a tiny step changes the value only slightly.
    assert (value_noise(p + 1e-4, seed=3) - a).abs().max() < 1e-3
    # At lattice points the noise equals the hash value there, so it is not constant.
    assert a.std() > 0.1


def test_fbm_range():
    p = torch.rand(3000, 3, dtype=DTYPE) * 8.0
    n = fbm(p, octaves=3, seed=1)
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert 0.35 < n.mean() < 0.65


@pytest.mark.parametrize("pattern", ["checker", "stripes", "dots", "noise"])
def test_textures_produce_two_tone_albedo(pattern):
    texture = Texture(pattern, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), frequency=2.0, detail=0.0)
    p = torch.rand(5000, 3, dtype=DTYPE) * 4.0
    albedo = texture.albedo(p)
    assert albedo.shape == (5000, 3)
    assert albedo.min() >= 0.0 and albedo.max() <= 1.0
    # Both colours are present.
    assert (albedo[:, 0] > 0.9).any() and (albedo[:, 2] > 0.9).any()


def test_unknown_texture_pattern_raises():
    with pytest.raises(ValueError, match="unknown texture pattern"):
        Texture("marble", (0, 0, 0), (1, 1, 1)).albedo(torch.zeros(1, 3, dtype=DTYPE))


# --------------------------------------------------------------------------- motions


def test_rolling_ball_does_not_slip():
    """The material point in contact with the floor has zero velocity."""
    motion = RollOnCircle((0.2, -0.1), 1.1, 0.25, math.radians(140.0), math.radians(30.0))
    h = 1e-6
    for t in (0.1, 0.45, 0.8):
        pose_a, pose_b = motion.pose(t - h), motion.pose(t + h)
        pose = motion.pose(t)
        contact_world = pose.center - torch.tensor([0.0, 0.25, 0.0], dtype=DTYPE)
        local = pose.to_local(contact_world[None])
        velocity = (pose_b.to_world(local) - pose_a.to_world(local)) / (2 * h)
        centre_speed = (pose_b.center - pose_a.center).norm() / (2 * h)
        assert centre_speed > 0.5
        assert velocity.norm() < 1e-4 * centre_speed
        assert pose.center[1].item() == pytest.approx(0.25)
        assert torch.allclose(pose.R @ pose.R.T, torch.eye(3, dtype=DTYPE), atol=1e-12)


def test_squash_preserves_volume_and_stays_on_the_floor():
    motion = Squash((0.0, 1.0), (0.3, 0.3, 0.3), amplitude=0.3, cycles=1.5)
    for t in np.linspace(0.0, 1.0, 7):
        pose = motion.pose(float(t))
        assert pose.scale.prod().item() == pytest.approx(0.3**3)
        assert (pose.center[1] - pose.scale[1]).item() == pytest.approx(0.0, abs=1e-12)


def test_bounce_touches_the_floor_between_hops():
    motion = Bounce((0.0, 0.0), (1.0, 0.0), radius=0.2, height=0.5, bounces=2.0)
    assert motion.pose(0.0).center[1].item() == pytest.approx(0.2)
    assert motion.pose(0.25).center[1].item() == pytest.approx(0.7)
    assert motion.pose(0.5).center[1].item() == pytest.approx(0.2)


def test_motion_flags():
    assert not Static((0, 0, 0), (1, 1, 1)).is_dynamic
    assert SlideSpin((0, 0, 0), (1, 0, 0), (1, 1, 1)).is_dynamic


# --------------------------------------------------------------------- intersections


def _implicit(kind: str, p: torch.Tensor) -> torch.Tensor:
    """Signed 'distance-like' function that vanishes on the primitive's surface."""
    if kind == "sphere":
        return p.norm(dim=-1) - 1.0
    if kind == "box":
        return p.abs().amax(dim=-1) - 1.0
    radial = p[:, [0, 2]].norm(dim=-1)
    return torch.maximum(radial, p[:, 1].abs()) - 1.0


@pytest.mark.parametrize("kind", ["sphere", "box", "cylinder"])
def test_intersections_land_on_the_surface_and_are_nearest(kind):
    n = 4000
    origins = torch.nn.functional.normalize(torch.randn(n, 3, dtype=DTYPE), dim=-1) * 4.0
    targets = (torch.rand(n, 3, dtype=DTYPE) - 0.5) * 3.0
    dirs = targets - origins  # un-normalised on purpose
    t, normal = INTERSECT[kind](origins, dirs)
    hit = torch.isfinite(t)
    assert 0.2 < hit.float().mean() < 0.95
    points = origins[hit] + t[hit, None] * dirs[hit]
    assert _implicit(kind, points).abs().max() < 1e-9
    # The normal faces the ray origin.
    assert ((normal[hit] * dirs[hit]).sum(dim=-1) < 1e-9).all()
    # No point of the segment before the hit is inside the primitive (nearest hit).
    s = torch.linspace(0.0, 0.999, 60, dtype=DTYPE)
    along = origins[hit, None] + (s[None, :, None] * t[hit, None, None]) * dirs[hit, None]
    assert (_implicit(kind, along.reshape(-1, 3)) > -1e-9).all()
    # Rays reported as misses indeed stay outside along a long stretch.
    miss = ~hit
    far = (
        origins[miss, None]
        + torch.linspace(0.0, 3.0, 200, dtype=DTYPE)[None, :, None] * dirs[miss, None]
    )
    assert (_implicit(kind, far.reshape(-1, 3)) > -1e-3).all()


@pytest.mark.parametrize("kind", ["sphere", "box", "cylinder"])
def test_rays_starting_inside_hit_the_far_side(kind):
    origins = torch.zeros(100, 3, dtype=DTYPE)
    dirs = torch.randn(100, 3, dtype=DTYPE)
    t, _ = INTERSECT[kind](origins, dirs)
    assert torch.isfinite(t).all() and (t > 0).all()
    assert _implicit(kind, origins + t[:, None] * dirs).abs().max() < 1e-9


@pytest.mark.parametrize("kind", ["sphere", "box", "cylinder"])
def test_surface_samples_lie_on_the_surface(kind):
    rng = np.random.RandomState(0)
    scale = torch.tensor([0.5, 1.5, 0.8], dtype=DTYPE)
    local, normal = sample_surface(kind, scale, 2000, rng)
    assert _implicit(kind, local).abs().max() < 1e-9
    assert torch.allclose(normal.norm(dim=-1), torch.ones(2000, dtype=DTYPE), atol=1e-9)


def test_surface_sampling_is_uniform_in_world_area():
    """On a stretched box, each face receives samples in proportion to its world area."""
    rng = np.random.RandomState(0)
    scale = torch.tensor([1.0, 3.0, 0.5], dtype=DTYPE)
    _, normal = sample_surface("box", scale, 20000, rng)
    sx, sy, sz = scale.tolist()
    areas = np.array([sy * sz, sx * sz, sx * sy]) * 8.0  # both faces of each axis
    fractions = np.array([(normal[:, k].abs() > 0.5).float().mean().item() for k in range(3)])
    assert np.allclose(fractions, areas / areas.sum(), atol=0.015)
    assert surface_area("box", scale) == pytest.approx(areas.sum())


def test_surface_area_of_scaled_sphere_and_cylinder():
    assert surface_area("sphere", torch.full((3,), 0.5, dtype=DTYPE)) == pytest.approx(
        math.pi, rel=1e-6
    )
    # Cylinder of radius 0.5 and half-height 2: side 2*pi*r*h + two caps.
    expected = 2 * math.pi * 0.5 * 4.0 + 2 * math.pi * 0.25
    scale = torch.tensor([0.5, 2.0, 0.5], dtype=DTYPE)
    assert surface_area("cylinder", scale) == pytest.approx(expected, rel=0.02)


# -------------------------------------------------------------------- scene / render


def test_all_named_scenes_build_and_render():
    K = Intrinsics.from_fov(32, 24, 60.0).matrix(DTYPE)
    for name in SCENE_NAMES:
        spec = build_scene(name)
        c2w = spec.train_c2w(5)
        frame = render_frame(spec.scene, K, c2w[2], 0.5, 24, 32, spp=2)
        assert frame.rgb.shape == (24, 32, 3)
        assert frame.rgb.min() >= 0.0 and frame.rgb.max() <= 1.0
        assert frame.rgb.std() > 0.05, "image should be textured"
        assert (frame.depth > 0.3).all() and (frame.depth < 9.0).all()
        has_dynamic = bool(spec.scene.dynamic_ids())
        assert has_dynamic == (name != "still")
        assert len(spec.val_c2w()) == 2


def test_unknown_scene_raises():
    with pytest.raises(ValueError, match="unknown scene"):
        build_scene("nope")


def test_random_scenes_differ_and_render():
    K = Intrinsics.from_fov(32, 24, 60.0).matrix(DTYPE)
    images = []
    for seed in (1, 2):
        spec = random_scene(seed)
        frame = render_frame(spec.scene, K, spec.train_c2w(3)[1], 0.5, 24, 32, spp=1)
        assert torch.isfinite(frame.rgb).all() and torch.isfinite(frame.depth).all()
        images.append(frame.rgb)
    assert (images[0] - images[1]).abs().mean() > 0.02


def test_supersampling_antialiases_without_moving_geometry():
    spec = build_scene("rolling")
    K = Intrinsics.from_fov(32, 24, 60.0).matrix(DTYPE)
    c2w = spec.train_c2w(3)[1]
    sharp = render_frame(spec.scene, K, c2w, 0.5, 24, 32, spp=1)
    smooth = render_frame(spec.scene, K, c2w, 0.5, 24, 32, spp=3)
    assert torch.equal(sharp.depth, smooth.depth)
    assert torch.equal(sharp.obj, smooth.obj)
    assert not torch.equal(sharp.rgb, smooth.rgb)
    assert (sharp.rgb - smooth.rgb).abs().mean() < 0.1


def test_depth_buffer_is_consistent_with_material_points(tiny_rolling):
    """Back-projecting GT depth gives the world position of the lifted material point."""
    seq = tiny_rolling
    oracle = seq.gt.oracle
    centers = pixel_centers(seq.height, seq.width, dtype=DTYPE).reshape(-1, 2)
    for frame in (0, 5):
        obj, local = oracle.lift(frame, centers)
        world = oracle.scene.local_to_world(obj, local, oracle.times[frame])
        backprojected = backproject_depth(
            oracle.K, oracle.c2w[frame], seq.gt.depth[frame].to(DTYPE)
        ).reshape(-1, 3)
        assert (world - backprojected).norm(dim=-1).max() < 1e-4  # depth is stored as float32
        assert torch.equal(obj.reshape(seq.height, seq.width), seq.gt.obj[frame])


# ---------------------------------------------------------------------------- oracle


def test_tracks_pass_through_their_query(tiny_rolling):
    seq = tiny_rolling
    oracle = seq.gt.oracle
    generator = torch.Generator().manual_seed(0)
    n = 300
    query_frames = torch.randint(0, seq.num_frames, (n,), generator=generator)
    # Queries at pixel centres, so that they can be compared with the per-pixel buffers.
    size = torch.tensor([seq.width, seq.height])
    query_uv = torch.floor(torch.rand(n, 2, generator=generator) * size) + 0.5
    tracks = oracle.tracks(query_frames, query_uv)
    index = torch.arange(n)
    assert torch.allclose(tracks.uv[index, query_frames], query_uv, atol=1e-3)
    assert tracks.visible[index, query_frames].all()
    # Static points do not move; dynamic points do.
    spread = (tracks.xyz - tracks.xyz[:, :1]).norm(dim=-1).amax(dim=1)
    assert spread[~tracks.dynamic].max() < 1e-6
    assert tracks.dynamic.any() and spread[tracks.dynamic].min() > 1e-3
    # Labels and depth agree with the rendered buffers at the query pixel.
    row, col = query_uv[:, 1].long(), query_uv[:, 0].long()
    assert torch.equal(tracks.dynamic, seq.gt.dynamic_mask[query_frames, row, col])
    assert torch.allclose(
        tracks.depth[index, query_frames], seq.gt.depth[query_frames, row, col], atol=1e-4
    )


def test_static_flow_matches_depth_and_pose_reprojection(tiny_still):
    """Independent derivation of the flow from the GT depth map and the camera poses."""
    seq = tiny_still
    oracle = seq.gt.oracle
    src, dst = 1, 6
    flow, valid = oracle.flow(src, dst)
    points = backproject_depth(oracle.K, oracle.c2w[src], seq.gt.depth[src].to(DTYPE)).reshape(
        -1, 3
    )
    uv, _ = project(oracle.K, transform_points(invert_se3(oracle.c2w[dst]), points))
    centers = pixel_centers(seq.height, seq.width, dtype=DTYPE).reshape(-1, 2)
    expected = (uv - centers).reshape(seq.height, seq.width, 2).to(torch.float32)
    assert (flow - expected).abs().max() < 2e-3
    assert 0.5 < valid.float().mean() <= 1.0
    assert flow.norm(dim=-1).mean() > 0.5, "the camera should have moved"


def test_occluded_points_are_not_visible(tiny_rolling):
    oracle = tiny_rolling.gt.oracle
    scene = oracle.scene
    eye = oracle.c2w[0, :3, 3]
    # Take the first primitive, pick the point of its surface farthest from the camera.
    pose = scene.poses(0.0)[0]
    away = torch.nn.functional.normalize(pose.center - eye, dim=0)
    hidden_side = pose.center + away * pose.scale.min() * 0.999
    facing_side = pose.center - away * pose.scale.min() * 1.5
    assert not scene.visible_from(hidden_side[None], eye, 0.0)[0]
    # A point floating in free space in front of the object is visible.
    if scene.objects[0].kind == "sphere":
        assert scene.visible_from(facing_side[None], eye, 0.0)[0]


def test_surface_cloud(tiny_rolling):
    oracle = tiny_rolling.gt.oracle
    cloud = oracle.surface_cloud(n_static=3000, n_dynamic=400, min_views=2)
    assert cloud.static.shape[1] == 3 and 200 < cloud.static.shape[0] <= 3000
    assert cloud.dynamic.shape == (400, tiny_rolling.num_frames, 3)
    assert cloud.dynamic_observed.any() and not cloud.dynamic_observed.all()
    lo = torch.tensor(oracle.scene.room_lo) - 1e-4
    hi = torch.tensor(oracle.scene.room_hi) + 1e-4
    assert ((cloud.static >= lo) & (cloud.static <= hi)).all()


def test_random_tracks_respect_dynamic_fraction(tiny_rolling, tiny_still):
    tracks = tiny_rolling.gt.oracle.random_tracks(500, seed=1, dynamic_fraction=0.4)
    assert tracks.dynamic.float().mean().item() == pytest.approx(0.4, abs=0.01)
    assert not tiny_still.gt.oracle.random_tracks(100).dynamic.any()


# ------------------------------------------------------------------------- sequences


def test_sequence_structure(tiny_rolling):
    seq = tiny_rolling
    assert seq.images.shape == (8, 48, 64, 3)
    assert seq.test_indices == [2, 6]
    assert seq.train_indices == [0, 1, 3, 4, 5, 7]
    assert seq.gt.depth.shape == (8, 48, 64)
    assert seq.gt.dynamic_mask.any() and not seq.gt.dynamic_mask.all()
    assert torch.allclose(seq.timestamps, torch.linspace(0, 1, 8))
    # Images are 8-bit quantised.
    assert torch.allclose(seq.images * 255, torch.round(seq.images * 255), atol=1e-4)
    assert len(seq.gt.val_cameras) == 2
    val = seq.gt.val_cameras[0]
    assert val.frames == [0, 4]
    assert val.images.shape == (2, 48, 64, 3)
    assert val.covisible.dtype == torch.bool
    assert 0.3 < val.covisible.float().mean() < 1.0


def test_sequences_are_reproducible_and_cacheable(tmp_path, tiny_rolling):
    cfg = tiny_config("rolling")
    again = build_synthetic_sequence(cfg)
    assert torch.equal(again.images, tiny_rolling.images)
    assert torch.equal(again.gt.depth, tiny_rolling.gt.depth)

    cached = build_synthetic_sequence(cfg, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    reloaded = build_synthetic_sequence(cfg, cache_dir=tmp_path)
    for a, b in ((cached, tiny_rolling), (reloaded, tiny_rolling)):
        assert torch.equal(a.images, b.images)
        assert torch.equal(a.gt.depth, b.gt.depth)
        assert torch.equal(a.gt.obj, b.gt.obj)
        assert torch.equal(a.gt.val_cameras[1].covisible, b.gt.val_cameras[1].covisible)
        assert torch.equal(a.gt.val_cameras[1].images, b.gt.val_cameras[1].images)


def test_config_cache_key_depends_on_every_field():
    assert tiny_config().cache_key() == tiny_config().cache_key()
    assert tiny_config().cache_key() != tiny_config(seed=1).cache_key()
    assert tiny_config().cache_key() != tiny_config(width=65).cache_key()
