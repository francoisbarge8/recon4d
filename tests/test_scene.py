"""Gaussian clouds, density control, the 4D scene and its optimisation."""

import pytest
import torch

from recon4d.frontend.tracking import grid_queries
from recon4d.gaussians.densify import (
    DensificationStats,
    DensifyConfig,
    densify_and_prune,
    reset_opacity,
)
from recon4d.gaussians.init import InitConfig, init_scene, lift_tracks, scene_extent
from recon4d.gaussians.model import CloudOptimizer, GaussianCloud, knn_scale
from recon4d.gaussians.motion import MotionBases
from recon4d.gaussians.render import Camera, render_gaussians
from recon4d.gaussians.scene import MOTION_LOGITS, GaussianScene
from recon4d.gaussians.trainer import SceneTrainer, TrainConfig, Window, render_views
from recon4d.geometry import Intrinsics, invert_se3, look_at, quat_to_rotmat, so3_exp
from recon4d.metrics import psnr
from recon4d.types import Tracks, TrainingData

LRS = {
    "means": 1e-3,
    "log_scales": 5e-3,
    "quats": 1e-3,
    "opacity_logits": 5e-2,
    "sh_dc": 2.5e-3,
    "sh_rest": 1e-4,
}


def small_cloud(n: int = 50, seed: int = 0, **kwargs) -> GaussianCloud:
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(n, 3, generator=generator) - 0.5
    colors = torch.rand(n, 3, generator=generator)
    return GaussianCloud.from_points(points, colors, **kwargs)


def camera(width: int = 48, height: int = 36) -> Camera:
    K = Intrinsics.from_fov(width, height, 60.0).matrix()
    w2c = invert_se3(look_at(torch.tensor([0.2, 0.3, -2.5]), torch.zeros(3)))
    return Camera(K, w2c, width, height)


# ----------------------------------------------------------------------------- cloud


def test_cloud_from_points():
    points = torch.rand(40, 3)
    colors = torch.rand(40, 3)
    cloud = GaussianCloud.from_points(points, colors, sh_degree=2, opacity=0.3, scale_factor=0.5)
    assert len(cloud) == 40
    assert torch.allclose(cloud.means, points)
    assert torch.allclose(cloud.opacities, torch.full((40,), 0.3), atol=1e-6)
    assert torch.allclose(cloud.colors(torch.zeros(3)), colors, atol=1e-6)
    assert torch.allclose(cloud.scales[:, 0], 0.5 * knn_scale(points), atol=1e-6)
    assert cloud.sh.shape == (40, 9, 3) and cloud.active_sh_degree == 0
    for _ in range(5):
        cloud.raise_sh_degree()
    assert cloud.active_sh_degree == 2
    assert torch.allclose(cloud.quats.norm(dim=-1), torch.ones(40))


def test_cloud_validates_its_parameters():
    cloud = small_cloud()
    params = cloud.detached()
    with pytest.raises(ValueError, match="missing"):
        GaussianCloud({k: v for k, v in params.items() if k != "quats"}, 0)
    with pytest.raises(ValueError, match="rows"):
        GaussianCloud({**params, "quats": params["quats"][:10]}, 0)
    with pytest.raises(ValueError, match="sh_rest"):
        GaussianCloud(params, 2)


def test_optimizer_keeps_adam_state_through_structure_edits():
    cloud = small_cloud(20)
    optimizer = CloudOptimizer(cloud, LRS)
    cloud.means.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    state = optimizer.optimizer.state[cloud.params["means"]]
    moment = state["exp_avg"].clone()
    assert moment.abs().sum() > 0

    keep = torch.arange(20) % 2 == 0
    optimizer.prune(keep)
    assert len(cloud) == 10
    state = optimizer.optimizer.state[cloud.params["means"]]
    assert torch.equal(state["exp_avg"], moment[keep])

    extra = {name: value[:3].clone() for name, value in cloud.detached().items()}
    optimizer.append(extra)
    assert len(cloud) == 13
    state = optimizer.optimizer.state[cloud.params["means"]]
    assert (
        torch.equal(state["exp_avg"][:10], moment[keep]) and state["exp_avg"][10:].abs().sum() == 0
    )

    optimizer.reset("opacity_logits", torch.full((13,), -2.0))
    assert torch.allclose(cloud.params["opacity_logits"], torch.full((13,), -2.0))
    # The edited parameters are the ones the optimiser now updates.
    cloud.means.sum().backward()
    before = cloud.means.detach().clone()
    optimizer.step()
    assert not torch.equal(cloud.means.detach(), before)
    with pytest.raises(ValueError, match="missing parameters"):
        optimizer.append({"means": torch.zeros(1, 3)})
    with pytest.raises(ValueError, match="no learning rate"):
        CloudOptimizer(cloud, {"means": 1e-3})


# ----------------------------------------------------------------------- densification


def test_densify_clones_small_splits_large_and_prunes_transparent():
    cloud = small_cloud(6, extras={MOTION_LOGITS: torch.randn(6, 3)})
    with torch.no_grad():
        cloud.params["log_scales"][:] = torch.log(torch.tensor(0.01))
        cloud.params["log_scales"][1] = torch.log(torch.tensor(0.5))  # large -> split
        cloud.params["opacity_logits"][5] = -10.0  # transparent -> pruned
    optimizer = CloudOptimizer(cloud, {**LRS, MOTION_LOGITS: 1e-2})
    stats = DensificationStats(6)
    stats.grad_sum = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    stats.count = torch.ones(6)
    before = cloud.detached()
    cfg = DensifyConfig(grad_threshold=0.5, size_threshold=0.1, max_scale=10.0, max_growth=1.0)
    counts = densify_and_prune(cloud, optimizer, stats, cfg, extent=1.0)
    assert counts == {"cloned": 1, "split": 1, "pruned": 1, "total": 7}
    # Survivors 0, 2, 3, 4, then the clone of 0, then the two children of 1.
    assert torch.equal(cloud.means[:4], before["means"][[0, 2, 3, 4]])
    assert torch.equal(cloud.means[4], before["means"][0])
    children = cloud.scales[5:]
    assert torch.allclose(children, torch.full((2, 3), 0.5 / 1.6), atol=1e-6)
    # Extra per-Gaussian parameters follow their Gaussians.
    assert torch.equal(cloud.params[MOTION_LOGITS][4], before[MOTION_LOGITS][0])
    assert torch.equal(cloud.params[MOTION_LOGITS][5], before[MOTION_LOGITS][1])
    assert stats.grad_sum.shape == (7,) and stats.grad_sum.sum() == 0


def test_densification_respects_the_budget_and_growth_rate():
    def run(**overrides):
        cloud = small_cloud(100)
        optimizer = CloudOptimizer(cloud, LRS)
        stats = DensificationStats(100)
        stats.grad_sum = torch.linspace(1.0, 2.0, 100)
        stats.count = torch.ones(100)
        cfg = DensifyConfig(grad_threshold=0.5, size_threshold=10.0, max_scale=1e3, **overrides)
        return densify_and_prune(cloud, optimizer, stats, cfg, extent=1.0), cloud

    counts, cloud = run(max_gaussians=1000, max_growth=1.0)
    assert counts["cloned"] == 100 and len(cloud) == 200
    counts, _ = run(max_gaussians=107, max_growth=1.0)
    assert counts["cloned"] == 7
    counts, cloud = run(max_gaussians=1000, max_growth=0.1)
    assert counts["cloned"] == 10
    # The candidates with the largest gradients (the last ones) are the ones densified.
    assert torch.equal(cloud.means[100:], cloud.means[90:100])


def test_density_schedule_and_opacity_reset():
    cfg = DensifyConfig(start=200, stop=1000, interval=100, opacity_reset_interval=300)
    assert [i for i in range(0, 1200, 50) if cfg.is_due(i)] == list(range(200, 1000, 100))
    assert [i for i in range(0, 1200, 50) if cfg.reset_is_due(i)] == [300, 600, 900]
    cloud = small_cloud(10, opacity=0.6)
    optimizer = CloudOptimizer(cloud, LRS)
    reset_opacity(cloud, optimizer, 0.05)
    assert torch.allclose(cloud.opacities, torch.full((10,), 0.05), atol=1e-6)


def test_stats_average_gradients_in_ndc_units():
    stats = DensificationStats(3)
    grad = torch.tensor([[0.01, 0.0], [0.0, 0.02], [5.0, 5.0]])
    visible = torch.tensor([True, True, False])
    stats.update(grad, visible, width=200, height=100)
    stats.update(grad, visible, width=200, height=100)
    assert torch.allclose(stats.mean(), torch.tensor([1.0, 1.0, 0.0]))


# ----------------------------------------------------------------------------- scene


def test_render_outputs_depth_alpha_and_extras():
    cloud = small_cloud(200, opacity=0.9)
    cam = camera()
    extras = torch.ones(200, 2)
    out = render_gaussians(
        cloud.means,
        cloud.quats,
        cloud.scales,
        cloud.opacities,
        cloud.colors(cam.center),
        cam,
        background=torch.tensor([0.0, 1.0, 0.0]),
        extras=extras,
    )
    assert out.color.shape == (36, 48, 3) and out.depth.shape == out.alpha.shape == (36, 48)
    hit = out.alpha > 0.5
    assert hit.any()
    # Depth is the alpha-normalised z of the Gaussians: within the cloud's depth range.
    z = out.projection.depths
    assert (out.depth[hit] > z.min() - 1e-3).all() and (out.depth[hit] < z.max() + 1e-3).all()
    # Extras that are 1 for every Gaussian composite to the accumulated opacity.
    assert torch.allclose(out.extras[..., 0], out.alpha, atol=1e-5)
    # The background shows through where nothing was hit.
    empty = out.alpha < 1e-4
    assert torch.allclose(
        out.color[empty], torch.tensor([0.0, 1.0, 0.0]).expand(int(empty.sum()), 3)
    )
    assert (out.depth[empty] == 0).all()


def moving_scene(n_frames: int = 4) -> tuple[GaussianScene, torch.Tensor, torch.Tensor]:
    """A static cloud plus a dynamic cloud that rotates and translates rigidly."""
    static = small_cloud(60, seed=1, opacity=0.8)
    dynamic = small_cloud(40, seed=2, opacity=0.8, extras={MOTION_LOGITS: torch.zeros(40, 1)})
    times = torch.arange(n_frames, dtype=torch.float32)
    R = so3_exp(times[:, None] * torch.tensor([0.0, 0.3, 0.1]))
    t = times[:, None] * torch.tensor([0.1, 0.0, 0.05])
    return GaussianScene(static, dynamic, MotionBases(R[None], t[None])), R, t


def test_dynamic_scene_equals_static_render_of_the_moved_gaussians():
    scene, R, t = moving_scene()
    cam = camera()
    frame = 2
    out = scene.render(cam, frame)
    dynamic = scene.dynamic
    moved = dynamic.means @ R[frame].T + t[frame]
    rotations = torch.cat(
        [quat_to_rotmat(scene.static.quats), R[frame] @ quat_to_rotmat(dynamic.quats)]
    )
    reference = render_gaussians(
        torch.cat([scene.static.means, moved]),
        rotations,
        torch.cat([scene.static.scales, dynamic.scales]),
        torch.cat([scene.static.opacities, dynamic.opacities]),
        torch.cat([scene.static.colors(cam.center), dynamic.colors(cam.center, moved)]),
        cam,
    )
    assert torch.allclose(out.color, reference.color, atol=1e-5)
    assert torch.allclose(out.depth, reference.depth, atol=1e-4)
    assert out.n_static == 60
    # The dynamic map is the share of each pixel explained by dynamic Gaussians.
    assert out.dynamic.min() >= 0 and (out.dynamic <= out.alpha + 1e-5).all()
    assert out.dynamic.max() > 0.3
    assert torch.allclose(scene.dynamic_means(frame), moved, atol=1e-5)
    assert torch.allclose(scene.dynamic_trajectories()[:, frame], moved, atol=1e-5)


def test_rendered_positions_are_dense_3d_correspondences():
    """A lone opaque Gaussian: every pixel it covers reports its position at the other frame."""
    static = small_cloud(1, seed=3, opacity=0.99)
    dynamic = small_cloud(1, seed=4, opacity=0.99, extras={MOTION_LOGITS: torch.zeros(1, 1)})
    with torch.no_grad():
        static.params["means"][:] = torch.tensor([[-0.5, 0.0, 0.0]])
        dynamic.params["means"][:] = torch.tensor([[0.5, 0.0, 0.0]])
        for cloud in (static, dynamic):
            cloud.params["log_scales"][:] = torch.log(torch.tensor(0.15))
    t = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]]])
    scene = GaussianScene(static, dynamic, MotionBases(torch.eye(3).expand(1, 2, 3, 3), t))
    out = scene.render(camera(), 0, position_frames=(0, 1))
    covered_dynamic = out.dynamic > 0.9
    covered_static = (out.alpha > 0.9) & (out.dynamic < 0.01)
    assert covered_dynamic.any() and covered_static.any()
    now, later = out.positions
    assert torch.allclose(
        now[covered_dynamic],
        torch.tensor([0.5, 0.0, 0.0]).expand_as(now[covered_dynamic]),
        atol=1e-3,
    )
    assert torch.allclose(
        later[covered_dynamic],
        torch.tensor([0.5, 0.4, 0.0]).expand_as(later[covered_dynamic]),
        atol=1e-3,
    )
    assert torch.allclose(
        later[covered_static],
        torch.tensor([-0.5, 0.0, 0.0]).expand_as(later[covered_static]),
        atol=1e-3,
    )


def test_scene_requires_consistent_dynamic_parts():
    static = small_cloud(5)
    with pytest.raises(ValueError, match="together"):
        GaussianScene(static, small_cloud(5), None)
    with pytest.raises(ValueError, match="motion_logits"):
        GaussianScene(static, small_cloud(5), MotionBases.identity(1, 2))
    assert not GaussianScene(static).is_dynamic


# --------------------------------------------------------------------------- trainer


def ground_truth_data(seq, with_tracks: bool = True) -> TrainingData:
    """Training data built from the ground truth of a synthetic sequence."""
    gt = seq.gt
    tracks = None
    if with_tracks:
        frames, uv = [], []
        for f in range(0, seq.num_frames, 3):
            grid = grid_queries(seq.height, seq.width, 4, margin=2)
            frames.append(torch.full((len(grid),), f, dtype=torch.int64))
            uv.append(grid)
        truth = gt.oracle.tracks(torch.cat(frames), torch.cat(uv))
        tracks = Tracks(truth.uv, truth.visible, truth.dynamic)
    return TrainingData(
        seq.images,
        seq.K(),
        invert_se3(gt.c2w),
        seq.train_indices,
        depth=gt.depth,
        dynamic_mask=gt.dynamic_mask,
        tracks=tracks,
    )


def test_window_crops_consistently():
    window = Window(10, 5, 20, 12)
    image = torch.arange(40 * 30, dtype=torch.float32).reshape(30, 40)
    assert window.crop(image).shape == (12, 20) and window.crop(image)[0, 0] == image[5, 10]
    K = Intrinsics.from_fov(40, 30, 60.0).matrix()
    assert window.intrinsics(K)[0, 2] == K[0, 2] - 10 and window.intrinsics(K)[1, 2] == K[1, 2] - 5
    uv = torch.tensor([[10.2, 6.0], [15.0, 10.0], [29.9, 10.0], [15.0, 17.4]])
    assert window.contains(uv).tolist() == [False, True, False, False]
    assert torch.equal(window.to_local(uv[1:2]), torch.tensor([[5.0, 5.0]]))


def test_cropped_render_matches_the_crop_of_the_full_render():
    cloud = small_cloud(150, opacity=0.8)
    cam = camera(48, 36)
    full = render_gaussians(
        cloud.means, cloud.quats, cloud.scales, cloud.opacities, cloud.colors(cam.center), cam
    )
    window = Window(8, 4, 24, 16)
    cropped_cam = Camera(window.intrinsics(cam.K), cam.w2c, window.width, window.height)
    part = render_gaussians(
        cloud.means,
        cloud.quats,
        cloud.scales,
        cloud.opacities,
        cloud.colors(cam.center),
        cropped_cam,
    )
    # Identical, except near the border: the Jacobian of Gaussians centred outside the
    # image is evaluated at a clamped position that depends on the image size.
    difference = (window.crop(full.color) - part.color).abs()
    assert difference[4:-4, 4:-4].max() < 1e-3
    assert difference.max() < 0.1


def test_static_training_improves_held_out_views(tiny_still):
    seq = tiny_still
    data = ground_truth_data(seq, with_tracks=False)
    scene, extent = init_scene(data, InitConfig(n_static=900, keyframes=3, stride=1))
    assert extent == pytest.approx(scene_extent(data)) and 1.0 < extent < 6.0
    assert not scene.is_dynamic and 300 < len(scene.static) <= 900

    def held_out_psnr() -> float:
        test = seq.test_indices
        out = render_views(scene, data.K, data.w2c[test], test, seq.width, seq.height)
        assert out["color"].shape == (len(test), seq.height, seq.width, 3)
        return psnr(out["color"], seq.images[test]).item()

    before = held_out_psnr()
    cfg = TrainConfig(
        iterations=60,
        densify=DensifyConfig(start=20, stop=50, interval=20, max_gaussians=1200),
        log_every=20,
    )
    trainer = SceneTrainer(scene, data, cfg, extent)
    history = trainer.fit()
    after = held_out_psnr()
    assert len(history) == 3 and history[-1]["iteration"] == 60
    assert history[-1]["loss"] < history[0]["loss"]
    assert after > before + 2.0
    assert {"l1", "dssim", "depth"} <= set(history[-1])
    assert all(torch.isfinite(p).all() for p in scene.parameters())


def test_dynamic_initialisation_from_ground_truth(tiny_rolling):
    seq = tiny_rolling
    data = ground_truth_data(seq)
    cfg = InitConfig(
        n_static=800, n_dynamic=300, num_bases=3, keyframes=3, stride=1, basis_fit_iterations=100
    )
    scene, extent = init_scene(data, cfg)
    assert scene.is_dynamic and 50 < len(scene.dynamic) <= 300
    assert scene.dynamic.params[MOTION_LOGITS].shape == (len(scene.dynamic), scene.motion.num_bases)
    # With exact depth the lifted tracks are the true 3D trajectories.
    xyz, ok = lift_tracks(data, data.tracks.dynamic)
    assert ok.any()
    # Dynamic Gaussians, moved to each frame, lie on the moving objects seen in that frame.
    for frame in (1, 5):
        moved = scene.dynamic_means(frame).detach()
        on_objects = xyz[:, frame][ok[:, frame]]
        distance = torch.cdist(moved, on_objects).amin(dim=1)
        assert distance.median() < 0.08 * extent
    # The scene can also be built without its dynamic part.
    static_only, _ = init_scene(data, cfg, dynamic=False)
    assert not static_only.is_dynamic


def test_dynamic_training_runs_with_all_losses_and_crops(tiny_rolling):
    seq = tiny_rolling
    data = ground_truth_data(seq)
    cfg = InitConfig(
        n_static=600, n_dynamic=200, num_bases=3, keyframes=3, stride=1, basis_fit_iterations=50
    )
    scene, extent = init_scene(data, cfg)
    train = TrainConfig(
        iterations=24,
        crop=(40, 32),
        densify=DensifyConfig(start=8, stop=20, interval=8, max_gaussians=900),
        max_dynamic_gaussians=300,
        log_every=8,
    )
    trainer = SceneTrainer(scene, data, train, extent)
    history = trainer.fit()
    expected = {"l1", "dssim", "depth", "mask", "track", "track_depth", "rigid", "smooth", "loss"}
    assert expected <= set(history[-1])
    assert all(torch.isfinite(torch.tensor(v)) for v in history[-1].values())
    assert all(torch.isfinite(p).all() for p in scene.parameters())
    assert history[-1]["n_dynamic"] == len(scene.dynamic) <= 300
    out = render_views(scene, data.K, data.w2c[:2], [0, 1], seq.width, seq.height)
    assert out["dynamic"].shape == (2, seq.height, seq.width)
    # Training is reproducible given the seed.
    scene2, _ = init_scene(data, cfg)
    history2 = SceneTrainer(scene2, data, train, extent).fit()
    assert history2[-1]["loss"] == pytest.approx(history[-1]["loss"], rel=1e-4)
