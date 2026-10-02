"""Visualisation helpers, PLY export and video loading."""

import numpy as np
import pytest
import torch
from PIL import Image

from recon4d.data.video import load_video
from recon4d.gaussians.model import GaussianCloud
from recon4d.geometry import invert_se3, look_at, project, transform_points
from recon4d.report import orbit_poses
from recon4d.types import Tracks
from recon4d.viz import (
    colorize,
    colorize_depth,
    draw_tracks,
    load_ply_vertices,
    overlay_mask,
    save_gaussians,
    save_gif,
    save_image,
    save_point_cloud,
    save_video,
    side_by_side,
)


def test_colorize_and_depth_colours():
    values = torch.linspace(0.0, 1.0, 12).reshape(3, 4)
    image = colorize(values, 0.0, 1.0)
    assert image.shape == (3, 4, 3) and image.min() >= 0 and image.max() <= 1
    assert not torch.equal(image[0, 0], image[-1, -1])
    depth = torch.tensor([[1.0, 2.0], [4.0, 0.0]])
    colours = colorize_depth(depth)
    assert colours.shape == (2, 2, 3)
    assert (colours[1, 1] == 0).all(), "invalid depth is drawn black"
    assert colours[0, 0, 0] > colours[1, 0, 0], "near is warmer (more red) than far"
    assert (colorize_depth(torch.zeros(2, 2)) == 0).all()


def test_overlay_and_side_by_side():
    image = torch.zeros(2, 4, 6, 3)
    mask = torch.zeros(2, 4, 6, dtype=torch.bool)
    mask[:, :2] = True
    tinted = overlay_mask(image, mask, color=(1.0, 0.0, 0.0), alpha=0.5)
    assert torch.allclose(tinted[:, 0, 0], torch.tensor([0.5, 0.0, 0.0]).expand(2, 3))
    assert (tinted[:, 3] == 0).all()
    panel = side_by_side(image, tinted, gap=2)
    assert panel.shape == (2, 4, 14, 3)
    assert (panel[:, :, 6:8] == 1).all()


def test_images_gifs_and_videos_are_written(tmp_path):
    frames = torch.rand(5, 12, 16, 3)
    save_image(tmp_path / "a" / "frame.png", frames[0], scale=2)
    assert Image.open(tmp_path / "a" / "frame.png").size == (32, 24)
    save_gif(tmp_path / "clip.gif", frames, fps=10, scale=3)
    with Image.open(tmp_path / "clip.gif") as gif:
        assert gif.size == (48, 36) and gif.n_frames == 5
    save_video(tmp_path / "clip.mp4", frames, fps=10, scale=2)
    assert (tmp_path / "clip.mp4").stat().st_size > 0


def test_draw_tracks():
    images = torch.zeros(4, 20, 30, 3)
    uv = torch.tensor([[[5.5, 5.5], [8.5, 5.5], [11.5, 5.5], [14.5, 5.5]]])
    tracks = Tracks(uv, torch.ones(1, 4, dtype=torch.bool), torch.tensor([True]))
    drawn = draw_tracks(images, tracks)
    assert drawn.shape == images.shape
    assert drawn[0].sum() > 0 and drawn[3, 5, 14].sum() > 0
    assert drawn[3].sum() > drawn[0].sum(), "the trail grows with time"
    assert draw_tracks(images, Tracks(uv, torch.zeros(1, 4, dtype=torch.bool))).sum() == 0


def test_point_cloud_ply_round_trip(tmp_path):
    points = torch.rand(50, 3)
    colors = torch.rand(50, 3)
    save_point_cloud(tmp_path / "points.ply", points, colors)
    data = load_ply_vertices(tmp_path / "points.ply")
    assert np.allclose(np.stack([data["x"], data["y"], data["z"]], 1), points.numpy())
    assert np.abs(data["red"] / 255.0 - colors[:, 0].numpy()).max() < 0.5 / 255 + 1e-6
    save_point_cloud(tmp_path / "bare.ply", points)
    assert set(load_ply_vertices(tmp_path / "bare.ply")) == {"x", "y", "z"}


def test_gaussian_ply_uses_the_reference_layout(tmp_path):
    cloud = GaussianCloud.from_points(torch.rand(20, 3), torch.rand(20, 3), sh_degree=1)
    with torch.no_grad():
        cloud.params["sh_rest"].copy_(torch.randn(20, 3, 3))
        cloud.params["quats"].copy_(torch.randn(20, 4))
    save_gaussians(tmp_path / "gaussians.ply", cloud)
    data = load_ply_vertices(tmp_path / "gaussians.ply")
    expected = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    expected += [f"f_rest_{i}" for i in range(9)] + ["opacity"]
    expected += [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    assert list(data) == expected
    assert np.allclose(data["opacity"], cloud.params["opacity_logits"].detach().numpy())
    assert np.allclose(data["scale_1"], cloud.params["log_scales"][:, 1].detach().numpy())
    # Higher-order SH are stored channel by channel: f_rest_3 is coefficient 0 of green.
    assert np.allclose(data["f_rest_3"], cloud.params["sh_rest"][:, 0, 1].detach().numpy())
    quats = np.stack([data[f"rot_{i}"] for i in range(4)], 1)
    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-5)
    # Moved Gaussians: positions are replaced and orientations composed.
    moved = cloud.means.detach() + 1.0
    save_gaussians(tmp_path / "moved.ply", cloud, moved, torch.eye(3).expand(20, 3, 3))
    data_moved = load_ply_vertices(tmp_path / "moved.ply")
    assert np.allclose(data_moved["x"], data["x"] + 1.0, atol=1e-6)
    same = np.abs(np.sum(quats * np.stack([data_moved[f"rot_{i}"] for i in range(4)], 1), axis=1))
    assert np.allclose(same, 1.0, atol=1e-5), "an identity rotation leaves orientations unchanged"


def test_orbit_poses_circle_the_reference_view_and_look_at_the_scene():
    eyes = torch.stack([torch.tensor([x, 0.5, -3.0]) for x in (-0.5, 0.0, 0.5)])
    w2c = invert_se3(torch.stack([look_at(e, torch.zeros(3)) for e in eyes]))
    depth = torch.full((3, 4, 4), 3.0)
    poses = orbit_poses(w2c, depth, n_views=8, amplitude=0.1)
    assert poses.shape == (8, 4, 4)
    R = poses[:, :3, :3]
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(8, 3, 3), atol=1e-5)
    centers = invert_se3(poses)[:, :3, 3]
    reference = invert_se3(w2c)[1]
    assert torch.allclose(
        (centers - reference[:3, 3]).norm(dim=1), torch.full((8,), 0.3), atol=1e-4
    )
    # Every pose looks at the point at median depth in front of the reference camera.
    target = reference[:3, 3] + 3.0 * reference[:3, 2]
    K = torch.tensor([[50.0, 0, 32.0], [0, 50.0, 24.0], [0, 0, 1]])
    uv, z = project(K, transform_points(poses, target[None, None].expand(8, 1, 3)))
    assert torch.allclose(uv[:, 0], torch.tensor([32.0, 24.0]).expand(8, 2), atol=1e-3)
    assert (z > 0).all()


def test_load_video_from_a_folder_of_frames(tmp_path):
    for i in range(7):
        frame = (np.random.RandomState(i).rand(40, 100, 3) * 255).astype(np.uint8)
        Image.fromarray(frame).save(tmp_path / f"frame_{i:03d}.png")
    seq = load_video(tmp_path, max_frames=5, max_side=64, fov_x_deg=60.0)
    assert seq.images.shape == (5, 24, 64, 3)
    assert seq.images.min() >= 0.0 and seq.images.max() <= 1.0
    assert seq.gt is None and seq.intrinsics.width == 64
    assert torch.allclose(seq.timestamps, torch.linspace(0, 1, 5))
    every_other = load_video(tmp_path, stride=2, max_side=200)
    assert every_other.images.shape == (4, 40, 96, 3) and every_other.intrinsics is None
    with pytest.raises(ValueError, match="at least 3"):
        load_video(tmp_path, start=5)
    with pytest.raises(OSError, match="cannot open"):
        load_video(tmp_path / "missing.mp4")
