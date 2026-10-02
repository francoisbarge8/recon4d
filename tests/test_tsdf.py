"""TSDF fusion and surface extraction (points and surface-nets mesh)."""

import math

import pytest
import torch

from recon4d.fusion import TSDFConfig, TSDFVolume, fuse_depth_maps, volume_bounds
from recon4d.geometry import invert_se3
from recon4d.metrics import nearest_distances
from recon4d.viz import load_ply_vertices, save_mesh

RADIUS = 0.6
CENTER = torch.tensor([0.05, -0.02, 0.03])


def sphere_volume(resolution: int = 40) -> TSDFVolume:
    """A volume filled with the exact (truncated) signed distance to a sphere."""
    cfg = TSDFConfig(resolution=resolution, margin=0.0)
    volume = TSDFVolume(torch.full((3,), -1.0), torch.full((3,), 1.0), cfg)
    distance = (volume.voxel_centers() - CENTER).norm(dim=-1) - RADIUS
    truncation = cfg.truncation * volume.voxel_size
    volume.tsdf = (distance / truncation).clamp(-1.0, 1.0)
    volume.weight = torch.full(volume.dims, cfg.min_weight)
    volume.color = (volume.voxel_centers() + 1.0) / 2.0
    return volume


def test_volume_layout():
    volume = TSDFVolume(torch.tensor([0.0, 0.0, 0.0]), torch.tensor([2.0, 1.0, 0.5]))
    assert volume.voxel_size == pytest.approx(2.2 / 128)
    centers = volume.voxel_centers()
    assert centers.shape == (*volume.dims, 3)
    # The box, grown by its margin on every side, is covered.
    assert torch.allclose(centers[0, 0, 0], torch.tensor([-0.1, -0.05, -0.025]))
    assert (centers[-1, -1, -1] >= torch.tensor([2.1, 1.05, 0.525]) - 1e-6).all()
    lower, upper = volume_bounds(torch.rand(5000, 3, generator=torch.Generator().manual_seed(0)))
    assert (lower < 0.03).all() and (upper > 0.97).all() and (lower > 0).all()


def test_zero_crossings_lie_on_the_surface():
    volume = sphere_volume()
    points, colors = volume.extract_points()
    assert points.shape[0] > 1000
    error = ((points - CENTER).norm(dim=-1) - RADIUS).abs()
    assert error.max() < 0.05 * volume.voxel_size
    # Colours are interpolated along the edges like positions.
    assert torch.allclose(colors, (points + 1.0) / 2.0, atol=1e-5)


def test_surface_nets_mesh_of_a_sphere_is_closed_and_oriented():
    volume = sphere_volume()
    vertices, faces, colors = volume.extract_mesh()
    assert faces.min() >= 0 and faces.max() < vertices.shape[0]
    assert colors.shape == vertices.shape
    # One vertex per crossed cell: within half a cell diagonal of the surface, and much
    # closer on average.
    error = ((vertices - CENTER).norm(dim=-1) - RADIUS).abs()
    assert error.max() < 0.5 * volume.voxel_size and error.mean() < 0.1 * volume.voxel_size

    # Closed manifold: every edge is shared by exactly two triangles, with opposite
    # directions, and the Euler characteristic is that of a sphere.
    directed = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n = vertices.shape[0]
    forward = directed[:, 0] * n + directed[:, 1]
    backward = directed[:, 1] * n + directed[:, 0]
    assert torch.unique(forward).numel() == forward.numel(), "no edge is traversed twice"
    assert torch.equal(torch.sort(forward).values, torch.sort(backward).values)
    assert n - forward.numel() // 2 + faces.shape[0] == 2

    # Outward orientation: the signed volume is positive and matches the sphere's.
    v0, v1, v2 = (vertices[faces[:, i]] for i in range(3))
    signed_volume = (v0 * torch.linalg.cross(v1, v2)).sum(dim=-1).sum() / 6.0
    assert float(signed_volume) == pytest.approx(4.0 / 3.0 * math.pi * RADIUS**3, rel=0.02)


def plane_views(n_frames: int, noise: float = 0.0):
    """A camera at the origin looking down +z at the plane z = 2."""
    generator = torch.Generator().manual_seed(0)
    K = torch.tensor([[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]])
    depth = 2.0 + noise * torch.randn(n_frames, 48, 64, generator=generator)
    return depth, K, torch.eye(4).repeat(n_frames, 1, 1)


def test_fusion_averages_the_noise_of_the_depth_maps():
    noise = 0.02
    depth, K, w2c = plane_views(16, noise)
    volume = fuse_depth_maps(depth, K, w2c, cfg=TSDFConfig(resolution=48))
    assert volume.voxel_size > noise, "the noise is well inside the truncation band"
    points, _ = volume.extract_points()
    assert points.shape[0] > 200
    assert float((points[:, 2] - 2.0).abs().mean()) < 0.4 * noise
    assert float(points[:, 2].mean()) == pytest.approx(2.0, abs=0.25 * noise)


def test_surfaces_need_enough_observations():
    depth, K, w2c = plane_views(1)
    once = fuse_depth_maps(depth, K, w2c, cfg=TSDFConfig(resolution=32))
    assert once.extract_points()[0].shape[0] == 0, "a single view is not trusted"
    assert once.extract_mesh()[1].shape[0] == 0
    trusting = fuse_depth_maps(depth, K, w2c, cfg=TSDFConfig(resolution=32, min_weight=1.0))
    assert trusting.extract_points()[0].shape[0] > 0

    # Masked pixels and invalid depth leave no surface.
    depth, K, w2c = plane_views(3)
    mask = torch.ones_like(depth, dtype=torch.bool)
    mask[:, :, :32] = False
    depth[:, :10] = 0.0
    volume = fuse_depth_maps(depth, K, w2c, mask=mask, cfg=TSDFConfig(resolution=32))
    points, _ = volume.extract_points()
    assert points.shape[0] > 0
    assert points[:, 0].min() > -volume.voxel_size, "nothing on the masked (left) half"
    row = K[1, 1] * points[:, 1] / points[:, 2] + K[1, 2]
    assert row.min() > 10 - 1.0, "nothing where the depth is invalid (top rows)"
    with pytest.raises(ValueError, match="no usable depth"):
        fuse_depth_maps(torch.zeros(2, 8, 8), K, w2c[:2])


def test_fusing_a_synthetic_scene_recovers_its_surface(tiny_still):
    seq = tiny_still
    gt = seq.gt
    volume = fuse_depth_maps(
        gt.depth, seq.K(), invert_se3(gt.c2w), seq.images, cfg=TSDFConfig(resolution=64)
    )
    vertices, faces, colors = volume.extract_mesh()
    assert faces.shape[0] > 2000
    assert colors.min() >= 0.0 and colors.max() <= 1.0 and colors.std() > 0.05
    surface = gt.oracle.surface_cloud(20000, 0, 2).static
    # Accuracy: the mesh lies on the true surface, to within a fraction of a voxel on
    # average (depth is exact; the error is the discretisation of pixels and voxels).
    accuracy = nearest_distances(vertices, surface)
    assert float(accuracy.mean()) < 0.5 * volume.voxel_size
    assert float(accuracy.quantile(0.95)) < 1.5 * volume.voxel_size
    # Completeness: most of the surface the video sees twice is reconstructed.
    completeness = nearest_distances(surface, vertices)
    assert float((completeness < 2.0 * volume.voxel_size).float().mean()) > 0.8


def test_mesh_export(tmp_path):
    vertices, faces, colors = sphere_volume(16).extract_mesh()
    save_mesh(tmp_path / "mesh.ply", vertices, faces, colors)
    data = load_ply_vertices(tmp_path / "mesh.ply")
    assert torch.allclose(torch.from_numpy(data["x"].copy()), vertices[:, 0])
    assert set(data) == {"x", "y", "z", "red", "green", "blue"}
    raw = (tmp_path / "mesh.ply").read_bytes()
    header, body = raw.split(b"end_header\n")
    assert f"element face {faces.shape[0]}".encode() in header
    assert len(body) == vertices.shape[0] * 15 + faces.shape[0] * 13
    save_mesh(tmp_path / "plain.ply", vertices, faces)
    assert set(load_ply_vertices(tmp_path / "plain.ply")) == {"x", "y", "z"}
