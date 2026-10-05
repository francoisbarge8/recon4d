"""COLMAP text models: round trip, foreign files, dataset export and the pycolmap glue."""

import sys
import types

import pytest
import torch

from recon4d.frontend.pose import SfMConfig, triangulate_with_poses
from recon4d.frontend.pose.colmap import colmap_poses, frame_indices
from recon4d.geometry.camera import invert_se3, look_at, project, transform_points
from recon4d.io.colmap import (
    ColmapModel,
    export_dataset,
    frame_name,
    model_from_tracks,
    read_model,
    write_model,
)
from recon4d.types import Tracks

WIDTH, HEIGHT = 64, 48


def toy_reconstruction(n_points: int = 40, n_frames: int = 4):
    """Cameras on an arc looking at a blob of points, with exact observations."""
    generator = torch.Generator().manual_seed(0)
    K = torch.tensor([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    points = torch.rand(n_points, 3, generator=generator, dtype=torch.float64) - 0.5
    eyes = [
        torch.tensor([0.6 * i - 0.9, 0.1 * i, -3.0], dtype=torch.float64) for i in range(n_frames)
    ]
    c2w = torch.stack([look_at(eye, torch.zeros(3, dtype=torch.float64)) for eye in eyes])
    w2c = invert_se3(c2w)
    uv = torch.stack([project(K, transform_points(pose, points))[0] for pose in w2c], dim=1)
    visible = torch.rand(n_points, n_frames, generator=generator) > 0.2
    visible[:, :2] = True
    visible[0] = False  # a track that was never reconstructed
    return K, w2c, points, Tracks(uv.to(torch.float32), visible)


def test_model_round_trip(tmp_path):
    K, w2c, points, tracks = toy_reconstruction()
    names = [frame_name(i) for i in range(w2c.shape[0])]
    colors = torch.rand(len(tracks), 3)
    errors = torch.rand(len(tracks), dtype=torch.float64)
    model = model_from_tracks(
        K, WIDTH, HEIGHT, w2c, names, tracks, points, tracks.visible, colors, errors
    )
    assert model.points.shape[0] == len(tracks) - 1, "tracks without observations are dropped"
    write_model(tmp_path, model)
    loaded = read_model(tmp_path)

    assert (loaded.width, loaded.height, loaded.names) == (WIDTH, HEIGHT, names)
    assert torch.allclose(loaded.K, K)
    assert torch.allclose(loaded.w2c, w2c, atol=1e-12)
    assert torch.allclose(loaded.points, points[1:], atol=1e-12)
    assert torch.equal(loaded.colors, model.colors)
    assert torch.allclose(loaded.errors, errors[1:], atol=1e-6)
    for frame in range(len(names)):
        assert torch.equal(loaded.point_ids[frame], model.point_ids[frame])
        assert torch.allclose(loaded.keypoints[frame], model.keypoints[frame], atol=1e-4)
        # Every observation reprojects onto its point: the conventions are COLMAP's own.
        seen = loaded.points[loaded.point_ids[frame]]
        uv, _ = project(loaded.K, transform_points(loaded.w2c[frame], seen))
        assert (uv - loaded.keypoints[frame]).abs().max() < 1e-3


def test_model_without_points(tmp_path):
    K, w2c, _, _ = toy_reconstruction()
    names = [frame_name(i) for i in range(w2c.shape[0])]
    write_model(tmp_path, ColmapModel(K, WIDTH, HEIGHT, names, w2c))
    loaded = read_model(tmp_path)
    assert torch.allclose(loaded.w2c, w2c, atol=1e-12)
    assert loaded.points.shape == (0, 3) and all(len(k) == 0 for k in loaded.keypoints)


def test_reading_a_model_written_by_colmap(tmp_path):
    (tmp_path / "cameras.txt").write_text(
        "# Camera list\n3 SIMPLE_RADIAL 640 480 500.0 320.0 240.0 0.02\n", encoding="utf-8"
    )
    # Arbitrary identifiers, images out of order, a name with a space, an observation of a
    # point that is not in the model and an image without observations.
    (tmp_path / "images.txt").write_text(
        "# Image list\n"
        "7 1 0 0 0 0.5 0 2 3 b frame.png\n"
        "10.5 20.5 42 30.0 40.0 -1 50 60 999\n"
        "2 0.7071067811865476 0 0.7071067811865476 0 0 0 0 3 a.png\n"
        "\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text(
        "# 3D point list\n42 1.0 2.0 3.0 255 128 0 0.25 7 0\n", encoding="utf-8"
    )
    model = read_model(tmp_path)
    assert model.names == ["a.png", "b frame.png"]
    assert model.K.tolist() == [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    # Quaternion (w, x, y, z) = a rotation of 90 degrees about +y.
    expected = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64
    )
    assert torch.allclose(model.w2c[0, :3, :3], expected, atol=1e-12)
    assert model.w2c[1, :3, 3].tolist() == [0.5, 0.0, 2.0]
    assert model.point_ids[1].tolist() == [0, -1, -1] and len(model.point_ids[0]) == 0
    assert model.points.tolist() == [[1.0, 2.0, 3.0]] and model.colors.tolist() == [[255, 128, 0]]


def test_unsupported_models_are_rejected(tmp_path):
    (tmp_path / "images.txt").write_text("1 1 0 0 0 0 0 0 1 a.png\n\n", encoding="utf-8")
    cameras = tmp_path / "cameras.txt"
    cameras.write_text("1 PINHOLE 64 48 60 60 32 24\n2 PINHOLE 64 48 70 70 32 24\n", "utf-8")
    with pytest.raises(ValueError, match="single"):
        read_model(tmp_path)
    cameras.write_text("1 FISHEYE 64 48 60 60 32 24\n", "utf-8")
    with pytest.raises(ValueError, match="unsupported camera model"):
        read_model(tmp_path)
    cameras.write_text("1 PINHOLE 64 48 60 60 32\n", "utf-8")
    with pytest.raises(ValueError, match="4 parameters"):
        read_model(tmp_path)


def test_dataset_export(tmp_path):
    from PIL import Image

    K, w2c, points, tracks = toy_reconstruction()
    n_frames = w2c.shape[0]
    images = torch.rand(n_frames, HEIGHT, WIDTH, 3)
    registered = torch.tensor([True, True, False, True])
    out = export_dataset(
        tmp_path, images, K, w2c, tracks, points, tracks.visible, registered=registered
    )
    assert sorted(p.name for p in (out / "images").iterdir()) == [
        frame_name(i) for i in range(n_frames)
    ]
    with Image.open(out / "images" / frame_name(1)) as image:
        assert image.size == (WIDTH, HEIGHT)
    model = read_model(out / "sparse" / "0")
    assert model.names == [frame_name(i) for i in (0, 1, 3)], "unregistered frames are left out"
    assert torch.allclose(model.w2c, w2c[registered], atol=1e-12)
    # Points take the colour of the pixel where they were first observed.
    first_point = int(torch.nonzero(tracks.visible.any(dim=1))[0])
    u, v = tracks.uv[first_point, 0].floor().to(torch.int64).tolist()
    expected = (images[0, v, u] * 255).round().to(torch.uint8)
    assert torch.equal(model.colors[0], expected)


def test_known_poses_with_missing_frames_and_arbitrary_scale():
    K, w2c, points, tracks = toy_reconstruction(n_points=60, n_frames=5)
    tracks = Tracks(tracks.uv, torch.ones_like(tracks.visible))
    # Poses as an external system would return them: another scale, one frame missing.
    scaled = w2c.clone()
    scaled[:, :3, 3] *= 7.0
    registered = torch.tensor([True, True, False, True, True])
    scaled[2] = torch.eye(4, dtype=torch.float64)
    cfg = SfMConfig(min_track_length=2)
    result = triangulate_with_poses(tracks, K, scaled, cfg, registered, normalize=True)
    assert result.valid.all() and torch.equal(result.registered, registered)
    depths = transform_points(result.w2c[0], result.points)[:, 2]
    assert 0.8 < float(depths.median()) < 1.25, "the median depth of the points is about 1"
    assert not result.inlier[:, 2].any(), "an unregistered frame holds no inlier"
    assert torch.equal(result.w2c[2], result.w2c[1]), "it borrows the pose of a neighbour"
    # Without normalisation the structure comes out at the scale of the poses.
    raw = triangulate_with_poses(tracks, K, scaled, cfg, registered)
    assert torch.allclose(raw.points, 7.0 * points, atol=1e-3)


def test_frame_indices():
    assert frame_indices([frame_name(3), "sub/" + frame_name(12)]) == [3, 12]


def test_colmap_backend_without_pycolmap(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycolmap", None)
    with pytest.raises(ImportError, match="pip install pycolmap"):
        colmap_poses(torch.rand(3, HEIGHT, WIDTH, 3))


@pytest.mark.parametrize("version", [3, 4])
def test_colmap_backend_with_stubbed_bindings(monkeypatch, tmp_path, version):
    """The glue around pycolmap, with the bindings replaced by a stub that "reconstructs"
    a known model and leaves one frame out. pycolmap 3 takes the camera model as an argument
    of ``extract_features``, pycolmap 4 only through the reader options."""
    K, w2c, _, _ = toy_reconstruction()
    calls: dict[str, object] = {}

    class Options:
        ba_refine_focal_length = True
        ba_refine_principal_point = True
        camera_model = "SIMPLE_RADIAL"
        camera_params = ""

    class Reconstruction:
        def num_reg_images(self):
            return 3

        def write_text(self, path):
            keep = [0, 1, 3]
            names = [frame_name(i) for i in keep]
            write_model(path, ColmapModel(K, WIDTH, HEIGHT, names, w2c[keep]))

    def extract_features_3(database, image_dir, camera_mode, camera_model, reader_options):
        """extract_features(database_path, image_path, camera_mode, camera_model: str, ...)"""
        calls["frames"] = sorted(p.name for p in image_dir.iterdir())
        calls["camera"] = (camera_mode, camera_model, reader_options.camera_params)

    def extract_features_4(database, image_dir, camera_mode, reader_options):
        """extract_features(database_path, image_path, camera_mode, reader_options, ...)"""
        calls["frames"] = sorted(p.name for p in image_dir.iterdir())
        calls["camera"] = (camera_mode, reader_options.camera_model, reader_options.camera_params)

    def incremental_mapping(database, image_dir, output, options):
        calls["options"] = options
        return {0: Reconstruction()}

    stub = types.SimpleNamespace(
        ImageReaderOptions=Options,
        IncrementalPipelineOptions=Options,
        CameraMode=types.SimpleNamespace(SINGLE="single"),
        extract_features=extract_features_3 if version == 3 else extract_features_4,
        match_sequential=lambda database: calls.setdefault("matcher", "sequential"),
        match_exhaustive=lambda database: calls.setdefault("matcher", "exhaustive"),
        incremental_mapping=incremental_mapping,
    )
    monkeypatch.setitem(sys.modules, "pycolmap", stub)

    images = torch.rand(4, HEIGHT, WIDTH, 3)
    K_out, poses, registered = colmap_poses(images, K, work_dir=tmp_path)
    assert registered.tolist() == [True, True, False, True]
    assert torch.allclose(poses[registered], w2c[registered], atol=1e-12)
    assert torch.equal(poses[2], torch.eye(4, dtype=torch.float64))
    assert torch.equal(K_out, K)
    assert calls["frames"] == [frame_name(i) for i in range(4)]
    assert calls["camera"] == ("single", "PINHOLE", "60.0,60.0,32.0,24.0")
    assert calls["matcher"] == "sequential"
    # Known intrinsics are kept fixed; options unknown to this version are skipped.
    assert calls["options"].ba_refine_focal_length is False
    assert not hasattr(calls["options"], "ba_refine_extra_params")

    with pytest.raises(ValueError, match="unknown matcher"):
        colmap_poses(images, K, matcher="nope")
