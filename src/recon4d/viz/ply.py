"""PLY export of point clouds and Gaussian scenes.

Gaussians are written in the layout of the reference 3DGS implementation, which standard
viewers (SuperSplat, the SIBR viewer, gsplat's viewer, Blender add-ons) can open.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from recon4d.gaussians.model import GaussianCloud
from recon4d.geometry.rotations import normalize_quat, quat_multiply, rotmat_to_quat


def _write_ply(path: Path, names: list[str], formats: list[str], columns: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype([(name, fmt) for name, fmt in zip(names, formats, strict=True)])
    vertices = np.empty(columns[0].shape[0], dtype=dtype)
    for name, column in zip(names, columns, strict=True):
        vertices[name] = column
    ply_types = {"<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(vertices)}"]
    header += [
        f"property {ply_types[fmt]} {name}" for name, fmt in zip(names, formats, strict=True)
    ]
    header.append("end_header")
    with path.open("wb") as file:
        file.write(("\n".join(header) + "\n").encode("ascii"))
        file.write(vertices.tobytes())


def save_point_cloud(path: str | Path, points: Tensor, colors: Tensor | None = None) -> None:
    """Write ``points (N, 3)`` and optional RGB ``colors (N, 3)`` in ``[0, 1]`` as a binary PLY."""
    xyz = points.detach().cpu().numpy().astype("<f4")
    names, formats, columns = ["x", "y", "z"], ["<f4"] * 3, [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    if colors is not None:
        rgb = (colors.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        names += ["red", "green", "blue"]
        formats += ["u1"] * 3
        columns += [rgb[:, 0], rgb[:, 1], rgb[:, 2]]
    _write_ply(Path(path), names, formats, columns)


def save_mesh(
    path: str | Path, vertices: Tensor, faces: Tensor, colors: Tensor | None = None
) -> None:
    """Write a triangle mesh (``vertices (V, 3)``, ``faces (F, 3)``, optional RGB in
    ``[0, 1]`` per vertex) as a binary PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = vertices.detach().cpu().numpy().astype("<f4")
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {xyz.shape[0]}"]
    header += ["property float x", "property float y", "property float z"]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    vertex_data = np.empty(xyz.shape[0], dtype=np.dtype(fields))
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if colors is not None:
        rgb = (colors.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        vertex_data["red"], vertex_data["green"], vertex_data["blue"] = (
            rgb[:, 0],
            rgb[:, 1],
            rgb[:, 2],
        )
    triangles = faces.detach().cpu().numpy().astype("<i4")
    face_data = np.empty(triangles.shape[0], dtype=np.dtype([("n", "u1"), ("v", "<i4", (3,))]))
    face_data["n"], face_data["v"] = 3, triangles
    header += [
        f"element face {triangles.shape[0]}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    with path.open("wb") as file:
        file.write(("\n".join(header) + "\n").encode("ascii"))
        file.write(vertex_data.tobytes())
        file.write(face_data.tobytes())


def save_gaussians(
    path: str | Path,
    cloud: GaussianCloud,
    means: Tensor | None = None,
    rotations: Tensor | None = None,
) -> None:
    """Write a Gaussian cloud in the reference 3DGS ``.ply`` layout.

    Args:
        means: overrides the positions (for dynamic Gaussians moved to a given frame).
        rotations: ``(N, 3, 3)`` rotation applied to every Gaussian (idem).
    """
    params = {name: value.detach().cpu() for name, value in cloud.params.items()}
    xyz = (params["means"] if means is None else means.detach().cpu()).numpy()
    quats = normalize_quat(params["quats"])
    if rotations is not None:
        quats = quat_multiply(rotmat_to_quat(rotations.detach().cpu()), quats)
    n = xyz.shape[0]
    names = ["x", "y", "z", "nx", "ny", "nz"]
    columns = [xyz[:, 0], xyz[:, 1], xyz[:, 2]] + [np.zeros(n, dtype="<f4")] * 3
    dc = params["sh_dc"][:, 0].numpy()
    names += [f"f_dc_{i}" for i in range(3)]
    columns += [dc[:, i] for i in range(3)]
    # The reference layout stores higher-order coefficients channel by channel.
    rest = params["sh_rest"].permute(0, 2, 1).reshape(n, -1).numpy()
    names += [f"f_rest_{i}" for i in range(rest.shape[1])]
    columns += [rest[:, i] for i in range(rest.shape[1])]
    names.append("opacity")
    columns.append(params["opacity_logits"].numpy())
    scales = params["log_scales"].numpy()
    names += [f"scale_{i}" for i in range(3)]
    columns += [scales[:, i] for i in range(3)]
    names += [f"rot_{i}" for i in range(4)]
    columns += [quats.numpy()[:, i] for i in range(4)]
    _write_ply(
        Path(path), names, ["<f4"] * len(names), [np.asarray(c, dtype="<f4") for c in columns]
    )


def load_ply_vertices(path: str | Path) -> dict[str, np.ndarray]:
    """Read back the vertex properties of a binary PLY written by this module."""
    with Path(path).open("rb") as file:
        names, formats, count, element = [], [], 0, ""
        while True:
            line = file.readline().decode("ascii").strip()
            if line.startswith("element"):
                element = line.split()[1]
                if element == "vertex":
                    count = int(line.split()[-1])
            elif line.startswith("property") and element == "vertex":
                _, kind, name = line.split()
                names.append(name)
                formats.append({"float": "<f4", "uchar": "u1"}[kind])
            elif line == "end_header":
                break
        dtype = np.dtype([(name, fmt) for name, fmt in zip(names, formats, strict=True)])
        data = np.frombuffer(file.read(count * dtype.itemsize), dtype=dtype)
    return {name: data[name] for name in names}
