"""Truncated signed distance fusion of depth maps, and surface extraction.

The classical way to turn posed depth maps into a surface (Curless and Levoy, SIGGRAPH
1996; KinectFusion, Newcombe et al., ISMAR 2011). Every voxel of a regular grid stores the
running average of the truncated signed distance to the surface observed along the ray
through it: positive in front of the surface, negative behind. Averaging over the views
cancels the noise of the individual maps, and the surface is the zero level set.

The level set is extracted either as points (the zero crossings of the grid edges) or as
a triangle mesh by *naive surface nets* (Gibson, 1998): one vertex per grid cell crossed by
the surface, placed at the mean of the crossings of its edges, and one quad per crossed
grid edge, joining the four cells around it. Unlike marching cubes it needs no case table
and vectorises in a few tensor operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.geometry.camera import invert_se3, pixel_centers, transform_points, unproject


@dataclass(frozen=True)
class TSDFConfig:
    """Parameters of the fusion.

    Attributes:
        resolution: number of voxels along the longest side of the volume.
        truncation: truncation distance, in voxels.
        max_weight: cap on the per-voxel weight (number of observations).
        min_weight: a voxel takes part in the surface only if it was observed at least
            this many times: surfaces seen by a single frame are not trusted.
        margin: the volume is grown by this fraction of its size on every side.
    """

    resolution: int = 128
    truncation: float = 3.0
    max_weight: float = 64.0
    min_weight: float = 2.0
    margin: float = 0.05


def volume_bounds(points: Tensor, quantile: float = 0.01) -> tuple[Tensor, Tensor]:
    """Robust axis-aligned bounds ``(lower, upper)`` of a point cloud ``(N, 3)``."""
    step = max(points.shape[0] // 200_000, 1)
    q = torch.tensor([quantile, 1.0 - quantile], dtype=points.dtype, device=points.device)
    bounds = torch.quantile(points[::step], q, dim=0)
    return bounds[0], bounds[1]


class TSDFVolume:
    """A dense TSDF grid over the box ``[lower, upper]``.

    Voxel ``(i, j, k)`` is the sample at ``origin + voxel_size * (i, j, k)``.
    """

    def __init__(
        self,
        lower: Tensor,
        upper: Tensor,
        cfg: TSDFConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg or TSDFConfig()
        lower, upper = lower.to(torch.float32).cpu(), upper.to(torch.float32).cpu()
        pad = self.cfg.margin * (upper - lower)
        lower, upper = lower - pad, upper + pad
        self.voxel_size = float((upper - lower).max()) / self.cfg.resolution
        dims = torch.ceil((upper - lower) / self.voxel_size).to(torch.int64) + 1
        self.dims = tuple(int(d) for d in dims.clamp_min(2))
        self.origin = lower.to(device)
        self.device = torch.device(device)
        self.tsdf = torch.ones(self.dims, device=device)
        self.weight = torch.zeros(self.dims, device=device)
        self.color = torch.zeros(*self.dims, 3, device=device)

    def voxel_centers(self) -> Tensor:
        """World positions ``(X, Y, Z, 3)`` of the voxels."""
        axes = [torch.arange(n, device=self.device, dtype=torch.float32) for n in self.dims]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
        return self.origin + self.voxel_size * grid

    @torch.no_grad()
    def integrate(
        self,
        depth: Tensor,
        K: Tensor,
        w2c: Tensor,
        color: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> None:
        """Fuse one depth map.

        Args:
            depth: ``(H, W)`` z-depth; non-positive values are ignored.
            K, w2c: intrinsics ``(3, 3)`` and world-to-camera pose ``(4, 4)``.
            color: optional ``(H, W, 3)`` image.
            mask: optional ``(H, W)`` pixels to use (e.g. valid and static).
        """
        height, width = depth.shape
        depth, K, w2c = depth.to(self.device), K.to(self.device), w2c.to(self.device)
        cam = transform_points(w2c, self.voxel_centers().reshape(-1, 3))
        z = cam[:, 2]
        in_front = z > 1e-6
        z_safe = torch.where(in_front, z, torch.ones_like(z))
        # Nearest pixel: interpolating depth across an occlusion boundary would create
        # surfaces that do not exist. Pixel (r, c) covers [c, c + 1) x [r, r + 1).
        col = torch.floor(K[0, 0] * cam[:, 0] / z_safe + K[0, 2]).to(torch.int64)
        row = torch.floor(K[1, 1] * cam[:, 1] / z_safe + K[1, 2]).to(torch.int64)
        inside = in_front & (col >= 0) & (col < width) & (row >= 0) & (row < height)
        col, row = col.clamp(0, width - 1), row.clamp(0, height - 1)
        observed = depth[row, col]
        usable = inside & (observed > 0)
        if mask is not None:
            usable &= mask.to(self.device)[row, col]
        truncation = self.cfg.truncation * self.voxel_size
        distance = observed - z
        # Voxels far behind the surface are hidden from this view: leave them alone.
        usable &= distance > -truncation
        value = (distance / truncation).clamp_max(1.0)

        tsdf, weight = self.tsdf.reshape(-1), self.weight.reshape(-1)
        index = torch.nonzero(usable)[:, 0]
        w_old = weight[index]
        w_new = (w_old + 1.0).clamp_max(self.cfg.max_weight)
        tsdf[index] = (tsdf[index] * w_old + value[index]) / (w_old + 1.0)
        if color is not None:
            near = index[distance[index].abs() < truncation]
            w_near = weight[near][:, None]
            sampled = color.to(self.device)[row[near], col[near]]
            flat_color = self.color.reshape(-1, 3)
            flat_color[near] = (flat_color[near] * w_near + sampled) / (w_near + 1.0)
        weight[index] = w_new

    # ------------------------------------------------------------------- extraction

    def _edge_crossings(self, axis: int) -> tuple[Tensor, Tensor]:
        """Sign changes along the grid edges parallel to ``axis``.

        Returns ``crossing`` (bool) and ``t`` (position of the zero along the edge, in
        ``[0, 1]``), both shaped like the grid with one fewer sample along ``axis``.
        """
        n = self.dims[axis]
        d0, d1 = self.tsdf.narrow(axis, 0, n - 1), self.tsdf.narrow(axis, 1, n - 1)
        w0, w1 = self.weight.narrow(axis, 0, n - 1), self.weight.narrow(axis, 1, n - 1)
        seen = (w0 >= self.cfg.min_weight) & (w1 >= self.cfg.min_weight)
        crossing = seen & ((d0 < 0) != (d1 < 0))
        t = d0 / torch.where(crossing, d0 - d1, torch.ones_like(d0))
        return crossing, t.clamp(0.0, 1.0)

    @torch.no_grad()
    def extract_points(self) -> tuple[Tensor, Tensor]:
        """Zero crossings of the grid edges: ``points (N, 3)`` and ``colors (N, 3)``."""
        points, colors = [], []
        for axis in range(3):
            crossing, t = self._edge_crossings(axis)
            index = torch.nonzero(crossing)
            if index.numel() == 0:
                continue
            t = t[crossing]
            position = index.to(torch.float32)
            position[:, axis] += t
            points.append(self.origin + self.voxel_size * position)
            other = index.clone()
            other[:, axis] += 1
            c0 = self.color[index[:, 0], index[:, 1], index[:, 2]]
            c1 = self.color[other[:, 0], other[:, 1], other[:, 2]]
            colors.append(c0 + t[:, None] * (c1 - c0))
        if not points:
            empty = torch.zeros(0, 3, device=self.device)
            return empty, empty.clone()
        return torch.cat(points), torch.cat(colors)

    @torch.no_grad()
    def extract_mesh(self) -> tuple[Tensor, Tensor, Tensor]:
        """Triangle mesh of the zero level set by naive surface nets.

        Returns ``vertices (V, 3)``, ``faces (F, 3)`` (indices, counter-clockwise when seen
        from the free-space side) and ``colors (V, 3)``.
        """
        nx, ny, nz = self.dims
        cells = (nx - 1, ny - 1, nz - 1)
        total = torch.zeros(*cells, 3, device=self.device)
        tint = torch.zeros(*cells, 3, device=self.device)
        count = torch.zeros(cells, device=self.device)
        crossings = []
        for axis in range(3):
            b, c = (axis + 1) % 3, (axis + 2) % 3
            crossing, t = self._edge_crossings(axis)
            crossings.append(crossing)
            n = self.dims[axis]
            edge_color = self.color.narrow(axis, 0, n - 1) + t[..., None] * (
                self.color.narrow(axis, 1, n - 1) - self.color.narrow(axis, 0, n - 1)
            )
            # A cell owns four edges parallel to each axis, at offsets (ob, oc) of its
            # lower corner along the two other axes.
            for ob in (0, 1):
                for oc in (0, 1):
                    hit = crossing.narrow(b, ob, self.dims[b] - 1).narrow(c, oc, self.dims[c] - 1)
                    along = t.narrow(b, ob, self.dims[b] - 1).narrow(c, oc, self.dims[c] - 1)
                    shade = edge_color.narrow(b, ob, self.dims[b] - 1).narrow(
                        c, oc, self.dims[c] - 1
                    )
                    hit_f = hit.to(torch.float32)
                    offset = torch.zeros(3, device=self.device)
                    offset[b], offset[c] = float(ob), float(oc)
                    contribution = hit_f[..., None] * offset
                    contribution[..., axis] = hit_f * along
                    total += contribution
                    tint += hit_f[..., None] * shade
                    count += hit_f

        active = count > 0
        cell_index = torch.nonzero(active)
        n_vertices = cell_index.shape[0]
        if n_vertices == 0:
            empty = torch.zeros(0, 3, device=self.device)
            return empty, torch.zeros(0, 3, dtype=torch.int64, device=self.device), empty.clone()
        weight = count[active][:, None]
        local = total[active] / weight
        vertices = self.origin + self.voxel_size * (cell_index.to(torch.float32) + local)
        colors = tint[active] / weight
        vertex_id = torch.full(cells, -1, dtype=torch.int64, device=self.device)
        vertex_id[active] = torch.arange(n_vertices, device=self.device)

        faces = []
        for axis in range(3):
            b, c = (axis + 1) % 3, (axis + 2) % 3
            edges = torch.nonzero(crossings[axis])
            # The four cells around the edge exist only away from the border of the grid.
            interior = (
                (edges[:, b] >= 1)
                & (edges[:, b] <= self.dims[b] - 2)
                & (edges[:, c] >= 1)
                & (edges[:, c] <= self.dims[c] - 2)
            )
            edges = edges[interior]
            if edges.numel() == 0:
                continue
            corners = []
            for ob, oc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
                cell = edges.clone()
                cell[:, b] += ob
                cell[:, c] += oc
                corners.append(vertex_id[cell[:, 0], cell[:, 1], cell[:, 2]])
            # All four cells contain the crossed edge, so each of them has a vertex.
            quad = torch.stack(corners, dim=1)
            # The quad as listed turns counter-clockwise around +axis: that is the outward
            # orientation when the edge goes from inside (negative) to outside.
            outward = self.tsdf[edges[:, 0], edges[:, 1], edges[:, 2]] < 0
            quad = torch.where(outward[:, None], quad, quad.flip(1))
            faces.append(quad[:, [0, 1, 2]])
            faces.append(quad[:, [0, 2, 3]])
        if not faces:
            return vertices, torch.zeros(0, 3, dtype=torch.int64, device=self.device), colors
        return vertices, torch.cat(faces), colors


def fuse_depth_maps(
    depth: Tensor,
    K: Tensor,
    w2c: Tensor,
    images: Tensor | None = None,
    mask: Tensor | None = None,
    cfg: TSDFConfig | None = None,
    device: torch.device | str = "cpu",
) -> TSDFVolume:
    """Fuse posed depth maps into a TSDF volume sized to what they see.

    Args:
        depth: ``(T, H, W)`` z-depth in the frame of the poses.
        K, w2c: intrinsics ``(3, 3)`` and world-to-camera poses ``(T, 4, 4)``.
        images: optional ``(T, H, W, 3)`` colours.
        mask: optional ``(T, H, W)`` pixels to fuse (e.g. valid and static).
    """
    n_frames, height, width = depth.shape
    usable = depth > 0 if mask is None else mask & (depth > 0)
    if not usable.any():
        raise ValueError("no usable depth to fuse")
    # Bounds from a subsampled back-projection of every frame.
    step = max(int((usable.sum() // 400_000) ** 0.5), 1)
    centers = pixel_centers(height, width, dtype=depth.dtype)[::step, ::step].reshape(-1, 2)
    c2w = invert_se3(w2c)
    clouds = []
    for t in range(n_frames):
        z = depth[t, ::step, ::step].reshape(-1)
        keep = usable[t, ::step, ::step].reshape(-1)
        clouds.append(transform_points(c2w[t], unproject(K, centers[keep], z[keep])))
    lower, upper = volume_bounds(torch.cat(clouds))
    volume = TSDFVolume(lower, upper, cfg, device)
    for t in range(n_frames):
        volume.integrate(
            depth[t],
            K,
            w2c[t],
            None if images is None else images[t],
            None if mask is None else mask[t],
        )
    return volume
