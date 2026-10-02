"""Differentiable 3D Gaussian splatting in pure PyTorch.

This is a re-implementation of the rasterizer of *3D Gaussian Splatting for Real-Time
Radiance Field Rendering* (Kerbl et al., SIGGRAPH 2023) with autograd doing the backward
pass, so it runs unmodified on CPU and on any accelerator PyTorch supports:

1. **Projection** (:func:`project_gaussians`): each Gaussian is moved to the camera frame
   and its covariance is pushed through the local affine approximation of the perspective
   projection (EWA splatting), ``Sigma' = J W Sigma W^T J^T``, plus a small isotropic
   low-pass filter.
2. **Binning**: every Gaussian is assigned to the screen tiles overlapped by the bounding
   box of its footprint; pairs are ordered by ``(tile, depth)`` with one stable sort.
3. **Compositing** (:func:`rasterize`): inside each tile the Gaussians are alpha-blended
   front to back, ``C = sum_i f_i alpha_i prod_{j<i} (1 - alpha_j)``.

Instead of a per-pixel loop, tiles holding a similar number of Gaussians are grouped into
buckets and processed as dense ``(tiles, pixels, gaussians)`` tensors. Two details make
this fast enough for CPU training:

* The exponent of a Gaussian is a quadratic polynomial of the pixel position, so for a
  whole tile it is one small matrix product between per-Gaussian coefficients and a fixed
  monomial basis of the tile's pixels, rather than a dozen element-wise passes.
* The bounding box is *opacity-aware and exact*: a Gaussian of opacity ``o`` contributes
  only where ``o exp(-q/2) >= alpha_min``, an ellipse whose axis-aligned box is known in
  closed form. Faint Gaussians touch few tiles, nothing is ever truncated, and the image
  is independent of the tile size.

Because gradients come from autograd, every input is differentiable, including the camera
pose, which is what enables photometric pose refinement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from recon4d.geometry.rotations import quat_to_rotmat

_PAD_EXPONENT = -1.0e4  # exp() of this is exactly 0: used for the padded slots of a tile


@dataclass(frozen=True)
class RasterSettings:
    """Rasterizer parameters; the defaults follow the reference implementation.

    Attributes:
        tile_size: side of the square screen tiles, in pixels.
        near: Gaussians closer than this (camera z) are culled.
        alpha_min: contributions below this opacity are skipped (1/255).
        alpha_max: per-Gaussian opacity is clamped to this value (0.99).
        cov_blur: variance (pixels^2) of the screen-space low-pass filter added to every
            projected covariance; it guarantees a minimal footprint of about one pixel.
        max_sigma: upper bound on the footprint extent, in standard deviations. The
            exact extent is ``sqrt(2 ln(o / alpha_min))`` (at most 3.33 for
            ``alpha_min = 1/255``), so the default never truncates anything. Lower values
            trade exactness for speed.
        antialias: compensate the opacity for the energy added by ``cov_blur``
            (as in Mip-Splatting); off reproduces the original behaviour.
        max_per_tile: optional cap on the number of Gaussians composited per tile (the
            farthest ones are dropped). ``None`` keeps the result exact.
        chunk_elements: upper bound on ``tiles * pixels * gaussians`` processed at once,
            which bounds peak memory.
    """

    tile_size: int = 8
    near: float = 0.05
    alpha_min: float = 1.0 / 255.0
    alpha_max: float = 0.99
    cov_blur: float = 0.3
    max_sigma: float = 3.33
    antialias: bool = False
    max_per_tile: int | None = None
    chunk_elements: int = 6_000_000


@dataclass
class Projection:
    """Screen-space description of a set of Gaussians."""

    means2d: Tensor
    """``(N, 2)`` pixel coordinates of the centres."""
    conics: Tensor
    """``(N, 3)`` upper triangle ``(a, b, c)`` of the inverse 2D covariance."""
    depths: Tensor
    """``(N,)`` camera-space z of the centres."""
    extents: Tensor
    """``(N, 2)`` half-sizes of the footprint's bounding box in pixels; 0 when culled."""
    compensation: Tensor
    """``(N,)`` opacity factor of the anti-aliasing filter (1 when disabled)."""

    @property
    def visible(self) -> Tensor:
        return self.extents[:, 0] > 0

    @property
    def radii(self) -> Tensor:
        """``(N,)`` largest half-size of the bounding box (0 when culled)."""
        return self.extents.amax(dim=-1)


@dataclass
class RasterOutput:
    image: Tensor
    """``(H, W, C)`` composited features (over the background when one is given)."""
    alpha: Tensor
    """``(H, W)`` accumulated opacity ``1 - prod_i (1 - alpha_i)``."""


def covariance_3d(quats: Tensor, scales: Tensor) -> Tensor:
    """World-space covariances ``R S S^T R^T`` from rotations and per-axis std-devs."""
    M = quat_to_rotmat(quats) * scales[..., None, :]
    return M @ M.transpose(-1, -2)


def project_gaussians(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    K: Tensor,
    w2c: Tensor,
    width: int,
    height: int,
    settings: RasterSettings | None = None,
    opacities: Tensor | None = None,
) -> Projection:
    """Project 3D Gaussians to screen space (EWA splatting).

    Args:
        means: ``(N, 3)`` world-space centres.
        quats: ``(N, 4)`` rotations ``(w, x, y, z)``; they need not be normalised.
        scales: ``(N, 3)`` standard deviations along the local axes.
        K: ``(3, 3)`` intrinsics.
        w2c: ``(4, 4)`` world-to-camera matrix.
        opacities: optional ``(N,)`` opacities, used to shrink the bounding box of faint
            Gaussians (and to cull those below ``alpha_min`` altogether).
    """
    settings = settings or RasterSettings()
    R_wc = w2c[:3, :3]
    cam = means @ R_wc.transpose(0, 1) + w2c[:3, 3]
    x, y, z = cam.unbind(-1)
    in_front = z > settings.near
    z_safe = torch.where(in_front, z, torch.ones_like(z))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy

    # The affine approximation degrades far outside the frustum; like the reference
    # implementation, clamp the point at which the Jacobian is evaluated.
    lim_x_lo, lim_x_hi = 1.3 * cx / fx, 1.3 * (width - cx) / fx
    lim_y_lo, lim_y_hi = 1.3 * cy / fy, 1.3 * (height - cy) / fy
    tx = z_safe * torch.minimum(torch.maximum(x / z_safe, -lim_x_lo), lim_x_hi)
    ty = z_safe * torch.minimum(torch.maximum(y / z_safe, -lim_y_lo), lim_y_hi)

    zero = torch.zeros_like(z)
    J = torch.stack(
        [
            torch.stack([fx / z_safe, zero, -fx * tx / z_safe**2], dim=-1),
            torch.stack([zero, fy / z_safe, -fy * ty / z_safe**2], dim=-1),
        ],
        dim=-2,
    )
    M = quat_to_rotmat(quats) * scales[..., None, :]  # R S, so that Sigma = M M^T
    T = J @ (R_wc @ M)
    cov = T @ T.transpose(-1, -2)
    a = cov[:, 0, 0] + settings.cov_blur
    b = cov[:, 0, 1]
    c = cov[:, 1, 1] + settings.cov_blur
    det = a * c - b * b
    valid = in_front & (det > 0)
    det_safe = torch.where(valid, det, torch.ones_like(det))
    conics = torch.stack([c / det_safe, -b / det_safe, a / det_safe], dim=-1)

    if settings.antialias:
        det_orig = cov[:, 0, 0] * cov[:, 1, 1] - b * b
        compensation = torch.sqrt((det_orig / det_safe).clamp_min(1e-12))
    else:
        compensation = torch.ones_like(det)

    with torch.no_grad():
        # Footprint {q <= 2 ln(o / alpha_min)}: an ellipse whose axis-aligned bounding box
        # has half-sizes sigma * sqrt(Sigma_xx) and sigma * sqrt(Sigma_yy).
        if opacities is None:
            n_sigma = torch.full_like(z, settings.max_sigma)
        else:
            ratio = (opacities * compensation / settings.alpha_min).clamp_min(1.0)
            n_sigma = torch.sqrt(2.0 * torch.log(ratio)).clamp_max(settings.max_sigma)
        # The tiny inflation makes the box robustly conservative under rounding.
        ext_x = n_sigma * torch.sqrt(a.clamp_min(0.0)) * (1.0 + 1e-5) + 1e-5
        ext_y = n_sigma * torch.sqrt(c.clamp_min(0.0)) * (1.0 + 1e-5) + 1e-5
        on_screen = (
            (u + ext_x >= 0.5)
            & (u - ext_x <= width - 0.5)
            & (v + ext_y >= 0.5)
            & (v - ext_y <= height - 0.5)
        )
        keep = valid & on_screen & (n_sigma > 0)
        extents = torch.stack([ext_x, ext_y], dim=-1) * keep[:, None]

    return Projection(torch.stack([u, v], dim=-1), conics, z, extents, compensation)


def _bin_gaussians(
    means2d: Tensor, depths: Tensor, extents: Tensor, tiles_x: int, tiles_y: int, tile_size: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Assign Gaussians to the tiles holding at least one pixel centre of their bounding box.

    Returns one ``gaussian_id`` and ``tile_id`` per overlapping (Gaussian, tile) pair,
    grouped by tile and depth-sorted within each tile, together with the ``start`` offset
    and ``count`` of every tile in that list.
    """
    n_tiles = tiles_x * tiles_y
    device = means2d.device
    x, y = means2d[:, 0], means2d[:, 1]
    ex, ey = extents[:, 0], extents[:, 1]
    # Tile t holds the pixel centres t * ts + 0.5 ... (t + 1) * ts - 0.5.
    x0 = (torch.ceil((x - ex + 0.5) / tile_size) - 1).clamp(0, tiles_x).to(torch.int64)
    x1 = (torch.floor((x + ex - 0.5) / tile_size) + 1).clamp(0, tiles_x).to(torch.int64)
    y0 = (torch.ceil((y - ey + 0.5) / tile_size) - 1).clamp(0, tiles_y).to(torch.int64)
    y1 = (torch.floor((y + ey - 0.5) / tile_size) + 1).clamp(0, tiles_y).to(torch.int64)
    span_x = (x1 - x0).clamp_min(0)
    span_y = (y1 - y0).clamp_min(0)
    count = torch.where(ex > 0, span_x * span_y, torch.zeros_like(span_x))

    # Enumerate pairs with the Gaussians taken front to back; a *stable* sort by tile
    # index then leaves each tile's list depth-sorted.
    order = torch.argsort(depths)
    count_sorted = count[order]
    total = int(count_sorted.sum())
    if total == 0:
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        zeros = torch.zeros(n_tiles, dtype=torch.int64, device=device)
        return empty, empty, zeros, zeros
    gaussian_ids = torch.repeat_interleave(order, count_sorted)
    first = torch.cumsum(count_sorted, dim=0) - count_sorted
    local = torch.arange(total, device=device) - torch.repeat_interleave(first, count_sorted)
    width_g = span_x[gaussian_ids]
    tile_ids = (y0[gaussian_ids] + local // width_g) * tiles_x + x0[gaussian_ids] + local % width_g
    tile_ids, permutation = torch.sort(tile_ids, stable=True)
    tile_count = torch.bincount(tile_ids, minlength=n_tiles)
    tile_start = torch.cumsum(tile_count, dim=0) - tile_count
    return gaussian_ids[permutation], tile_ids, tile_start, tile_count


def rasterize(
    projection: Projection,
    opacities: Tensor,
    features: Tensor,
    width: int,
    height: int,
    settings: RasterSettings | None = None,
    background: Tensor | None = None,
) -> RasterOutput:
    """Alpha-composite projected Gaussians into an image.

    Args:
        projection: output of :func:`project_gaussians`.
        opacities: ``(N,)`` opacities in ``[0, 1]``.
        features: ``(N, C)`` per-Gaussian values to composite (colour, depth, ...).
        background: optional ``(C,)`` background composited behind everything.
    """
    settings = settings or RasterSettings()
    ts = settings.tile_size
    tiles_x = math.ceil(width / ts)
    tiles_y = math.ceil(height / ts)
    n_tiles = tiles_x * tiles_y
    n_pix = ts * ts
    n_channels = features.shape[1]
    device, dtype = features.device, features.dtype

    with torch.no_grad():
        gaussian_ids, pair_tiles, tile_start, tile_count = _bin_gaussians(
            projection.means2d, projection.depths, projection.extents, tiles_x, tiles_y, ts
        )
        n_pairs = gaussian_ids.shape[0]
        if settings.max_per_tile is not None:
            tile_count = tile_count.clamp_max(settings.max_per_tile)
        # Monomial basis of the pixel centres of a tile, relative to the tile centre.
        local = torch.arange(ts, device=device, dtype=dtype) + 0.5 - 0.5 * ts
        pix_y, pix_x = torch.meshgrid(local, local, indexing="ij")
        pix_x, pix_y = pix_x.reshape(-1), pix_y.reshape(-1)  # (P,) row-major inside a tile
        basis = torch.stack(
            [torch.ones_like(pix_x), pix_x, pix_y, pix_x * pix_x, pix_y * pix_y, pix_x * pix_y],
            dim=-1,
        )  # (P, 6)
        center_x = (pair_tiles % tiles_x * ts).to(dtype) + 0.5 * ts
        center_y = (torch.div(pair_tiles, tiles_x, rounding_mode="floor") * ts).to(dtype) + 0.5 * ts
        # Bucket b holds the tiles whose Gaussian count is in (2^(b-1), 2^b]; tiles with
        # at most 8 Gaussians share one bucket to limit the number of Python iterations.
        nonempty = torch.nonzero(tile_count > 0)[:, 0]
        bucket = torch.ceil(torch.log2(tile_count[nonempty].to(torch.float64))).to(torch.int64)
        bucket = bucket.clamp_min(3)
        # Index n_pairs designates the padding entry appended below.
        gaussian_ids = torch.cat([gaussian_ids, gaussian_ids.new_zeros(1)])

    # With (du, dv) the offset from the Gaussian centre to the tile centre and (px, py) the
    # pixel position relative to the tile centre, the exponent of a (Gaussian, tile) pair,
    #   -1/2 [a (du + px)^2 + c (dv + py)^2] - b (du + px)(dv + py) + log o,
    # is a polynomial in (px, py). Its coefficients are computed once for all pairs.
    # The clamp keeps the gradient of log(o) finite for opacities that are exactly zero.
    pair_gaussians = gaussian_ids[:n_pairs]
    log_opacity = torch.log((opacities * projection.compensation).clamp_min(1e-10))
    mean = projection.means2d[pair_gaussians]
    du = center_x - mean[:, 0]
    dv = center_y - mean[:, 1]
    ca, cb, cc = projection.conics[pair_gaussians].unbind(-1)
    coeffs = torch.stack(
        [
            -0.5 * (ca * du * du + cc * dv * dv) - cb * du * dv + log_opacity[pair_gaussians],
            -ca * du - cb * dv,
            -cc * dv - cb * du,
            -0.5 * ca,
            -0.5 * cc,
            -cb,
        ],
        dim=-1,
    )  # (pairs, 6)
    pad = coeffs.new_zeros(1, 6)
    pad[0, 0] = _PAD_EXPONENT
    coeffs = torch.cat([coeffs, pad])

    tile_lists: list[Tensor] = []
    colors: list[Tensor] = []
    alphas: list[Tensor] = []
    for b in torch.unique(bucket).tolist():
        tiles_b = nonempty[bucket == b]
        k = int(tile_count[tiles_b].max())
        slot = torch.arange(k, device=device)
        chunk = max(1, settings.chunk_elements // (k * n_pix))
        for tiles in torch.split(tiles_b, chunk):
            index = tile_start[tiles][:, None] + slot
            index = torch.where(slot < tile_count[tiles][:, None], index, n_pairs)  # (S, K)

            power = torch.matmul(basis, coeffs[index].transpose(1, 2))  # (S, P, K)
            alpha = F.threshold(
                torch.exp(power).clamp_max(settings.alpha_max), settings.alpha_min, 0.0
            )
            transmittance = torch.cumprod(1.0 - alpha, dim=2)
            weights = alpha * torch.cat(
                [torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=2
            )
            colors.append(torch.bmm(weights, features[gaussian_ids[index]]))  # (S, P, C)
            alphas.append(1.0 - transmittance[..., -1])  # sum of the weights
            tile_lists.append(tiles)

    tile_color = torch.zeros(n_tiles, n_pix, n_channels, dtype=dtype, device=device)
    tile_alpha = torch.zeros(n_tiles, n_pix, dtype=dtype, device=device)
    if tile_lists:
        order = torch.cat(tile_lists)
        tile_color = tile_color.index_copy(0, order, torch.cat(colors))
        tile_alpha = tile_alpha.index_copy(0, order, torch.cat(alphas))

    def to_image(tiles: Tensor) -> Tensor:
        channels = tiles.shape[2:]
        grid = tiles.reshape(tiles_y, tiles_x, ts, ts, *channels)
        grid = grid.permute(0, 2, 1, 3, *range(4, 4 + len(channels)))
        return grid.reshape(tiles_y * ts, tiles_x * ts, *channels)[:height, :width]

    image = to_image(tile_color)
    alpha_map = to_image(tile_alpha)
    if background is not None:
        image = image + (1.0 - alpha_map)[..., None] * background
    return RasterOutput(image, alpha_map)
