"""Bundle adjustment by Levenberg-Marquardt with the Schur complement, in PyTorch.

Unknowns are the camera poses, the 3D points and optionally one focal length shared by all
cameras. The cost is the robustified sum of squared reprojection errors.

The normal equations have the classical arrow structure

    [ U   W ] [ dc ]     [ g_c ]
    [ W^T V ] [ dp ] = - [ g_p ]

where ``V`` is block-diagonal (one 3x3 block per point). Eliminating the points gives the
reduced camera system ``(U - W V^-1 W^T) dc = -(g_c - W V^-1 g_p)``, a dense system whose
size only depends on the number of cameras. Jacobians are analytic (and verified against
autograd in the tests); everything is batched tensor algebra in float64.

Poses are updated on the manifold with a left perturbation,
``R <- exp(w) R,  t <- exp(w) t + v``, so that the Jacobian of a camera-frame point
``p = R X + t`` is ``dp/dw = -[p]_x`` and ``dp/dv = I``.

The coupling ``W`` is stored densely as ``(points, camera parameters, 3)``, which is simple
and fast for the problem sizes of this project (tens of cameras, a few thousand points).

Monocular depth prior
---------------------
Point tracks alone constrain the *shape* of the reconstruction weakly along one direction:
trading camera translation against rotation (and stretching the scene in depth) changes
the reprojection error little, and the systematic errors of real trackers push the
solution along that valley. A monocular depth network, on the other hand, knows the
relative depth of the points seen in a frame well, although it knows neither the scale
(nor, for affine-invariant models, the shift) of each frame. :class:`DepthPrior` adds, for
every observation with a predicted inverse depth ``q``, the residual

    ``weight * (z * (a_c q + b_c) - 1)``

where ``z`` is the depth of the point in camera ``c`` and ``(a_c, b_c)`` are per-camera
alignment parameters estimated jointly with everything else. This is the relative error
of the aligned prediction; it ties the depth relief of the reconstruction to the network's
while leaving every frame free to choose its own scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.geometry.rotations import skew, so3_exp

DTYPE = torch.float64


@dataclass
class DepthPrior:
    """Monocular depth attached to the observations of a bundle adjustment problem.

    Attributes:
        inverse: ``(M,)`` predicted inverse depth (or disparity) under each observation.
        valid: ``(M,)`` observations that have a usable prediction.
        params: ``(C, 2)`` initial per-camera alignment ``(a, b)``; the aligned inverse
            depth is ``a * inverse + b``.
        affine: optimise the shift ``b`` as well (affine-invariant disparity); otherwise
            ``b`` keeps its initial value (scale-invariant depth: ``b = 0``).
        weight: how many pixels of reprojection error a unit relative depth error is
            worth; with 10, a 5% depth error costs as much as half a pixel.
        huber: Huber threshold of the depth residual, as a relative error.
    """

    inverse: Tensor
    valid: Tensor
    params: Tensor
    affine: bool = False
    weight: float = 10.0
    huber: float = 0.1


@dataclass
class BAProblem:
    """A bundle adjustment problem.

    Attributes:
        w2c: ``(C, 4, 4)`` initial world-to-camera poses.
        points: ``(P, 3)`` initial 3D points.
        cam_index, point_index: ``(M,)`` camera and point of every observation; a point is
            observed at most once per camera.
        uv: ``(M, 2)`` observed pixel coordinates.
        K: ``(3, 3)`` intrinsics (``fx == fy`` is assumed when the focal is optimised).
        fixed_cameras: ``(C,)`` True for cameras held fixed (gauge anchoring).
        depth: optional monocular depth prior.
    """

    w2c: Tensor
    points: Tensor
    cam_index: Tensor
    point_index: Tensor
    uv: Tensor
    K: Tensor
    fixed_cameras: Tensor
    depth: DepthPrior | None = None


@dataclass
class BAResult:
    w2c: Tensor
    points: Tensor
    K: Tensor
    initial_cost: float
    final_cost: float
    iterations: int
    residuals: Tensor
    """``(M,)`` final reprojection error of every observation, in pixels."""
    depth_params: Tensor | None = None
    """``(C, 2)`` refined per-camera depth alignment, when a depth prior was given."""


def reprojection_residuals(
    R: Tensor,
    t: Tensor,
    points: Tensor,
    K: Tensor,
    cam_index: Tensor,
    point_index: Tensor,
    uv: Tensor,
) -> tuple[Tensor, Tensor]:
    """Residuals ``projection - observation (M, 2)`` and camera-frame points ``(M, 3)``."""
    cam = (R[cam_index] @ points[point_index][..., None])[..., 0] + t[cam_index]
    z = cam[:, 2].clamp_min(1e-9)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return torch.stack([u, v], dim=-1) - uv, cam


def projection_jacobians(
    R: Tensor, cam: Tensor, K: Tensor, cam_index: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Analytic Jacobians of the reprojection residuals for every observation.

    Returns ``J_pose (M, 2, 6)`` (rotation then translation), ``J_point (M, 2, 3)`` and
    ``J_focal (M, 2)`` (derivative w.r.t. a focal length shared by ``fx`` and ``fy``).
    """
    x, y = cam[:, 0], cam[:, 1]
    z = cam[:, 2].clamp_min(1e-9)
    fx, fy = K[0, 0], K[1, 1]
    zero = torch.zeros_like(z)
    J_proj = torch.stack(
        [
            torch.stack([fx / z, zero, -fx * x / z**2], dim=-1),
            torch.stack([zero, fy / z, -fy * y / z**2], dim=-1),
        ],
        dim=-2,
    )  # (M, 2, 3)
    J_pose = torch.cat([-J_proj @ skew(cam), J_proj], dim=-1)
    J_point = J_proj @ R[cam_index]
    J_focal = torch.stack([x / z, y / z], dim=-1)
    return J_pose, J_point, J_focal


def depth_residuals(cam: Tensor, params: Tensor, prior: DepthPrior, cam_index: Tensor) -> Tensor:
    """Relative error ``z (a q + b) - 1`` of the aligned depth prediction, ``(M,)``."""
    a, b = params[cam_index, 0], params[cam_index, 1]
    error = cam[:, 2] * (a * prior.inverse + b) - 1.0
    return torch.where(prior.valid, error, torch.zeros_like(error))


def depth_jacobians(
    R: Tensor, cam: Tensor, params: Tensor, prior: DepthPrior, cam_index: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Jacobians of :func:`depth_residuals`.

    Returns ``J_pose (M, 6)``, ``J_point (M, 3)`` and ``J_params (M, 2)`` (w.r.t. the
    camera's ``(a, b)``); rows of observations without a prediction are zero.
    """
    a, b = params[cam_index, 0], params[cam_index, 1]
    slope = torch.where(prior.valid, a * prior.inverse + b, torch.zeros_like(a))
    z = torch.where(prior.valid, cam[:, 2], torch.zeros_like(a))
    zero = torch.zeros_like(a)
    # z is the third coordinate of p = R X + t: dz/dw = (p_y, -p_x, 0), dz/dv = (0, 0, 1).
    dz_pose = torch.stack([cam[:, 1], -cam[:, 0], zero, zero, zero, torch.ones_like(a)], dim=-1)
    J_pose = slope[:, None] * dz_pose
    J_point = slope[:, None] * R[cam_index][:, 2, :]
    J_params = torch.stack([z * prior.inverse, z], dim=-1)
    return J_pose, J_point, J_params


def _huber_weights(squared: Tensor, delta: float) -> Tensor:
    """IRLS weights of the Huber loss, as a function of the squared residual norm."""
    norm = torch.sqrt(squared.clamp_min(1e-24))
    return torch.where(norm <= delta, torch.ones_like(norm), delta / norm)


def _huber_cost(squared: Tensor, delta: float) -> Tensor:
    norm = torch.sqrt(squared.clamp_min(1e-24))
    return torch.where(norm <= delta, 0.5 * squared, delta * (norm - 0.5 * delta)).sum()


def bundle_adjust(
    problem: BAProblem,
    max_iterations: int = 30,
    huber_delta: float | None = 1.0,
    optimize_focal: bool = False,
    initial_damping: float = 1e-4,
    tolerance: float = 1e-8,
) -> BAResult:
    """Minimise the (Huber) reprojection error over poses, points and optionally the focal.

    With ``problem.depth`` set, the relative error of the aligned monocular depth is added
    to the cost and the per-camera alignment parameters are optimised too.

    Args:
        huber_delta: threshold (pixels) of the Huber loss; ``None`` uses plain least squares.
        optimize_focal: also refine a single focal length ``fx = fy``.
        initial_damping: initial Levenberg-Marquardt damping (multiplies the diagonal).
        tolerance: stop when the relative cost decrease falls below this value.
    """
    cam_index, point_index = problem.cam_index, problem.point_index
    n_cams, n_points = problem.w2c.shape[0], problem.points.shape[0]
    R = problem.w2c[:, :3, :3].to(DTYPE).clone()
    t = problem.w2c[:, :3, 3].to(DTYPE).clone()
    points = problem.points.to(DTYPE).clone()
    K = problem.K.to(DTYPE).clone()
    uv = problem.uv.to(DTYPE)
    prior = problem.depth
    if prior is not None:
        prior = DepthPrior(
            prior.inverse.to(DTYPE),
            prior.valid,
            prior.params.to(DTYPE),
            prior.affine,
            prior.weight,
            prior.huber,
        )
        depth_params = prior.params.clone()
    else:
        depth_params = None

    # Layout of the camera-side parameter vector: for each camera a block
    # [rotation (3), translation (3)] (+ [a, b] with a depth prior), then the focal.
    block = 6 if prior is None else 8
    n_block_params = block * n_cams
    n_cam_params = n_block_params + (1 if optimize_focal else 0)
    free_block = torch.ones(n_cams, block, dtype=torch.bool)
    free_block[problem.fixed_cameras, :6] = False
    if prior is not None and not prior.affine:
        free_block[:, 7] = False
    free = free_block.reshape(-1)
    if optimize_focal:
        free = torch.cat([free, torch.ones(1, dtype=torch.bool)])
    free_index = torch.nonzero(free)[:, 0]
    block_rows = torch.arange(n_block_params).reshape(n_cams, block)

    def evaluate(R_, t_, points_, K_, params_):
        """Residuals (M, rows), camera-frame points and total robust cost."""
        res, cam_ = reprojection_residuals(R_, t_, points_, K_, cam_index, point_index, uv)
        squared = (res**2).sum(dim=-1)
        cost_ = 0.5 * squared.sum() if huber_delta is None else _huber_cost(squared, huber_delta)
        if prior is not None:
            scaled = prior.weight * depth_residuals(cam_, params_, prior, cam_index)
            cost_ = cost_ + _huber_cost(scaled**2, prior.weight * prior.huber)
            res = torch.cat([res, scaled[:, None]], dim=1)
        return res, cam_, cost_

    res, cam, cost = evaluate(R, t, points, K, depth_params)
    initial_cost = float(cost)
    damping = initial_damping
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        squared = (res[:, :2] ** 2).sum(dim=-1)
        w = (
            torch.ones_like(squared)
            if huber_delta is None
            else _huber_weights(squared, huber_delta)
        )
        J_pose, J_point, J_focal = projection_jacobians(R, cam, K, cam_index)
        weights = w[:, None].expand(-1, 2)
        J_cam = J_pose
        if prior is not None:
            d_pose, d_point, d_params = depth_jacobians(R, cam, depth_params, prior, cam_index)
            lam = prior.weight
            zeros = torch.zeros(J_pose.shape[0], 2, 2, dtype=DTYPE)
            J_cam = torch.cat(
                [
                    torch.cat([J_pose, zeros], dim=2),
                    lam * torch.cat([d_pose, d_params], dim=1)[:, None, :],
                ],
                dim=1,
            )  # (M, 3, 8)
            J_point = torch.cat([J_point, lam * d_point[:, None, :]], dim=1)  # (M, 3, 3)
            J_focal = torch.cat([J_focal, torch.zeros_like(J_focal[:, :1])], dim=1)
            w_depth = _huber_weights(res[:, 2] ** 2, lam * prior.huber) * prior.valid
            weights = torch.cat([weights, w_depth[:, None]], dim=1)
        wJc = J_cam * weights[:, :, None]
        wJp = J_point * weights[:, :, None]

        # Camera block U and gradient g_c.
        U = torch.zeros(n_cam_params, n_cam_params, dtype=DTYPE)
        U_blocks = torch.zeros(n_cams, block, block, dtype=DTYPE)
        U_blocks.index_add_(0, cam_index, J_cam.transpose(1, 2) @ wJc)
        U[block_rows[:, :, None], block_rows[:, None, :]] = U_blocks
        g_c = torch.zeros(n_cam_params, dtype=DTYPE)
        g_block = torch.zeros(n_cams, block, dtype=DTYPE)
        g_block.index_add_(0, cam_index, (wJc.transpose(1, 2) @ res[..., None])[..., 0])
        g_c[:n_block_params] = g_block.reshape(-1)

        # Point blocks V and gradient g_p.
        V = torch.zeros(n_points, 3, 3, dtype=DTYPE)
        V.index_add_(0, point_index, J_point.transpose(1, 2) @ wJp)
        g_p = torch.zeros(n_points, 3, dtype=DTYPE)
        g_p.index_add_(0, point_index, (wJp.transpose(1, 2) @ res[..., None])[..., 0])

        # Coupling W[p, camera parameter, :]: one (block x 3) matrix per observation.
        W = torch.zeros(n_points, n_cam_params, 3, dtype=DTYPE)
        W[point_index[:, None], block_rows[cam_index]] = J_cam.transpose(1, 2) @ wJp

        if optimize_focal:
            f = n_block_params
            wJf = J_focal * weights
            cross = torch.zeros(n_cams, block, dtype=DTYPE)
            cross.index_add_(0, cam_index, (J_cam.transpose(1, 2) @ wJf[..., None])[..., 0])
            U[f, f] = (J_focal * wJf).sum()
            U[f, :n_block_params] = cross.reshape(-1)
            U[:n_block_params, f] = cross.reshape(-1)
            g_c[f] = (wJf * res).sum()
            focal_point = torch.zeros(n_points, 3, dtype=DTYPE)
            focal_point.index_add_(
                0, point_index, (J_point.transpose(1, 2) @ wJf[..., None])[..., 0]
            )
            W[:, f, :] = focal_point

        W_flat = W.permute(1, 0, 2).reshape(n_cam_params, n_points * 3)
        accepted = False
        decrease = 0.0
        for _attempt in range(10):
            # Marquardt damping scales the diagonal, which keeps the step scale-invariant.
            V_inv = torch.linalg.inv(
                V + damping * torch.diag_embed(V.diagonal(dim1=1, dim2=2).clamp_min(1e-9))
            )
            WV = W @ V_inv  # (P, C', 3)
            WV_flat = WV.permute(1, 0, 2).reshape(n_cam_params, n_points * 3)
            schur = U + damping * torch.diag(U.diagonal().clamp_min(1e-9)) - WV_flat @ W_flat.T
            rhs = g_c - WV_flat @ g_p.reshape(-1)
            delta_c = torch.zeros(n_cam_params, dtype=DTYPE)
            try:
                delta_c[free_index] = torch.linalg.solve(
                    schur[free_index][:, free_index], -rhs[free_index]
                )
            except RuntimeError:  # singular reduced system: increase the damping
                damping *= 10.0
                continue
            coupling = (W_flat.T @ delta_c).reshape(n_points, 3)
            delta_p = -(V_inv @ (g_p + coupling)[..., None])[..., 0]

            step = delta_c[:n_block_params].reshape(n_cams, block)
            dR = so3_exp(step[:, :3])
            R_new = dR @ R
            t_new = (dR @ t[..., None])[..., 0] + step[:, 3:6]
            points_new = points + delta_p
            params_new = None if prior is None else depth_params + step[:, 6:8]
            K_new = K.clone()
            if optimize_focal:
                K_new[0, 0] += delta_c[-1]
                K_new[1, 1] += delta_c[-1]
            res_new, cam_new, cost_new = evaluate(R_new, t_new, points_new, K_new, params_new)
            if torch.isfinite(cost_new) and cost_new < cost and K_new[0, 0] > 0:
                decrease = float((cost - cost_new) / cost.clamp_min(1e-30))
                R, t, points, K, depth_params = R_new, t_new, points_new, K_new, params_new
                res, cam, cost = res_new, cam_new, cost_new
                damping = max(damping / 10.0, 1e-12)
                accepted = True
                break
            damping *= 10.0
        if not accepted or decrease < tolerance:
            break

    w2c = torch.eye(4, dtype=DTYPE).repeat(n_cams, 1, 1)
    w2c[:, :3, :3] = R
    w2c[:, :3, 3] = t
    return BAResult(
        w2c,
        points,
        K,
        initial_cost,
        float(cost),
        iterations,
        res[:, :2].norm(dim=-1),
        depth_params,
    )
