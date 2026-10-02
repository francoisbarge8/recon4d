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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from recon4d.geometry.rotations import skew, so3_exp

DTYPE = torch.float64


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
    """

    w2c: Tensor
    points: Tensor
    cam_index: Tensor
    point_index: Tensor
    uv: Tensor
    K: Tensor
    fixed_cameras: Tensor


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
    """Analytic Jacobians of the residuals for every observation.

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

    n_pose_params = 6 * n_cams
    n_cam_params = n_pose_params + (1 if optimize_focal else 0)
    free = (~problem.fixed_cameras).repeat_interleave(6)
    if optimize_focal:
        free = torch.cat([free, torch.ones(1, dtype=torch.bool)])
    free_index = torch.nonzero(free)[:, 0]
    pose_rows = torch.arange(n_pose_params).reshape(n_cams, 6)

    def residuals(R_: Tensor, t_: Tensor, points_: Tensor, K_: Tensor) -> tuple[Tensor, Tensor]:
        return reprojection_residuals(R_, t_, points_, K_, cam_index, point_index, uv)

    def total_cost(res: Tensor) -> Tensor:
        squared = (res**2).sum(dim=-1)
        return 0.5 * squared.sum() if huber_delta is None else _huber_cost(squared, huber_delta)

    res, cam = residuals(R, t, points, K)
    cost = total_cost(res)
    initial_cost = float(cost)
    damping = initial_damping
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        squared = (res**2).sum(dim=-1)
        w = (
            torch.ones_like(squared)
            if huber_delta is None
            else _huber_weights(squared, huber_delta)
        )
        J_pose, J_point, J_focal = projection_jacobians(R, cam, K, cam_index)
        wJc = J_pose * w[:, None, None]
        wJp = J_point * w[:, None, None]

        # Camera block U and gradient g_c.
        U = torch.zeros(n_cam_params, n_cam_params, dtype=DTYPE)
        U_blocks = torch.zeros(n_cams, 6, 6, dtype=DTYPE)
        U_blocks.index_add_(0, cam_index, J_pose.transpose(1, 2) @ wJc)
        U[pose_rows[:, :, None], pose_rows[:, None, :]] = U_blocks
        g_c = torch.zeros(n_cam_params, dtype=DTYPE)
        g_pose = torch.zeros(n_cams, 6, dtype=DTYPE)
        g_pose.index_add_(0, cam_index, (wJc.transpose(1, 2) @ res[..., None])[..., 0])
        g_c[:n_pose_params] = g_pose.reshape(-1)

        # Point blocks V and gradient g_p.
        V = torch.zeros(n_points, 3, 3, dtype=DTYPE)
        V.index_add_(0, point_index, J_point.transpose(1, 2) @ wJp)
        g_p = torch.zeros(n_points, 3, dtype=DTYPE)
        g_p.index_add_(0, point_index, (wJp.transpose(1, 2) @ res[..., None])[..., 0])

        # Coupling W[p, camera parameter, :]: one 6x3 block per observation.
        W = torch.zeros(n_points, n_cam_params, 3, dtype=DTYPE)
        W[point_index[:, None], pose_rows[cam_index]] = J_pose.transpose(1, 2) @ wJp

        if optimize_focal:
            f = n_pose_params
            wJf = J_focal * w[:, None]
            cross = torch.zeros(n_cams, 6, dtype=DTYPE)
            cross.index_add_(0, cam_index, (J_pose.transpose(1, 2) @ wJf[..., None])[..., 0])
            U[f, f] = (J_focal * wJf).sum()
            U[f, :n_pose_params] = cross.reshape(-1)
            U[:n_pose_params, f] = cross.reshape(-1)
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

            pose_step = delta_c[:n_pose_params].reshape(n_cams, 6)
            dR = so3_exp(pose_step[:, :3])
            R_new = dR @ R
            t_new = (dR @ t[..., None])[..., 0] + pose_step[:, 3:]
            points_new = points + delta_p
            K_new = K.clone()
            if optimize_focal:
                K_new[0, 0] += delta_c[-1]
                K_new[1, 1] += delta_c[-1]
            res_new, cam_new = residuals(R_new, t_new, points_new, K_new)
            cost_new = total_cost(res_new)
            if torch.isfinite(cost_new) and cost_new < cost and K_new[0, 0] > 0:
                decrease = float((cost - cost_new) / cost.clamp_min(1e-30))
                R, t, points, K, res, cam, cost = (
                    R_new,
                    t_new,
                    points_new,
                    K_new,
                    res_new,
                    cam_new,
                    cost_new,
                )
                damping = max(damping / 10.0, 1e-12)
                accepted = True
                break
            damping *= 10.0
        if not accepted or decrease < tolerance:
            break

    w2c = torch.eye(4, dtype=DTYPE).repeat(n_cams, 1, 1)
    w2c[:, :3, :3] = R
    w2c[:, :3, 3] = t
    return BAResult(w2c, points, K, initial_cost, float(cost), iterations, res.norm(dim=-1))
