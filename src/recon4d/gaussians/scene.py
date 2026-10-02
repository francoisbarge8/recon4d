"""A 4D scene: static Gaussians plus dynamic Gaussians driven by motion bases."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from recon4d.gaussians.model import GaussianCloud
from recon4d.gaussians.motion import MotionBases
from recon4d.gaussians.rasterizer import Projection, RasterSettings
from recon4d.gaussians.render import Camera, render_gaussians
from recon4d.geometry.rotations import quat_to_rotmat

MOTION_LOGITS = "motion_logits"
"""Name of the per-Gaussian motion coefficients stored in the dynamic cloud."""


@dataclass
class SceneRender:
    color: Tensor
    """``(H, W, 3)``"""
    depth: Tensor
    """``(H, W)`` expected z-depth."""
    alpha: Tensor
    """``(H, W)`` accumulated opacity."""
    dynamic: Tensor | None
    """``(H, W)`` share of each pixel explained by dynamic Gaussians (None if static)."""
    positions: list[Tensor]
    """For each requested frame, ``(H, W, 3)`` expected world position *at that frame* of
    the surface seen through each pixel (i.e. dense 3D correspondences across time)."""
    projection: Projection
    n_static: int


class GaussianScene(nn.Module):
    """Static background Gaussians and, optionally, moving foreground Gaussians.

    Dynamic Gaussians live in a canonical space and are carried to frame ``t`` by the
    blend of :class:`MotionBases` selected by their ``motion_logits``; their orientation
    rotates with them.
    """

    def __init__(
        self,
        static: GaussianCloud,
        dynamic: GaussianCloud | None = None,
        motion: MotionBases | None = None,
    ) -> None:
        super().__init__()
        if (dynamic is None) != (motion is None):
            raise ValueError("dynamic Gaussians and motion bases must be given together")
        if dynamic is not None and MOTION_LOGITS not in dynamic.params:
            raise ValueError(f"the dynamic cloud needs a {MOTION_LOGITS!r} parameter")
        self.static = static
        self.dynamic = dynamic
        self.motion = motion

    @property
    def is_dynamic(self) -> bool:
        return self.dynamic is not None and len(self.dynamic) > 0

    def clouds(self) -> dict[str, GaussianCloud]:
        clouds = {"static": self.static}
        if self.dynamic is not None:
            clouds["dynamic"] = self.dynamic
        return clouds

    def dynamic_means(self, frame: int) -> Tensor:
        """World positions ``(N_dyn, 3)`` of the dynamic Gaussians at ``frame``."""
        assert self.dynamic is not None and self.motion is not None
        return self.motion.transform(self.dynamic.params[MOTION_LOGITS], self.dynamic.means, frame)

    def dynamic_trajectories(self) -> Tensor:
        """World positions ``(N_dyn, T, 3)`` of the dynamic Gaussians at every frame."""
        assert self.dynamic is not None and self.motion is not None
        return self.motion.trajectories(self.dynamic.params[MOTION_LOGITS], self.dynamic.means)

    def render(
        self,
        camera: Camera,
        frame: int,
        settings: RasterSettings | None = None,
        background: Tensor | None = None,
        position_frames: tuple[int, ...] = (),
        with_dynamic_map: bool = True,
    ) -> SceneRender:
        """Render the scene at time ``frame`` from ``camera``.

        Args:
            position_frames: frames for which to also render the per-pixel expected 3D
                position of the visible surface (used to supervise motion with 2D tracks).
            with_dynamic_map: also render the dynamic-vs-static indicator.
        """
        static = self.static
        means = [static.means]
        rotations = [quat_to_rotmat(static.quats)]
        scales = [static.scales]
        opacities = [static.opacities]
        colors = [static.colors(camera.center)]
        n_static = len(static)

        dynamic_now = None
        if self.is_dynamic:
            dynamic = self.dynamic
            logits = dynamic.params[MOTION_LOGITS]
            rotation, translation = self.motion.blend(logits, frame)
            dynamic_now = (rotation @ dynamic.means[..., None])[..., 0] + translation
            means.append(dynamic_now)
            rotations.append(rotation @ quat_to_rotmat(dynamic.quats))
            scales.append(dynamic.scales)
            opacities.append(dynamic.opacities)
            colors.append(dynamic.colors(camera.center, dynamic_now))

        extras: list[Tensor] = []
        render_dynamic_map = self.is_dynamic and with_dynamic_map
        if render_dynamic_map:
            indicator = torch.zeros(n_static + len(self.dynamic), 1, dtype=static.means.dtype)
            indicator[n_static:] = 1.0
            extras.append(indicator)
        for target in position_frames:
            at_target = [static.means]
            if self.is_dynamic:
                at_target.append(dynamic_now if target == frame else self.dynamic_means(target))
            extras.append(torch.cat(at_target))

        out = render_gaussians(
            torch.cat(means),
            torch.cat(rotations),
            torch.cat(scales),
            torch.cat(opacities),
            torch.cat(colors),
            camera,
            settings,
            background,
            torch.cat(extras, dim=1) if extras else None,
        )
        offset = 1 if render_dynamic_map else 0
        norm = out.alpha.clamp_min(1e-4)[..., None]
        positions = [
            out.extras[..., offset + 3 * k : offset + 3 * (k + 1)] / norm
            for k in range(len(position_frames))
        ]
        return SceneRender(
            color=out.color,
            depth=out.depth,
            alpha=out.alpha,
            dynamic=out.extras[..., 0] if render_dynamic_map else None,
            positions=positions,
            projection=out.projection,
            n_static=n_static,
        )
