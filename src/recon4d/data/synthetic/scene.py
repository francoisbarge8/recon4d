"""A ray-traced scene made of a textured room and moving analytic primitives.

The ray tracer is the *ground-truth* image formation model of the benchmark. It is
deliberately unrelated to the splatting and volume renderers used for reconstruction, so
evaluating them on these images is not an inverse crime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from recon4d.data.synthetic.primitives import (
    DTYPE,
    INTERSECT,
    Motion,
    Pose,
    sample_surface,
    surface_area,
)
from recon4d.data.synthetic.textures import Texture

ROOM_ID = 0
"""Object index of the room; primitives are numbered from 1."""

_FACE_NAMES = ("x_min", "x_max", "floor", "ceiling", "z_min", "z_max")


@dataclass(frozen=True)
class SceneObject:
    name: str
    kind: str
    motion: Motion
    texture: Texture

    @property
    def is_dynamic(self) -> bool:
        return self.motion.is_dynamic


@dataclass(frozen=True)
class Hits:
    """Result of intersecting ``N`` rays with the scene."""

    t: Tensor
    """``(N,)`` ray parameter of the nearest hit (z-depth for z-normalised rays)."""
    obj: Tensor
    """``(N,)`` object index, :data:`ROOM_ID` for the room."""
    local: Tensor
    """``(N, 3)`` hit point in the hit object's local frame (world frame for the room)."""
    normal: Tensor
    """``(N, 3)`` world-space unit normal facing the incoming ray's side."""
    face: Tensor
    """``(N,)`` room face index (0 for primitives)."""


@dataclass
class Scene:
    """Room + primitives + a point light.

    Attributes:
        room_lo, room_hi: corners of the axis-aligned room; the camera lives inside.
        room_textures: one texture per face, ordered as ``x_min, x_max, floor, ceiling,
            z_min, z_max``.
        objects: primitives; object ``i`` of this list has index ``i + 1``.
        light: position of the point light.
        ambient: ambient term in ``[0, 1]``; the diffuse term is scaled by ``1 - ambient``.
        shadows: strength in ``[0, 1]`` of cast shadows (0 disables shadow rays).
    """

    room_lo: tuple[float, float, float]
    room_hi: tuple[float, float, float]
    room_textures: tuple[Texture, ...]
    objects: list[SceneObject] = field(default_factory=list)
    light: tuple[float, float, float] = (0.0, 2.7, 0.0)
    ambient: float = 0.55
    shadows: float = 0.0

    def __post_init__(self) -> None:
        if len(self.room_textures) != 6:
            raise ValueError("room_textures must have one entry per face (6)")

    # ------------------------------------------------------------------ bookkeeping

    @property
    def num_objects(self) -> int:
        """Number of indexable objects, including the room."""
        return len(self.objects) + 1

    def dynamic_ids(self) -> list[int]:
        return [i + 1 for i, o in enumerate(self.objects) if o.is_dynamic]

    def poses(self, t: float) -> list[Pose]:
        return [o.motion.pose(t) for o in self.objects]

    # ----------------------------------------------------------------- intersection

    def _intersect_room(self, origins: Tensor, dirs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        lo = torch.tensor(self.room_lo, dtype=DTYPE)
        hi = torch.tensor(self.room_hi, dtype=DTYPE)
        safe = torch.where(dirs.abs() < 1e-14, torch.full_like(dirs, 1e-14), dirs)
        t_hi = torch.maximum((lo - origins) / safe, (hi - origins) / safe)
        t_exit, axis = t_hi.min(dim=-1)
        positive = torch.gather(dirs, 1, axis[:, None])[:, 0] > 0
        face = 2 * axis + positive.to(torch.int64)
        normal = torch.zeros_like(origins)
        # Inward-facing normal of the wall the ray exits through.
        normal.scatter_(1, axis[:, None], torch.where(positive, -1.0, 1.0)[:, None].to(DTYPE))
        return t_exit, normal, face

    def intersect(self, origins: Tensor, dirs: Tensor, t: float) -> Hits:
        """Nearest intersection of rays ``origins + s * dirs`` with the scene at time ``t``."""
        origins = origins.to(DTYPE)
        dirs = dirs.to(DTYPE)
        best_t, best_normal, face = self._intersect_room(origins, dirs)
        best_obj = torch.zeros(origins.shape[0], dtype=torch.int64)
        best_local = origins + best_t[:, None] * dirs
        for index, (obj, pose) in enumerate(zip(self.objects, self.poses(t), strict=True), 1):
            o_local = pose.to_local(origins)
            d_local = pose.dir_to_local(dirs)
            t_hit, n_local = INTERSECT[obj.kind](o_local, d_local)
            closer = t_hit < best_t
            if not closer.any():
                continue
            t_safe = torch.where(closer, t_hit, torch.zeros_like(t_hit))
            best_t = torch.where(closer, t_hit, best_t)
            best_obj = torch.where(closer, torch.full_like(best_obj, index), best_obj)
            best_local = torch.where(
                closer[:, None], o_local + t_safe[:, None] * d_local, best_local
            )
            n_safe = torch.where(closer[:, None], n_local, torch.ones_like(n_local))
            best_normal = torch.where(closer[:, None], pose.normal_to_world(n_safe), best_normal)
            face = torch.where(closer, torch.zeros_like(face), face)
        return Hits(best_t, best_obj, best_local, best_normal, face)

    # -------------------------------------------------------------------- transport

    def local_to_world(self, obj: Tensor, local: Tensor, t: float) -> Tensor:
        """World position at time ``t`` of surface points given by ``(object index, local)``."""
        world = local.to(DTYPE).clone()
        for index, pose in enumerate(self.poses(t), 1):
            mask = obj == index
            if mask.any():
                world[mask] = pose.to_world(local[mask].to(DTYPE))
        return world

    def is_dynamic(self, obj: Tensor) -> Tensor:
        """Boolean mask telling which object indices belong to moving objects."""
        table = torch.zeros(self.num_objects, dtype=torch.bool)
        for index in self.dynamic_ids():
            table[index] = True
        return table[obj]

    def visible_from(self, points: Tensor, eye: Tensor, t: float, tol: float = 2e-3) -> Tensor:
        """Whether ``points (N, 3)`` are unoccluded when seen from ``eye`` at time ``t``."""
        points = points.to(DTYPE)
        eye = eye.to(DTYPE)
        direction = points - eye
        distance = direction.norm(dim=-1).clamp_min(1e-12)
        hits = self.intersect(eye.expand_as(points), direction / distance[:, None], t)
        return hits.t >= distance - tol

    # ---------------------------------------------------------------------- shading

    def albedo(self, hits: Hits) -> Tensor:
        color = torch.zeros(hits.t.shape[0], 3, dtype=torch.float32)
        room = hits.obj == ROOM_ID
        for face_index, texture in enumerate(self.room_textures):
            mask = room & (hits.face == face_index)
            if mask.any():
                color[mask] = texture.albedo(hits.local[mask])
        for index, obj in enumerate(self.objects, 1):
            mask = hits.obj == index
            if mask.any():
                color[mask] = obj.texture.albedo(hits.local[mask])
        return color

    def shade(self, hits: Hits, origins: Tensor, dirs: Tensor, t: float) -> Tensor:
        """Lambertian shading with an ambient term and optional cast shadows; ``(N, 3)`` RGB."""
        points = origins.to(DTYPE) + hits.t[:, None] * dirs.to(DTYPE)
        to_light = torch.tensor(self.light, dtype=DTYPE) - points
        distance = to_light.norm(dim=-1).clamp_min(1e-12)
        to_light = to_light / distance[:, None]
        diffuse = (hits.normal * to_light).sum(dim=-1).clamp_min(0.0)
        if self.shadows > 0:
            start = points + 1e-4 * hits.normal
            occluder = self.intersect(start, to_light, t)
            lit = occluder.t >= distance - 1e-3
            diffuse = diffuse * torch.where(lit, 1.0, 1.0 - self.shadows)
        intensity = self.ambient + (1.0 - self.ambient) * diffuse
        return (self.albedo(hits) * intensity[:, None].to(torch.float32)).clamp(0.0, 1.0)

    # --------------------------------------------------------------------- sampling

    def sample_surface_points(
        self,
        n: int,
        rng: np.random.RandomState,
        t: float = 0.0,
        dynamic_fraction: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Sample material points on the scene surfaces.

        Points are distributed proportionally to world surface area. With
        ``dynamic_fraction`` set, that fraction of the samples is placed on moving objects
        (useful for point tracks, since moving objects are small compared to the room).

        Returns ``obj (n,)`` indices and ``local (n, 3)`` coordinates.
        """
        lo = np.array(self.room_lo)
        hi = np.array(self.room_hi)
        ext = hi - lo
        # Two faces per axis, in the order x_min, x_max, floor, ceiling, z_min, z_max.
        face_areas = np.repeat([ext[1] * ext[2], ext[0] * ext[2], ext[0] * ext[1]], 2)
        poses = self.poses(t)
        obj_areas = np.array(
            [surface_area(o.kind, p.scale) for o, p in zip(self.objects, poses, strict=True)]
        )
        dyn = np.array([o.is_dynamic for o in self.objects], dtype=bool)
        static_area = face_areas.sum() + obj_areas[~dyn].sum()
        dynamic_area = obj_areas[dyn].sum()
        if dynamic_fraction is None or dynamic_area == 0:
            n_dynamic = round(n * float(dynamic_area / (static_area + dynamic_area)))
        else:
            n_dynamic = round(n * dynamic_fraction)
        n_static = n - n_dynamic

        objs: list[Tensor] = []
        locals_: list[Tensor] = []

        def add_objects(mask: np.ndarray, count: int) -> None:
            if count == 0 or not mask.any():
                return
            ids = np.flatnonzero(mask)
            alloc = rng.multinomial(count, obj_areas[ids] / obj_areas[ids].sum())
            for i, k in zip(ids, alloc, strict=True):
                if k == 0:
                    continue
                local, _ = sample_surface(self.objects[i].kind, poses[i].scale, int(k), rng)
                objs.append(torch.full((int(k),), i + 1, dtype=torch.int64))
                locals_.append(local)

        # Static samples are split between the room faces and the static primitives.
        static_obj_area = obj_areas[~dyn].sum()
        n_room = round(n_static * float(face_areas.sum() / static_area))
        if static_obj_area == 0:
            n_room = n_static
        face_alloc = rng.multinomial(n_room, face_areas / face_areas.sum())
        for face_index, k in enumerate(face_alloc):
            if k == 0:
                continue
            axis, side = divmod(face_index, 2)
            p = lo + rng.uniform(0.0, 1.0, size=(int(k), 3)) * ext
            p[:, axis] = hi[axis] if side else lo[axis]
            objs.append(torch.zeros(int(k), dtype=torch.int64))
            locals_.append(torch.as_tensor(p, dtype=DTYPE))
        add_objects(~dyn, n_static - n_room)
        add_objects(dyn, n_dynamic)
        return torch.cat(objs), torch.cat(locals_)

    def face_name(self, face_index: int) -> str:
        return _FACE_NAMES[face_index]
