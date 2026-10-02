"""Benchmark scene definitions.

All scenes share one layout grammar so that they belong to a single distribution (the
in-domain depth network is trained on :func:`random_scene` samples of that distribution):

* a 6 x 3 x 6 m textured room, lit by a point light near the ceiling;
* a *centre zone* (radius < 0.55 m) holding static centre-pieces;
* a *ring zone* (0.85 - 1.4 m) reserved for moving objects;
* an *outer zone* (1.75 - 2.3 m) with static props, on the side opposite to the camera;
* a camera orbiting the scene centre at about 2.7 m.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field

import numpy as np
from torch import Tensor

from recon4d.data.synthetic.primitives import (
    Bounce,
    Motion,
    RollOnCircle,
    SlideSpin,
    Squash,
    Static,
)
from recon4d.data.synthetic.scene import Scene, SceneObject
from recon4d.data.synthetic.textures import Color, Texture
from recon4d.data.synthetic.trajectories import fixed_camera, orbit_trajectory

SCENE_NAMES = ("still", "rolling", "sliding", "squash")

_TARGET = (0.0, 0.3, 0.0)
_CAMERA_SIDE_DEG = 90.0  # the camera orbits around this polar angle

_VIVID: tuple[Color, ...] = (
    (0.85, 0.25, 0.22),
    (0.95, 0.58, 0.15),
    (0.95, 0.82, 0.25),
    (0.30, 0.68, 0.35),
    (0.15, 0.62, 0.65),
    (0.22, 0.42, 0.85),
    (0.58, 0.35, 0.78),
    (0.92, 0.45, 0.62),
)
_NEUTRAL: tuple[Color, ...] = ((0.93, 0.93, 0.90), (0.16, 0.18, 0.23))
_WALLS: tuple[tuple[str, Color, Color, float], ...] = (
    ("stripes", (0.60, 0.68, 0.76), (0.40, 0.49, 0.61), 1.1),
    ("noise", (0.78, 0.67, 0.58), (0.58, 0.46, 0.40), 1.4),
    ("dots", (0.63, 0.73, 0.63), (0.42, 0.56, 0.46), 1.6),
    ("checker", (0.73, 0.71, 0.81), (0.52, 0.50, 0.67), 1.3),
)
_PATTERNS = ("checker", "stripes", "dots", "noise")


@dataclass(frozen=True)
class SceneSpec:
    """A scene together with its training trajectory and held-out validation cameras."""

    name: str
    scene: Scene
    description: str
    train_orbit: dict = field(default_factory=dict)
    val_cameras: tuple[dict, ...] = ()

    def train_c2w(self, n_frames: int) -> Tensor:
        return orbit_trajectory(n_frames, _TARGET, **self.train_orbit)

    def val_c2w(self) -> list[Tensor]:
        return [fixed_camera(_TARGET, **params) for params in self.val_cameras]


def _jitter(color: Color, rng: np.random.RandomState, amount: float = 0.05) -> Color:
    c = np.clip(np.array(color) + rng.uniform(-amount, amount, size=3), 0.0, 1.0)
    return (float(c[0]), float(c[1]), float(c[2]))


def _room_textures(rng: np.random.RandomState) -> tuple[Texture, ...]:
    order = rng.permutation(len(_WALLS))
    walls = []
    for k in order:
        pattern, a, b, frequency = _WALLS[k]
        walls.append(
            Texture(
                pattern,
                _jitter(a, rng),
                _jitter(b, rng),
                frequency=frequency * rng.uniform(0.85, 1.15),
                detail=0.22,
                detail_frequency=6.0,
                seed=int(rng.randint(1 << 20)),
            )
        )
    floor = Texture(
        "checker",
        _jitter((0.80, 0.76, 0.68), rng),
        _jitter((0.50, 0.47, 0.44), rng),
        frequency=rng.uniform(1.7, 2.3),
        detail=0.28,
        detail_frequency=7.0,
        seed=int(rng.randint(1 << 20)),
    )
    ceiling = Texture(
        "noise",
        (0.88, 0.88, 0.86),
        (0.78, 0.79, 0.80),
        frequency=0.9,
        detail=0.1,
        seed=int(rng.randint(1 << 20)),
    )
    # Face order: x_min, x_max, floor, ceiling, z_min, z_max.
    return (walls[0], walls[1], floor, ceiling, walls[2], walls[3])


def _object_texture(rng: np.random.RandomState) -> Texture:
    a = _VIVID[rng.randint(len(_VIVID))]
    if rng.uniform() < 0.6:
        b = _NEUTRAL[rng.randint(len(_NEUTRAL))]
    else:
        b = _VIVID[rng.randint(len(_VIVID))]
    pattern = _PATTERNS[rng.randint(len(_PATTERNS))]
    return Texture(
        pattern,
        _jitter(a, rng),
        _jitter(b, rng),
        frequency=rng.uniform(2.0, 3.2),
        detail=0.25,
        detail_frequency=rng.uniform(5.0, 8.0),
        seed=int(rng.randint(1 << 20)),
    )


def _static_object(
    name: str, rng: np.random.RandomState, x: float, z: float, size: float
) -> SceneObject:
    kind = ("box", "cylinder", "sphere")[rng.randint(3)]
    if kind == "sphere":
        scale = (size, size, size)
    elif kind == "box":
        scale = (
            size * rng.uniform(0.7, 1.1),
            size * rng.uniform(0.7, 1.5),
            size * rng.uniform(0.7, 1.1),
        )
    else:
        scale = (size * 0.8, size * rng.uniform(0.9, 1.6), size * 0.8)
    yaw = rng.uniform(0.0, math.pi)
    motion = Static((x, scale[1], z), scale, (0.0, yaw, 0.0))
    return SceneObject(name, kind, motion, _object_texture(rng))


def _static_layout(rng: np.random.RandomState, n_center: int, n_outer: int) -> list[SceneObject]:
    objects: list[SceneObject] = []
    for i in range(n_center):
        radius = rng.uniform(0.0, 0.22) if n_center == 1 else 0.3
        angle = rng.uniform(0.0, 2.0 * math.pi) + i * math.pi
        objects.append(
            _static_object(
                f"center_{i}",
                rng,
                radius * math.cos(angle),
                radius * math.sin(angle),
                rng.uniform(0.2, 0.3),
            )
        )
    # Outer props sit on the far side so that they never hide the scene from the camera.
    slots = np.linspace(205.0, 335.0, n_outer) + rng.uniform(-9.0, 9.0, size=n_outer)
    for i, angle_deg in enumerate(slots):
        radius = rng.uniform(1.8, 2.25)
        angle = math.radians(angle_deg)
        objects.append(
            _static_object(
                f"outer_{i}",
                rng,
                radius * math.cos(angle),
                radius * math.sin(angle),
                rng.uniform(0.24, 0.42),
            )
        )
    return objects


def _dynamic(name: str, kind: str, motion: Motion, rng: np.random.RandomState) -> SceneObject:
    return SceneObject(name, kind, motion, _object_texture(rng))


def _dynamics(name: str, rng: np.random.RandomState) -> list[SceneObject]:
    rad = math.radians
    if name == "still":
        return []
    if name == "rolling":
        front = RollOnCircle((0.0, 0.0), 0.95, 0.25, rad(150.0), rad(42.0))
        back = RollOnCircle((0.0, 0.0), 1.3, 0.2, rad(215.0), rad(310.0))
        return [
            _dynamic("ball_front", "sphere", front, rng),
            _dynamic("ball_back", "sphere", back, rng),
        ]
    if name == "sliding":
        crate = SlideSpin((-0.9, 0.2, 0.7), (0.8, 0.2, 0.9), (0.26, 0.2, 0.19), 0.5, 0.3, math.pi)
        bouncer = Bounce((-1.15, -0.75), (1.05, -1.0), 0.19, 0.55, 2.0, (2.0, 0.5, 3.0))
        return [
            _dynamic("crate", "box", crate, rng),
            _dynamic("bouncer", "sphere", bouncer, rng),
        ]
    if name == "squash":
        jelly = Squash((0.0, 0.88), (0.27, 0.27, 0.27), 0.32, 1.5, 0.55)
        can = SlideSpin((-1.1, 0.22, -0.2), (-0.8, 0.22, 0.65), (0.16, 0.22, 0.16), 0.5, 0.0, 2.5)
        return [
            _dynamic("jelly", "sphere", jelly, rng),
            _dynamic("can", "cylinder", can, rng),
        ]
    raise ValueError(f"unknown scene {name!r}; available: {SCENE_NAMES}")


_DESCRIPTIONS = {
    "still": "static scene (no moving object): camera pose, depth and static 3D quality",
    "rolling": "two balls rolling without slipping, one passing behind the centre-piece",
    "sliding": "a spinning crate sliding on the floor and a bouncing, tumbling ball",
    "squash": "a non-rigid squash-and-stretch blob swaying sideways and a sliding can",
}


def _name_seed(name: str, seed: int) -> int:
    return (zlib.crc32(name.encode()) + 7919 * seed) % (1 << 31)


def build_scene(name: str, seed: int = 0) -> SceneSpec:
    """Build one of the named benchmark scenes (:data:`SCENE_NAMES`)."""
    rng = np.random.RandomState(_name_seed(name, seed))
    objects = _static_layout(rng, n_center=1, n_outer=4) + _dynamics(name, rng)
    scene = Scene(
        room_lo=(-3.0, 0.0, -3.0),
        room_hi=(3.0, 3.0, 3.0),
        room_textures=_room_textures(rng),
        objects=objects,
        light=(0.4, 2.75, 0.6),
    )
    train_orbit = {
        "radius": 2.7,
        "height": 1.35,
        "angle_start_deg": _CAMERA_SIDE_DEG + 27.0,
        "angle_end_deg": _CAMERA_SIDE_DEG - 27.0,
        "radius_end": 2.55,
        "height_end": 1.55,
        "shake": 0.03,
        "seed": _name_seed(name, seed) + 1,
    }
    val_cameras = (
        {"radius": 2.45, "height": 1.9, "angle_deg": _CAMERA_SIDE_DEG + 13.0},
        {"radius": 2.3, "height": 1.05, "angle_deg": _CAMERA_SIDE_DEG - 12.0},
    )
    return SceneSpec(name, scene, _DESCRIPTIONS[name], train_orbit, val_cameras)


def random_scene(seed: int) -> SceneSpec:
    """Sample a scene from the benchmark distribution (used to train the depth network).

    Layout, textures, motions and camera are all randomised, but stay inside the same
    grammar as the named scenes.
    """
    rng = np.random.RandomState(seed)
    objects = _static_layout(rng, n_center=int(rng.randint(1, 3)), n_outer=int(rng.randint(2, 6)))
    for i in range(int(rng.randint(0, 4))):
        radius = rng.uniform(0.14, 0.32)
        path_radius = rng.uniform(0.85, 1.4)
        start = rng.uniform(0.0, 2.0 * math.pi)
        kind = ("sphere", "box", "cylinder")[rng.randint(3)]
        if kind == "sphere" and rng.uniform() < 0.5:
            motion: Motion = RollOnCircle(
                (0.0, 0.0), path_radius, radius, start, start + rng.uniform(-2.0, 2.0)
            )
        else:
            end = start + rng.uniform(-1.2, 1.2)
            height = radius * rng.uniform(0.7, 1.3)
            motion = SlideSpin(
                (path_radius * math.cos(start), height, path_radius * math.sin(start)),
                (path_radius * math.cos(end), height, path_radius * math.sin(end)),
                (radius, height, radius * rng.uniform(0.7, 1.0)),
                0.5,
                rng.uniform(0.0, math.pi),
                rng.uniform(-math.pi, math.pi),
            )
        objects.append(_dynamic(f"mover_{i}", kind, motion, rng))
    scene = Scene(
        room_lo=(-3.0, 0.0, -3.0),
        room_hi=(3.0, 3.0, 3.0),
        room_textures=_room_textures(rng),
        objects=objects,
        light=(rng.uniform(-0.8, 0.8), 2.75, rng.uniform(-0.8, 0.8)),
    )
    center = _CAMERA_SIDE_DEG + rng.uniform(-12.0, 12.0)
    half_arc = rng.uniform(18.0, 30.0)
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    train_orbit = {
        "radius": rng.uniform(2.3, 2.8),
        "height": rng.uniform(1.0, 1.9),
        "angle_start_deg": center + sign * half_arc,
        "angle_end_deg": center - sign * half_arc,
        "radius_end": rng.uniform(2.3, 2.8),
        "height_end": rng.uniform(1.0, 1.9),
        "shake": 0.03,
        "seed": seed + 1,
    }
    return SceneSpec(f"random_{seed}", scene, "random scene", train_orbit, ())
