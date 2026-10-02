"""Procedural 4D benchmark with exact ground truth.

A sequence is fully determined by a :class:`SyntheticConfig`; nothing is stored on disk.
"""

from recon4d.data.synthetic.builder import (
    SyntheticConfig,
    build_synthetic_sequence,
    render_sequence,
)
from recon4d.data.synthetic.oracle import GTTracks, SceneOracle, SurfaceCloud
from recon4d.data.synthetic.render import RenderedFrame, render_frame
from recon4d.data.synthetic.scene import Scene, SceneObject
from recon4d.data.synthetic.scenes import SCENE_NAMES, SceneSpec, build_scene, random_scene
from recon4d.data.synthetic.textures import Texture

__all__ = [
    "SCENE_NAMES",
    "GTTracks",
    "RenderedFrame",
    "Scene",
    "SceneObject",
    "SceneOracle",
    "SceneSpec",
    "SurfaceCloud",
    "SyntheticConfig",
    "Texture",
    "build_scene",
    "build_synthetic_sequence",
    "random_scene",
    "render_frame",
    "render_sequence",
]
