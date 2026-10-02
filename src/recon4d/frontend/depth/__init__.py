"""Monocular depth back-ends."""

from recon4d.frontend.depth.base import DepthEstimator
from recon4d.frontend.depth.oracle import DepthNoise, OracleDepth

__all__ = ["DepthEstimator", "DepthNoise", "OracleDepth"]
