"""Low-rank scene motion: a few rigid trajectories shared by all dynamic Gaussians.

Following *Shape of Motion* (Wang et al., 2024), the motion of a dynamic Gaussian is a
convex combination of ``B`` rigid "basis" trajectories ``T_b(t)`` in SE(3):

``x(t) = (sum_b w_b R_b(t)) x_c + sum_b w_b t_b(t)``

where ``x_c`` is the position in a canonical frame and ``w = softmax(logits)`` are
per-Gaussian coefficients. A rigid object is explained by a single basis, an articulated or
softly deforming one by a blend. The representation has only ``9 B T`` motion parameters,
which is what makes monocular 4D reconstruction well posed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from recon4d.geometry.align import batched_kabsch
from recon4d.geometry.rotations import rot6d_to_rotmat, rotmat_to_rot6d


class MotionBases(nn.Module):
    """``B`` rigid trajectories over ``T`` frames, mapping canonical space to each frame.

    Rotations are stored in the continuous 6D representation, which blends linearly and is
    re-orthonormalised after blending.
    """

    def __init__(self, rotations: Tensor, translations: Tensor) -> None:
        """
        Args:
            rotations: ``(B, T, 3, 3)`` rotation matrices.
            translations: ``(B, T, 3)``.
        """
        super().__init__()
        self.rot6d = nn.Parameter(rotmat_to_rot6d(rotations).clone())
        self.trans = nn.Parameter(translations.clone())

    @classmethod
    def identity(cls, num_bases: int, num_frames: int) -> MotionBases:
        eye = torch.eye(3).expand(num_bases, num_frames, 3, 3)
        return cls(eye, torch.zeros(num_bases, num_frames, 3))

    @property
    def num_bases(self) -> int:
        return self.rot6d.shape[0]

    @property
    def num_frames(self) -> int:
        return self.rot6d.shape[1]

    def blend(self, logits: Tensor, frame: int) -> tuple[Tensor, Tensor]:
        """Per-Gaussian rigid transforms at ``frame``.

        Args:
            logits: ``(N, B)`` un-normalised blending coefficients.

        Returns:
            ``R (N, 3, 3)`` and ``t (N, 3)`` such that ``x(frame) = R @ x_canonical + t``.
        """
        weights = torch.softmax(logits, dim=-1)
        rotation = rot6d_to_rotmat(weights @ self.rot6d[:, frame])
        return rotation, weights @ self.trans[:, frame]

    def transform(self, logits: Tensor, canonical: Tensor, frame: int) -> Tensor:
        """Positions ``(N, 3)`` of canonical points at ``frame``."""
        rotation, translation = self.blend(logits, frame)
        return (rotation @ canonical[..., None])[..., 0] + translation

    def trajectories(self, logits: Tensor, canonical: Tensor) -> Tensor:
        """Positions ``(N, T, 3)`` of canonical points at every frame."""
        weights = torch.softmax(logits, dim=-1)
        rotation = rot6d_to_rotmat(torch.einsum("nb,btk->ntk", weights, self.rot6d))
        translation = torch.einsum("nb,btk->ntk", weights, self.trans)
        return (rotation @ canonical[:, None, :, None])[..., 0] + translation

    def smoothness(self) -> Tensor:
        """Mean squared acceleration of the basis trajectories (translation and rotation)."""
        if self.num_frames < 3:
            return self.trans.new_zeros(())
        accel_t = self.trans[:, 2:] - 2.0 * self.trans[:, 1:-1] + self.trans[:, :-2]
        accel_r = self.rot6d[:, 2:] - 2.0 * self.rot6d[:, 1:-1] + self.rot6d[:, :-2]
        return (accel_t**2).mean() + (accel_r**2).mean()


def kmeans(points: Tensor, k: int, iterations: int = 30, seed: int = 0) -> Tensor:
    """Plain k-means with k-means++ seeding; returns cluster labels ``(N,)``."""
    n = points.shape[0]
    k = min(k, n)
    generator = torch.Generator().manual_seed(seed)
    centers = points[torch.randint(n, (1,), generator=generator)]
    for _ in range(k - 1):
        d2 = torch.cdist(points, centers).amin(dim=1) ** 2
        if d2.sum() <= 0:
            choice = torch.randint(n, (1,), generator=generator)
        else:
            choice = torch.multinomial(d2 / d2.sum(), 1, generator=generator)
        centers = torch.cat([centers, points[choice]])
    labels = torch.zeros(n, dtype=torch.int64)
    for _ in range(iterations):
        new_labels = torch.cdist(points, centers).argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = points[labels == j]
            if members.shape[0] > 0:
                centers[j] = members.mean(dim=0)
    return labels


def init_motion_bases(
    tracks_xyz: Tensor,
    visible: Tensor,
    num_bases: int,
    canonical_frame: int | None = None,
    seed: int = 0,
) -> tuple[MotionBases, Tensor, Tensor, int]:
    """Initialise motion bases from 3D point tracks.

    1. The canonical frame is the one in which most tracks are visible.
    2. Tracks are clustered by k-means on their (zero-filled) frame-to-frame velocities,
       so that points moving alike end up together.
    3. For every cluster and frame, the rigid transform from the canonical frame is the
       closed-form Procrustes alignment of the cluster's visible points. Frames where a
       cluster is not observed copy the nearest observed frame.

    Args:
        tracks_xyz: ``(N, T, 3)`` world positions of tracked points (arbitrary where
            invisible).
        visible: ``(N, T)`` validity of those positions.
        num_bases: number of rigid trajectories ``B``.

    Returns:
        ``(bases, logits (N, B), canonical (N, 3), canonical_frame)``. Only tracks visible
        in the canonical frame have a meaningful canonical position; for the others the
        position at their best-observed frame is mapped back with their cluster's motion.
    """
    n, n_frames, _ = tracks_xyz.shape
    dtype = tracks_xyz.dtype
    if canonical_frame is None:
        canonical_frame = int(visible.sum(dim=0).argmax())

    both = visible[:, 1:] & visible[:, :-1]
    velocity = torch.where(both[..., None], tracks_xyz[:, 1:] - tracks_xyz[:, :-1], 0.0)
    labels = kmeans(velocity.reshape(n, -1), num_bases, seed=seed)
    num_bases = max(int(labels.max()) + 1, 1) if n > 0 else num_bases

    rotations = torch.eye(3, dtype=dtype).repeat(num_bases, n_frames, 1, 1)
    translations = torch.zeros(num_bases, n_frames, 3, dtype=dtype)
    for b in range(num_bases):
        members = labels == b
        anchor = members & visible[:, canonical_frame]
        if anchor.sum() < 3:
            continue
        src = tracks_xyz[anchor, canonical_frame]  # (M, 3)
        dst = tracks_xyz[anchor].transpose(0, 1)  # (T, M, 3)
        weights = visible[anchor].transpose(0, 1).to(dtype)
        R, t = batched_kabsch(src[None].expand(n_frames, -1, -1), dst, weights)
        observed = weights.sum(dim=1) >= 3
        # Fill unobserved frames with the nearest observed one.
        observed_index = torch.nonzero(observed)[:, 0]
        if observed_index.numel() == 0:
            continue
        frames = torch.arange(n_frames)
        nearest = observed_index[(frames[:, None] - observed_index[None]).abs().argmin(dim=1)]
        rotations[b] = R[nearest]
        translations[b] = t[nearest]

    bases = MotionBases(rotations, translations)

    # Canonical positions: observed directly, or mapped back from the best other frame.
    logits = torch.full((n, num_bases), -4.0, dtype=dtype)
    logits[torch.arange(n), labels] = 4.0
    canonical = tracks_xyz[:, canonical_frame].clone()
    missing = ~visible[:, canonical_frame]
    if missing.any():
        first_visible = visible.to(torch.int64).argmax(dim=1)
        for i in torch.nonzero(missing)[:, 0].tolist():
            f = int(first_visible[i])
            R = rotations[labels[i], f]
            t = translations[labels[i], f]
            canonical[i] = R.T @ (tracks_xyz[i, f] - t)
    return bases, logits, canonical, canonical_frame
