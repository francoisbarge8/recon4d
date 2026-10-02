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

    def smoothness(self, extent: float = 1.0) -> Tensor:
        """Mean squared acceleration of the basis trajectories (translation and rotation).

        Translations are divided by ``extent`` so that the penalty is scale-free.
        """
        if self.num_frames < 3:
            return self.trans.new_zeros(())
        accel_t = (self.trans[:, 2:] - 2.0 * self.trans[:, 1:-1] + self.trans[:, :-2]) / extent
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


def _robust_kabsch(src: Tensor, dst: Tensor) -> tuple[Tensor, Tensor]:
    """Rigid transform ``dst ~ R src + t`` with one re-weighting pass against outliers."""
    weights = torch.ones(1, src.shape[0], dtype=src.dtype)
    R, t = batched_kabsch(src[None], dst[None], weights)
    residual = ((src @ R[0].T + t[0]) - dst).norm(dim=-1)
    inlier = residual <= 3.0 * residual.median().clamp_min(1e-12)
    if inlier.sum() >= 3 and not inlier.all():
        R, t = batched_kabsch(src[None], dst[None], inlier[None].to(src.dtype))
    return R[0], t[0]


def _linked_bases(points: Tensor, labels: Tensor, num_bases: int, factor: float = 3.0) -> Tensor:
    """``(B, B)`` adjacency of the clusters: True when two clusters touch each other.

    Two clusters are linked when their closest points are no farther apart than ``factor``
    times the typical spacing between neighbouring points, i.e. when they are most likely
    two parts of the same object.
    """
    distance = torch.cdist(points, points)
    distance.fill_diagonal_(float("inf"))
    spacing = distance.amin(dim=1).median()
    linked = torch.zeros(num_bases, num_bases, dtype=torch.bool)
    for a in range(num_bases):
        for b in range(a + 1, num_bases):
            between = distance[labels == a][:, labels == b]
            if between.numel() > 0 and between.min() <= factor * spacing:
                linked[a, b] = linked[b, a] = True
    return linked


def init_motion_bases(
    tracks_xyz: Tensor,
    visible: Tensor,
    num_bases: int,
    canonical_frame: int | None = None,
    seed: int = 0,
) -> tuple[MotionBases, Tensor, Tensor, int]:
    """Initialise motion bases from 3D point tracks, chaining through time.

    Tracks on moving objects are often short: a point on a rolling ball is visible for a
    fraction of a turn only. The initialisation therefore never relies on a track spanning
    the video. Instead it sweeps away from a canonical frame and, at each new frame,

    1. estimates the rigid transform of every basis from the tracks that already have a
       canonical position and are visible in that frame (closed-form Procrustes, with one
       re-weighting pass against outliers);
    2. lets a basis with too few such tracks follow a neighbouring basis of the same
       object (it applies the same frame-to-frame motion), or keep its previous transform
       when it has no such neighbour;
    3. adopts the tracks appearing in that frame: each takes the basis of its nearest
       already-labelled neighbour and gets a canonical position by undoing that basis'
       transform.

    Bases are seeded by k-means on the positions of the tracks visible in the canonical
    frame, so different objects, or different parts of one object, start in different
    bases; parts that move alike simply end up with identical trajectories.

    Args:
        tracks_xyz: ``(N, T, 3)`` world positions of tracked points (arbitrary where
            invisible).
        visible: ``(N, T)`` validity of those positions.
        num_bases: number of rigid trajectories ``B``.
        canonical_frame: reference frame (default: the one with most visible tracks).

    Returns:
        ``(bases, logits (N, B), canonical (N, 3), canonical_frame)``. Tracks that could
        never be attached to a basis get uniform coefficients.
    """
    n, n_frames, _ = tracks_xyz.shape
    dtype = tracks_xyz.dtype
    if canonical_frame is None:
        canonical_frame = int(visible.sum(dim=0).argmax())

    label = torch.full((n,), -1, dtype=torch.int64)
    canonical = torch.zeros(n, 3, dtype=dtype)
    seeds = torch.nonzero(visible[:, canonical_frame])[:, 0]
    num_bases = max(min(num_bases, seeds.numel()), 1)
    linked = torch.zeros(num_bases, num_bases, dtype=torch.bool)
    if seeds.numel() > 0:
        label[seeds] = kmeans(tracks_xyz[seeds, canonical_frame], num_bases, seed=seed)
        canonical[seeds] = tracks_xyz[seeds, canonical_frame]
        if seeds.numel() > 1:
            linked = _linked_bases(canonical[seeds], label[seeds], num_bases)

    rotations = torch.eye(3, dtype=dtype).repeat(num_bases, n_frames, 1, 1)
    translations = torch.zeros(num_bases, n_frames, 3, dtype=dtype)
    sweeps = (
        range(canonical_frame + 1, n_frames),
        range(canonical_frame - 1, -1, -1),
    )
    for sweep in sweeps:
        previous = canonical_frame
        for frame in sweep:
            here = visible[:, frame]
            support = torch.zeros(num_bases, dtype=torch.int64)
            for b in range(num_bases):
                members = here & (label == b)
                support[b] = members.sum()
                if support[b] >= 3:
                    R, t = _robust_kabsch(canonical[members], tracks_xyz[members, frame])
                    rotations[b, frame], translations[b, frame] = R, t
            for b in torch.nonzero(support < 3)[:, 0].tolist():
                donors = linked[b] & (support >= 3)
                if donors.any():
                    # Follow the best supported neighbour: apply its frame-to-frame motion.
                    d = int(torch.where(donors, support, -1).argmax())
                    step = rotations[d, frame] @ rotations[d, previous].T
                    rotations[b, frame] = step @ rotations[b, previous]
                    translations[b, frame] = (
                        step @ (translations[b, previous] - translations[d, previous])
                        + translations[d, frame]
                    )
                else:
                    rotations[b, frame] = rotations[b, previous]
                    translations[b, frame] = translations[b, previous]
            new = here & (label < 0)
            known = here & (label >= 0)
            if new.any() and known.any():
                nearest = torch.cdist(tracks_xyz[new, frame], tracks_xyz[known, frame]).argmin(
                    dim=1
                )
                adopted = label[known][nearest]
                R, t = rotations[adopted, frame], translations[adopted, frame]
                offset = (tracks_xyz[new, frame] - t)[..., None]
                canonical[new] = (R.transpose(1, 2) @ offset)[..., 0]
                label[new] = adopted
            previous = frame

    logits = torch.zeros(n, num_bases, dtype=dtype)
    assigned = label >= 0
    logits[assigned] = -4.0
    logits[assigned, label[assigned]] = 4.0
    orphans = ~assigned
    if orphans.any():
        first = visible[orphans].to(torch.int64).argmax(dim=1)
        canonical[orphans] = tracks_xyz[orphans, first]
    return MotionBases(rotations, translations), logits, canonical, canonical_frame
