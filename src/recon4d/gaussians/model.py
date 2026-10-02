"""A set of 3D Gaussians and an optimizer that survives adding / removing Gaussians."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor, nn

from recon4d.gaussians.sh import eval_sh, num_sh_coeffs, rgb_to_sh, sh_to_rgb
from recon4d.geometry.rotations import normalize_quat

BASE_PARAMS = ("means", "log_scales", "quats", "opacity_logits", "sh_dc", "sh_rest")


def inverse_sigmoid(x: float | Tensor) -> float | Tensor:
    if isinstance(x, Tensor):
        return torch.log(x / (1.0 - x))
    return math.log(x / (1.0 - x))


def knn_scale(points: Tensor, k: int = 3) -> Tensor:
    """Root-mean-square distance of each point to its ``k`` nearest neighbours."""
    array = points.detach().cpu().numpy().astype(np.float64)
    k = min(k, max(array.shape[0] - 1, 1))
    distances, _ = cKDTree(array).query(array, k=k + 1)
    rms = np.sqrt((distances[:, 1:] ** 2).mean(axis=1))
    return torch.as_tensor(rms, dtype=points.dtype)


class GaussianCloud(nn.Module):
    """``N`` anisotropic Gaussians with spherical-harmonic colour.

    Parameters are stored unconstrained and exposed through activations:

    * ``means (N, 3)``: centres;
    * ``log_scales (N, 3)``: log of the standard deviations along the local axes;
    * ``quats (N, 4)``: un-normalised rotation quaternions ``(w, x, y, z)``;
    * ``opacity_logits (N,)``: opacity before the sigmoid;
    * ``sh_dc (N, 1, 3)`` and ``sh_rest (N, K - 1, 3)``: SH colour coefficients.

    Additional per-Gaussian parameters (for instance motion coefficients) can be attached
    through ``extras``; they follow the Gaussians through densification and pruning.
    """

    def __init__(self, params: dict[str, Tensor], sh_degree: int) -> None:
        super().__init__()
        missing = [name for name in BASE_PARAMS if name not in params]
        if missing:
            raise ValueError(f"missing Gaussian parameters: {missing}")
        n = params["means"].shape[0]
        for name, value in params.items():
            if value.shape[0] != n:
                raise ValueError(f"parameter {name!r} has {value.shape[0]} rows, expected {n}")
        if params["sh_rest"].shape[1] != num_sh_coeffs(sh_degree) - 1:
            raise ValueError("sh_rest does not match sh_degree")
        self.params = nn.ParameterDict({k: nn.Parameter(v.clone()) for k, v in params.items()})
        self.sh_degree = sh_degree
        self.active_sh_degree = 0

    @classmethod
    def from_points(
        cls,
        points: Tensor,
        colors: Tensor,
        sh_degree: int = 0,
        opacity: float = 0.1,
        scales: Tensor | None = None,
        scale_factor: float = 1.0,
        extras: dict[str, Tensor] | None = None,
    ) -> GaussianCloud:
        """Initialise isotropic Gaussians on a coloured point cloud.

        Without explicit ``scales (N,)``, each Gaussian's size is the RMS distance to its
        three nearest neighbours, as in 3DGS.
        """
        n = points.shape[0]
        if scales is None:
            scales = knn_scale(points) if n > 1 else torch.full((n,), 0.01)
        scales = (scales * scale_factor).clamp_min(1e-6)
        quats = torch.zeros(n, 4, dtype=points.dtype)
        quats[:, 0] = 1.0
        params = {
            "means": points.clone(),
            "log_scales": torch.log(scales)[:, None].repeat(1, 3).to(points.dtype),
            "quats": quats,
            "opacity_logits": torch.full((n,), float(inverse_sigmoid(opacity)), dtype=points.dtype),
            "sh_dc": rgb_to_sh(colors.to(points.dtype))[:, None, :],
            "sh_rest": torch.zeros(n, num_sh_coeffs(sh_degree) - 1, 3, dtype=points.dtype),
        }
        params.update(extras or {})
        return cls(params, sh_degree)

    # ------------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return self.params["means"].shape[0]

    @property
    def means(self) -> Tensor:
        return self.params["means"]

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.params["log_scales"])

    @property
    def quats(self) -> Tensor:
        return normalize_quat(self.params["quats"])

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self.params["opacity_logits"])

    @property
    def sh(self) -> Tensor:
        return torch.cat([self.params["sh_dc"], self.params["sh_rest"]], dim=1)

    def colors(self, camera_center: Tensor, means: Tensor | None = None) -> Tensor:
        """View-dependent RGB ``(N, 3)`` seen from ``camera_center``.

        ``means`` overrides the positions used for the viewing directions (for Gaussians
        that have been moved by a motion model).
        """
        if self.active_sh_degree == 0:
            return sh_to_rgb(self.params["sh_dc"][:, 0]).clamp_min(0.0)
        positions = self.means if means is None else means
        dirs = torch.nn.functional.normalize(positions.detach() - camera_center, dim=-1)
        return eval_sh(self.active_sh_degree, self.sh, dirs)

    def raise_sh_degree(self) -> None:
        self.active_sh_degree = min(self.active_sh_degree + 1, self.sh_degree)

    def detached(self) -> dict[str, Tensor]:
        return {name: value.detach().clone() for name, value in self.params.items()}


class CloudOptimizer:
    """Adam over a :class:`GaussianCloud` whose number of Gaussians may change.

    Each parameter tensor gets its own learning rate. When Gaussians are pruned or
    appended, the tensors *and* Adam's first/second-moment estimates are edited in place
    so that the surviving Gaussians keep their optimisation state (new ones start at zero).
    """

    def __init__(self, cloud: GaussianCloud, lrs: dict[str, float], eps: float = 1e-15) -> None:
        unknown = set(cloud.params.keys()) - set(lrs)
        if unknown:
            raise ValueError(f"no learning rate given for parameters {sorted(unknown)}")
        self.cloud = cloud
        self.optimizer = torch.optim.Adam(
            [
                {"params": [cloud.params[name]], "lr": lrs[name], "name": name}
                for name in cloud.params
            ],
            eps=eps,
        )

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()

    def set_lr(self, name: str, lr: float) -> None:
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                group["lr"] = lr

    def _replace(self, edit) -> None:
        """Apply ``edit(tensor, is_state) -> tensor`` to every parameter and its Adam state."""
        for group in self.optimizer.param_groups:
            old = group["params"][0]
            state = self.optimizer.state.pop(old, None)
            new = nn.Parameter(edit(group["name"], old.detach(), False))
            if state is not None:
                state["exp_avg"] = edit(group["name"], state["exp_avg"], True)
                state["exp_avg_sq"] = edit(group["name"], state["exp_avg_sq"], True)
                self.optimizer.state[new] = state
            group["params"][0] = new
            self.cloud.params[group["name"]] = new

    def prune(self, keep: Tensor) -> None:
        """Keep only the Gaussians selected by the boolean mask ``keep (N,)``."""
        self._replace(lambda name, tensor, is_state: tensor[keep])

    def append(self, new: dict[str, Tensor]) -> None:
        """Append Gaussians; ``new`` maps every parameter name to its ``(M, ...)`` values."""
        missing = set(self.cloud.params.keys()) - set(new)
        if missing:
            raise ValueError(f"append() is missing parameters {sorted(missing)}")

        def edit(name: str, tensor: Tensor, is_state: bool) -> Tensor:
            extension = torch.zeros_like(new[name]) if is_state else new[name]
            return torch.cat([tensor, extension.to(tensor.dtype)], dim=0)

        self._replace(edit)

    def reset(self, name: str, value: Tensor) -> None:
        """Overwrite one parameter and clear its Adam state (used for opacity resets)."""

        def edit(param_name: str, tensor: Tensor, is_state: bool) -> Tensor:
            if param_name != name:
                return tensor
            return torch.zeros_like(tensor) if is_state else value.to(tensor.dtype)

        self._replace(edit)
