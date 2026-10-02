"""Image quality metrics: PSNR, SSIM and LPIPS.

All functions take images as ``(..., H, W, 3)`` tensors in ``[0, 1]`` and accept an optional
boolean ``mask (..., H, W)`` restricting the evaluation to some pixels (for instance the
co-visible region of a novel view, or the moving objects only).

``ssim`` is differentiable and doubles as the D-SSIM training loss.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import Tensor

_SSIM_C1 = 0.01**2
_SSIM_C2 = 0.03**2


def psnr(
    pred: Tensor, target: Tensor, mask: Tensor | None = None, max_value: float = 1.0
) -> Tensor:
    """Peak signal-to-noise ratio in dB, averaged over the (masked) pixels."""
    squared = (pred - target) ** 2
    if mask is None:
        mse = squared.mean()
    else:
        mse = (squared * mask[..., None]).sum() / (mask.sum() * squared.shape[-1]).clamp_min(1)
    return 10.0 * torch.log10(max_value**2 / mse.clamp_min(1e-12))


@lru_cache(maxsize=8)
def _gaussian_window(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, dtype=torch.float64) - (size - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def _ssim_index(mu_x, mu_y, xx, yy, xy) -> Tensor:
    sigma_x = xx - mu_x * mu_x
    sigma_y = yy - mu_y * mu_y
    sigma_xy = xy - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + _SSIM_C1) * (2.0 * sigma_xy + _SSIM_C2)
    denominator = (mu_x * mu_x + mu_y * mu_y + _SSIM_C1) * (sigma_x + sigma_y + _SSIM_C2)
    return numerator / denominator


def ssim_map(pred: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    """Per-pixel SSIM index (Wang et al., 2004) with a Gaussian window.

    Args:
        pred, target: ``(B, C, H, W)`` images in ``[0, 1]``.

    Returns:
        ``(B, C, H - window_size + 1, W - window_size + 1)``: the index is only defined
        where the window fits entirely in the image ("valid" convolution), as in the
        reference implementation.
    """
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma).to(pred.dtype).to(pred.device)
    window = window.expand(channels, 1, window_size, window_size)

    def blur(x: Tensor) -> Tensor:
        return F.conv2d(x, window, groups=channels)

    return _ssim_index(
        blur(pred), blur(target), blur(pred * pred), blur(target * target), blur(pred * target)
    )


def masked_ssim_map(
    pred: Tensor, target: Tensor, mask: Tensor, window_size: int = 11, sigma: float = 1.5
) -> Tensor:
    """SSIM index whose local statistics only use the pixels inside ``mask``.

    The Gaussian window is renormalised by the mask weight it covers (a normalised
    convolution, as in the DyCheck evaluation protocol), so pixels outside the mask never
    leak into the score and regions thinner than the window remain measurable.

    Args:
        pred, target: ``(B, C, H, W)``.
        mask: ``(B, 1, H, W)`` with values in ``{0, 1}``.

    Returns:
        ``(B, C, H, W)`` index map; it is only meaningful where ``mask`` is 1.
    """
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma).to(pred.dtype).to(pred.device)
    pad = window_size // 2
    weight = F.conv2d(mask, window[None, None], padding=pad).clamp_min(1e-8)
    window = window.expand(channels, 1, window_size, window_size)

    def blur(x: Tensor) -> Tensor:
        return F.conv2d(x * mask, window, groups=channels, padding=pad) / weight

    return _ssim_index(
        blur(pred), blur(target), blur(pred * pred), blur(target * target), blur(pred * target)
    )


def ssim(
    pred: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Mean structural similarity of ``(..., H, W, 3)`` images.

    Without a mask this is the standard SSIM (Gaussian window, valid region). With a
    ``mask (..., H, W)`` the statistics are restricted to the masked pixels, see
    :func:`masked_ssim_map`; an empty mask yields NaN.
    """
    x = pred.reshape(-1, *pred.shape[-3:]).permute(0, 3, 1, 2)
    y = target.reshape(-1, *target.shape[-3:]).permute(0, 3, 1, 2)
    if mask is None:
        return ssim_map(x, y, window_size, sigma).mean()
    m = mask.reshape(-1, 1, *mask.shape[-2:]).to(x.dtype)
    if m.sum() == 0:
        warnings.warn("SSIM mask is empty; returning NaN", stacklevel=2)
        return x.new_full((), float("nan"))
    index = masked_ssim_map(x, y, m, window_size, sigma)
    return (index * m).sum() / (m.sum() * index.shape[1])


class LPIPS:
    """Learned perceptual image patch similarity (Zhang et al., CVPR 2018).

    Thin wrapper around the reference ``lpips`` package, which is an optional dependency
    (``pip install recon4d[lpips]``). The backbone weights are downloaded on first use.

    Args:
        net: ``"alex"`` (the variant usually reported in the view synthesis literature),
            ``"vgg"`` or ``"squeeze"``.
    """

    def __init__(self, net: str = "alex", device: str | torch.device = "cpu") -> None:
        try:
            import lpips
        except ImportError as error:
            raise ImportError(
                "LPIPS needs the optional 'lpips' package: pip install recon4d[lpips]"
            ) from error
        self.net = net
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def is_available() -> bool:
        try:
            import lpips  # noqa: F401
        except ImportError:
            return False
        return True

    @torch.no_grad()
    def __call__(self, pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        """Mean LPIPS distance of ``(..., H, W, 3)`` images in ``[0, 1]``.

        With a ``mask``, pixels outside it are replaced by the target in the prediction
        (so they contribute zero distance), following the DyCheck masked-LPIPS protocol.
        """
        if mask is not None:
            pred = torch.where(mask[..., None], pred, target)
        x = pred.reshape(-1, *pred.shape[-3:]).permute(0, 3, 1, 2) * 2.0 - 1.0
        y = target.reshape(-1, *target.shape[-3:]).permute(0, 3, 1, 2) * 2.0 - 1.0
        device = next(self.model.parameters()).device
        return self.model(x.to(device).float(), y.to(device).float()).mean().cpu()
