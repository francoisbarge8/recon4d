"""Cross-check the pure-PyTorch rasterizer against gsplat's CUDA rasterizer.

Both implement the same equations, so the two renders of the same Gaussians must agree up
to the handling of the Gaussian tails (gsplat bounds every Gaussian by 3 standard
deviations, this project by the exact support of its ``alpha >= 1/255`` contribution).

Needs a CUDA GPU and ``pip install gsplat``::

    python scripts/compare_gsplat.py --out results/gsplat_check.json
"""

from __future__ import annotations

import argparse
import time

import torch

from recon4d.gaussians import RasterSettings, project_gaussians, rasterize
from recon4d.geometry import Intrinsics, invert_se3, look_at
from recon4d.metrics import psnr
from recon4d.utils import save_json


def random_gaussians(n: int, device: str, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    means = (torch.rand(n, 3, generator=generator) - 0.5) * torch.tensor([4.0, 3.0, 3.0])
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=generator), dim=-1)
    scales = torch.exp(torch.rand(n, 3, generator=generator) * 2.0 - 4.5)
    opacities = torch.rand(n, generator=generator) * 0.85 + 0.1
    colors = torch.rand(n, 3, generator=generator)
    return tuple(t.to(device) for t in (means, quats, scales, opacities, colors))


def timed(fn, repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/gsplat_check.json")
    parser.add_argument("--gaussians", type=int, default=30000)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=288)
    args = parser.parse_args()

    from gsplat import rasterization

    device = "cuda"
    width, height = args.width, args.height
    K = Intrinsics.from_fov(width, height, 60.0).matrix().to(device)
    w2c = invert_se3(look_at(torch.tensor([0.4, 0.3, -4.0]), torch.zeros(3))).to(device)
    means, quats, scales, opacities, colors = random_gaussians(args.gaussians, device)
    report = {"gaussians": args.gaussians, "width": width, "height": height}

    def ours(max_sigma: float = 3.33, tile: int = 8):
        settings = RasterSettings(tile_size=tile, max_sigma=max_sigma, chunk_elements=60_000_000)
        projection = project_gaussians(
            means, quats, scales, K, w2c, width, height, settings, opacities
        )
        return rasterize(projection, opacities, colors, width, height, settings)

    def theirs():
        image, alpha, _ = rasterization(
            means, quats, scales, opacities, colors, w2c[None], K[None], width, height, packed=False
        )
        return image[0], alpha[0, ..., 0]

    with torch.no_grad():
        reference_image, reference_alpha = theirs()
        for label, max_sigma in (("exact_support", 3.33), ("three_sigma", 3.0)):
            out = ours(max_sigma)
            report[label] = {
                "psnr": float(psnr(out.image, reference_image)),
                "max_abs_diff": float((out.image - reference_image).abs().max()),
                "mean_abs_diff": float((out.image - reference_image).abs().mean()),
                "alpha_max_abs_diff": float((out.alpha - reference_alpha).abs().max()),
            }
        report["forward_ms"] = {
            "recon4d_tile8": timed(lambda: ours(tile=8), 10),
            "recon4d_tile16": timed(lambda: ours(tile=16), 10),
            "gsplat": timed(theirs, 10),
        }

    # Gradients of a simple loss w.r.t. the means, through both rasterizers.
    target = torch.rand(
        height, width, 3, device=device, generator=torch.Generator(device).manual_seed(1)
    )
    grads = []
    for render in (lambda: ours().image, lambda: theirs()[0]):
        means.requires_grad_(True)
        loss = ((render() - target) ** 2).mean()
        (grad,) = torch.autograd.grad(loss, means)
        means.requires_grad_(False)
        grads.append(grad)
    cosine = torch.nn.functional.cosine_similarity(grads[0].flatten(), grads[1].flatten(), dim=0)
    report["gradient"] = {
        "cosine_similarity": float(cosine),
        "relative_difference": float((grads[0] - grads[1]).norm() / grads[1].norm()),
    }
    save_json(args.out, report)
    for key, value in report.items():
        print(key, value)


if __name__ == "__main__":
    main()
