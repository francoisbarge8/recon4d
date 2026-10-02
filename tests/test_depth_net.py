"""The in-domain depth network: architecture, losses, training and oracle depth."""

import pytest
import torch

from recon4d.frontend.depth import DepthNoise, OracleDepth
from recon4d.frontend.depth.tiny_net import (
    LearnedDepth,
    TinyDepthConfig,
    TinyDepthNet,
    gradient_matching_loss,
    load_checkpoint,
    save_checkpoint,
    scale_invariant_loss,
)
from recon4d.frontend.depth.train import (
    DepthTrainConfig,
    augment,
    make_dataset,
    train_depth_network,
)
from recon4d.metrics import depth_metrics, depth_temporal_error

SMALL = TinyDepthConfig(widths=(8, 16, 24))


def test_network_handles_any_image_size():
    model = TinyDepthNet(SMALL)
    for height, width in ((48, 64), (37, 51), (144, 192)):
        out = model(torch.rand(2, 3, height, width))
        assert out.shape == (2, height, width)
    assert sum(p.numel() for p in TinyDepthNet().parameters()) < 3_000_000


def test_scale_invariant_loss():
    generator = torch.Generator().manual_seed(0)
    target = torch.rand(3, 16, 20, generator=generator) + 0.5
    pred = target + 0.1 * torch.randn(3, 16, 20, generator=generator)
    base = scale_invariant_loss(pred, target, variance_focus=1.0)
    # With a variance focus of 1, a global scale (an offset in log space) costs nothing.
    shifted = scale_invariant_loss(pred + 0.7, target, variance_focus=1.0)
    assert shifted.item() == pytest.approx(base.item(), abs=1e-6)
    # The default keeps a small incentive to get the scale right.
    assert scale_invariant_loss(pred + 0.7, target) > scale_invariant_loss(pred, target)
    assert scale_invariant_loss(target, target).item() == pytest.approx(0.0, abs=1e-7)


def test_gradient_matching_loss_ignores_offsets_and_penalises_blur():
    target = torch.zeros(1, 32, 32)
    target[:, :, 16:] = 1.0  # a depth edge
    assert gradient_matching_loss(target + 3.0, target).item() == pytest.approx(0.0, abs=1e-7)
    blurred = torch.nn.functional.avg_pool2d(target[None], 5, 1, 2, count_include_pad=False)[0]
    assert gradient_matching_loss(blurred, target) > 0.01


def test_checkpoint_round_trip(tmp_path):
    model = TinyDepthNet(SMALL).eval()
    image = torch.rand(1, 3, 24, 32)
    save_checkpoint(tmp_path / "net.pth", model, {"note": "test"})
    loaded, meta = load_checkpoint(tmp_path / "net.pth")
    assert meta == {"note": "test"} and loaded.cfg == SMALL
    # Weights are stored in half precision.
    assert (loaded(image) - model(image)).abs().max() < 2e-2
    with pytest.raises(FileNotFoundError, match="not found"):
        LearnedDepth(tmp_path / "missing.pth")


def test_augmentation_flips_images_and_depth_together():
    generator = torch.Generator().manual_seed(0)
    images = torch.rand(64, 3, 8, 10)
    depth = torch.arange(10.0).expand(64, 8, 10) + 1.0
    out_images, out_depth = augment(images, depth, generator)
    assert out_images.shape == images.shape and out_images.min() >= 0.0 and out_images.max() <= 1.0
    flipped = out_depth[:, 0, 0] > out_depth[:, 0, -1]
    assert 10 < flipped.sum() < 54, "about half of the batch is flipped"
    assert torch.equal(out_depth[flipped], depth[flipped].flip(-1))
    assert torch.equal(out_depth[~flipped], depth[~flipped])


def test_training_reduces_the_validation_error(tmp_path):
    cfg = DepthTrainConfig(
        n_scenes=6,
        frames_per_scene=2,
        n_val_scenes=2,
        width=48,
        height=36,
        spp=1,
        epochs=6,
        batch_size=4,
    )
    images, depth = make_dataset([1, 2], cfg)
    assert images.shape == (4, 36, 48, 3) and images.dtype == torch.uint8
    assert depth.shape == (4, 36, 48) and depth.min() > 0.3
    best = train_depth_network(cfg, tmp_path / "net.pth", cache_dir=tmp_path, model_cfg=SMALL)
    assert (tmp_path / "net.pth").exists() and (tmp_path / "depth_train.npz").exists()
    # Even this toy run learns the layout prior of the scenes (far walls, near floor).
    assert best["abs_rel"] < 0.35 and best["delta1"] > 0.4
    estimator = LearnedDepth(tmp_path / "net.pth")
    prediction = estimator.predict(images[:2].float() / 255.0)
    assert prediction.shape == (2, 36, 48) and (prediction > 0).all()
    assert estimator.kind == "scale"


def test_oracle_depth_noise_model(tiny_rolling):
    seq = tiny_rolling
    gt = seq.gt
    assert torch.equal(OracleDepth(gt.depth).predict(seq.images), gt.depth)
    noisy = OracleDepth(gt.depth, DepthNoise(seed=1)).predict(seq.images)
    again = OracleDepth(gt.depth, DepthNoise(seed=1)).predict(seq.images)
    assert torch.equal(noisy, again) and (noisy > 0).all()
    # Per-frame scales differ a lot, the relief inside a frame only a little.
    ratio = (noisy / gt.depth).flatten(1).median(dim=1).values
    assert ratio.max() / ratio.min() > 1.15
    aligned = noisy / ratio[:, None, None]
    assert 0.01 < depth_metrics(aligned, gt.depth)["abs_rel"] < 0.1
    # Such a prediction flickers: its temporal error is far above the ground truth's.
    oracle = gt.oracle
    pairs = [oracle.scene_flow(t, t + 1) for t in range(seq.num_frames - 1)]
    flows, valid, scene_flow = (torch.stack([p[i] for p in pairs]) for i in range(3))
    K = seq.K()
    flicker = depth_temporal_error(noisy, K, gt.c2w, flows, valid, scene_flow)
    steady = depth_temporal_error(gt.depth, K, gt.c2w, flows, valid, scene_flow)
    assert flicker > 20 * steady
    with pytest.raises(ValueError, match="does not match"):
        OracleDepth(gt.depth[:2]).predict(seq.images)
