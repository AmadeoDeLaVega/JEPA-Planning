"""Photo-level metrics, head fitting and the training script on synthetic features."""

import json
import sys

import numpy as np
import pytest
import torch

from jepa_planning import linear_head as lh


def synthetic(targets=12, agents=4, patches=3, dim=6, seed=0):
    """Coordinate 0 carries the target; coordinates 1.. carry large agent noise."""
    gen = torch.Generator().manual_seed(seed)
    target = torch.randn(targets, 1, patches, generator=gen) * 2
    features = torch.randn(targets * agents, patches, dim, generator=gen) * 3
    groups = np.repeat(np.arange(targets), agents)
    features[:, :, 0] = target[groups, 0] + 0.05 * torch.randn(targets * agents, patches, generator=gen)
    return features, groups


def test_pairwise_matches_head_distance():
    features, _ = synthetic()
    head = lh.random_head(2, 6, 0)
    pairwise = lh.pairwise_head_distance(features[:5], features[5:9], head)
    expected = lh.head_distance(features[:5, None], features[None, 5:9], head)
    torch.testing.assert_close(pairwise, expected, rtol=1e-4, atol=1e-4)


def test_auroc_matches_brute_force():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 30).round(1), rng.normal(0, 1, 40).round(1)
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert lh.auroc(pos, neg) == pytest.approx(brute)


def test_retrieval_separates_signal_head_from_noise_head():
    features, groups = synthetic()
    signal = torch.zeros(1, 6)
    signal[0, 0] = 1
    noise = torch.zeros(1, 6)
    noise[0, 3] = 1
    good, bad = lh.retrieval_metrics(features, groups, signal), lh.retrieval_metrics(features, groups, noise)
    assert good["top1"] == 1.0 and good["auroc"] > 0.99
    assert bad["auroc"] < 0.7


def test_sensitivity_ratio_detects_collapse():
    features, groups = synthetic()
    moved = features.clone()
    moved[:, :, 0] += 1.0  # move the target only
    signal = torch.zeros(1, 6)
    signal[0, 0] = 1
    collapsed = torch.zeros(1, 6)
    collapsed[0, 4] = 1
    for head, expect_large in [(signal, True), (collapsed, False)]:
        within = lh.retrieval_metrics(features, groups, head)["within_median"]
        ratios = lh.sensitivity_ratios(features, moved, head, within)
        assert (np.median(ratios) > 10) == expect_large


def test_fit_head_learns_target_coordinate():
    features, groups = synthetic(targets=16)
    val_features, val_groups = synthetic(targets=8, seed=1)
    sampler = lh.AllocationSampler(groups, 16, 4, seed=0)
    head, curve = lh.fit_head(features, sampler, steps=150, batch_size=32, tau=0.5, lr=0.05, seed=0, rank=2,
                              validation=(val_features, val_groups), eval_every=50)
    assert [row["step"] for row in curve] == [50, 100, 150]
    torch.testing.assert_close(torch.linalg.matrix_norm(head), torch.tensor(1.0))
    assert head.square().sum(0)[0] > 0.9
    assert curve[-1]["val_auroc"] > 0.95 and curve[-1]["skipped"] == 0


def write_pool(directory, targets, seed, sensitivity=False):
    directory.mkdir()
    features, groups = synthetic(targets=targets, dim=40, seed=seed)
    np.savez(directory / "images.npz", group=groups)
    payload = {"groups": features.half()}
    if sensitivity:
        moved = features[::4].clone()
        moved[:, :, 0] += 1
        payload["sensitivity"] = moved.half()
        np.savez(directory / "sensitivity.npz", group=groups[::4], kind=np.array(["shift_px"] * len(moved)),
                 amount=np.full(len(moved), 10.0))
    torch.save(payload, directory / "features.pt")


def test_train_script_smoke(tmp_path, monkeypatch):
    from experiments import train_linear_head

    write_pool(tmp_path / "train", 10, 0)
    write_pool(tmp_path / "val", 4, 1, sensitivity=True)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["train", "--train", str(tmp_path / "train"), "--val", str(tmp_path / "val"),
                                      "--output", str(out), "--device", "cpu", "--smoke"])
    train_linear_head.main()
    heads = sorted(p.name for p in (out / "heads").iterdir())
    assert heads == ["rank32_B16_M8_K2_r0.pt", "rank32_B16_M8_K2_r0_shuffled.pt",
                     "rank4_B16_M8_K2_r0.pt", "rank4_B16_M8_K2_r0_shuffled.pt"]
    for rank in (4, 32):
        head, config = lh.load_head(out / "heads" / f"rank{rank}_B16_M8_K2_r0.pt")
        assert head.shape == (rank, 40) and config["M"] == 8
        assert config["rank"] == rank and config["compression"] == 40 / rank
    import csv
    rows = list(csv.DictReader((out / "validation.csv").open()))
    methods = {(row["method"], row["rank"]) for row in rows}
    for rank in ("4", "32"):
        for method in ("linear_head", "linear_head_shuffled", "random_projection", "coordinate_subset"):
            assert (method, rank) in methods
    assert ("plain_mse", "40") in methods
    assert all(row["sensitivity_shift_px_10"] for row in rows)
    chosen = json.loads((out / "metadata.json").read_text())["chosen"]
    assert set(chosen) == {"4", "32"} and chosen["4"]["tau"] == 0.1
    with pytest.raises(FileExistsError):
        train_linear_head.main()


def test_train_script_rejects_rank_above_feature_dim(tmp_path, monkeypatch):
    from experiments import train_linear_head

    write_pool(tmp_path / "train", 10, 0)
    write_pool(tmp_path / "val", 4, 1)
    monkeypatch.setattr(sys, "argv", ["train", "--train", str(tmp_path / "train"), "--val", str(tmp_path / "val"),
                                      "--output", str(tmp_path / "out"), "--device", "cpu", "--smoke",
                                      "--ranks", "41"])
    with pytest.raises(ValueError, match="ranks"):
        train_linear_head.main()
