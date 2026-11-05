"""Tiny synthetic CPU tests, not a rerun of the reported EEG benchmark."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "encoder"))

from cevr.model import CEVRModel, MultiLevelVisualProjector
from cevr.train import representation_loss, stage_level_alignment_loss
from relic_sc.full_score import _load_branch, fuse_branches, load_layer_banks, ranks, score_branch

training = importlib.import_module("cevr.train")


@pytest.fixture(autouse=True, scope="module")
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _synthetic_prepared_data(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(2026)
    eeg_root = tmp_path / "eeg"
    subject = eeg_root / "sub-01"
    subject.mkdir(parents=True)
    (eeg_root / "info.json").write_text(json.dumps({"ch_names": ["Cz", "Oz"]}), encoding="utf-8")
    # The prepared interchange format retains singleton image/repeat axes.
    np.save(subject / "train.npy", rng.normal(size=(8, 1, 1, 2, 6)).astype(np.float32))
    np.save(subject / "test.npy", rng.normal(size=(4, 1, 1, 2, 6)).astype(np.float32))
    bank_root = tmp_path / "visual"
    for layer in range(100, 103):
        directory = bank_root / f"level-{layer}"
        directory.mkdir(parents=True)
        np.save(directory / "image_train.npy", rng.normal(size=(8, 1, 4)).astype(np.float32))
        np.save(directory / "image_test.npy", rng.normal(size=(4, 1, 4)).astype(np.float32))
    return eeg_root, bank_root


def _tiny_config(branch: str, tmp_path: Path, eeg_root: Path, bank_root: Path) -> dict:
    config = json.loads((ROOT / "encoder" / "configs" / f"alljoined_{branch}.json").read_text(encoding="utf-8"))
    config["run"].update(cpu=True, seed=31, subject="sub-01", output_dir=str(tmp_path / "runs"))
    config["data"].update(
        eeg_data_dir=str(eeg_root), visual_feature_root=str(bank_root),
        visual_levels=["100", "101", "102"], selected_channels=["Cz", "Oz"], cache_averaged_eeg=False,
    )
    config["model"].update(
        window_ends=[2, 4, 6], additive_hidden_dim=8, embed_dim=4, num_heads=2,
        current_target_bottleneck_dim=2, stage_calibration_init_indices=[0, 1, 2],
    )
    config["train"].update(epochs=1, batch_size=4, lr=1e-3, eval_every=1, log_every=1,
                            hier_level_weights=[1, 2, 3])
    return config


def test_tiny_gh_training_checkpoint_export_and_full_score(tmp_path, monkeypatch) -> None:
    eeg_root, bank_root = _synthetic_prepared_data(tmp_path)
    train_banks, test_banks, names = load_layer_banks(bank_root)
    assert names == ["level-100", "level-101", "level-102"]
    branch_scores = {}
    captured = {}

    def capture_eeg(**kwargs):
        model = CEVRModel(**kwargs)
        captured["model"] = model
        captured["model_initial"] = deepcopy(model.state_dict())
        return model

    def capture_visual(**kwargs):
        model = MultiLevelVisualProjector(**kwargs)
        captured["visual_projector"] = model
        captured["visual_initial"] = deepcopy(model.state_dict())
        return model

    monkeypatch.setattr(training, "CEVRModel", capture_eeg)
    monkeypatch.setattr(training, "MultiLevelVisualProjector", capture_visual)
    for branch in ("G", "H"):
        config = _tiny_config(branch, tmp_path, eeg_root, bank_root)
        result = training.train(config)
        assert result["epochs"] == 1 and result["device"] == "cpu"
        run = config["run"]
        output = Path(run["output_dir"]) / run["subject"] / run["run_name"] / f"seed-{run['seed']}"
        checkpoint = torch.load(output / "final_checkpoint.pt", map_location="cpu", weights_only=False)
        encoder = captured["model"]
        visual = captured["visual_projector"]
        # Check actual training, not just that constructors or exports ran.
        for name in ("encoder.temporal_weight", "fusion.weight"):
            param = dict(encoder.named_parameters())[name]
            assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
            assert not torch.equal(checkpoint["model"][name], captured["model_initial"][name])
        global_name = "current_fusion_projector.0.weight" if branch == "G" else "current_fusion_projector.weight"
        names_to_check = [global_name]
        if branch == "H":
            names_to_check.extend(["stage_calibration_logits", "stage_calibration_projector.weight"])
        for name in names_to_check:
            param = dict(visual.named_parameters())[name]
            assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
            assert not torch.equal(checkpoint["visual_projector"][name], captured["visual_initial"][name])
        if branch == "G":
            assert visual.stage_calibration_projector.weight.grad is None
        for key, value in visual.state_dict().items():
            torch.testing.assert_close(value.cpu(), checkpoint["visual_projector"][key], rtol=0, atol=0)

        with np.load(output / "train_test_embeddings.npz") as archive:
            data = {key: archive[key] for key in archive.files}
        expected = {f"{split}_{key}" for split in ("train", "test")
                    for key in ("pred", "image", "object_idx", "image_idx")}
        assert set(data) == expected
        for split, count in (("train", 8), ("test", 4)):
            assert data[f"{split}_pred"].shape == data[f"{split}_image"].shape == (count, 4)
            assert np.isfinite(data[f"{split}_pred"]).all()
            assert np.isfinite(data[f"{split}_image"]).all()
            np.testing.assert_array_equal(data[f"{split}_object_idx"], np.arange(count))
            np.testing.assert_array_equal(data[f"{split}_image_idx"], np.zeros(count, dtype=np.int64))
        # Saved global targets are raw model outputs, not accidentally re-normalized.
        bank = torch.from_numpy(np.stack([np.asarray(b[:, 0]) for b in test_banks], axis=1))
        with torch.no_grad():
            raw_visual, _ = visual(bank)
        np.testing.assert_allclose(data["test_image"], raw_visual.numpy(), rtol=1e-6, atol=1e-7)
        assert not np.allclose(np.linalg.norm(data["test_image"], axis=1), 1.0)
        branch_scores[branch] = score_branch(*_load_branch(data, train_banks, test_banks))
        assert np.isfinite(branch_scores[branch].final_score).all()
        with pytest.raises(FileExistsError, match="Completed run already exists"):
            training.train(config)

    fused = fuse_branches(branch_scores)
    assert fused.shape == (4, 4) and np.isfinite(fused).all()
    assert np.all((fused >= 0) & (fused <= 1 + 1e-6))
    assert np.all((ranks(fused) >= 1) & (ranks(fused) <= 4))


def _independent_symmetric_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    scores = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).T / temperature
    return -0.5 * (scores.log_softmax(dim=1).diagonal().mean() + scores.log_softmax(dim=0).diagonal().mean())


def test_h_loss_matches_weighted_stage_equation_and_backpropagates() -> None:
    torch.manual_seed(19)
    outputs = [torch.randn(4, 4, requires_grad=True) for _ in range(3)]
    targets = torch.randn(4, 3, 4, requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0])
    actual = stage_level_alignment_loss(outputs, targets, weights, 0.07)
    expected = sum((idx + 1) * _independent_symmetric_loss(pred, targets[:, idx], 0.07)
                   for idx, pred in enumerate(outputs)) / 6.0
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(pred.grad is not None and pred.grad.abs().sum() > 0 for pred in outputs)
    assert targets.grad is not None and targets.grad.abs().sum() > 0


def test_legacy_global_only_loss_and_h_total_objective() -> None:
    torch.manual_seed(41)
    bank = torch.randn(4, 3, 4)
    eeg_output = {"current": torch.randn(4, 4), "window_outputs": [torch.randn(4, 4) for _ in range(3)]}
    weights = torch.tensor([1.0, 2.0, 3.0])
    legacy_config = json.loads((ROOT / "encoder" / "configs" / "main.json").read_text(encoding="utf-8"))
    assert legacy_config["train"].get("hier_loss_weight", 0.0) == 0.0
    assert legacy_config["model"].get("stage_calibration_mode", "none") == "none"
    legacy = MultiLevelVisualProjector(image_dim=4, embed_dim=4, num_layers=3)
    total, global_term, stage = representation_loss(eeg_output, bank, legacy, 0.07, 0.0, weights)
    target, _ = legacy(bank)
    expected = _independent_symmetric_loss(eeg_output["current"], target, 0.07)
    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(global_term, expected)
    assert stage.item() == 0.0

    hierarchical = MultiLevelVisualProjector(image_dim=4, embed_dim=4, num_layers=3,
                                            stage_calibration_mode="stage_scalar")
    total, global_term, stage = representation_loss(eeg_output, bank, hierarchical, 0.07, 0.2, weights)
    torch.testing.assert_close(total, global_term + 0.2 * stage)
    assert stage.item() > 0
    with pytest.raises(ValueError, match="Positive hier_loss_weight"):
        representation_loss(eeg_output, bank, legacy, 0.07, 0.2, weights)


@pytest.mark.parametrize("weights", [[0, 0, 0], [-1, 2, 3], [1, float("nan"), 3], [1, 2]])
def test_invalid_stage_weights_rejected(weights) -> None:
    with pytest.raises(ValueError):
        stage_level_alignment_loss([torch.ones(2, 4) for _ in range(3)], torch.ones(2, 3, 4),
                                   torch.tensor(weights, dtype=torch.float32), 0.07)


@pytest.mark.parametrize(("stage_mode", "weight"), [("none", 0.2), ("stage_scalar", 0.0), ("none", -0.2)])
def test_inconsistent_branch_configuration_rejected_before_loading_data(stage_mode, weight) -> None:
    with pytest.raises(ValueError):
        training.train({"run": {}, "data": {}, "model": {"stage_calibration_mode": stage_mode},
                        "train": {"hier_loss_weight": weight}})
