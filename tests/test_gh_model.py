"""Small CPU checks of the archived G/H visual-target computations.

The independent equations below mirror VisualEvidenceProjector's mean
fusion, bottleneck adapter, and stage_scalar path. No private archive or
dataset is required to run these tests.
"""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "encoder"))

from cevr.model import MultiLevelVisualProjector


def test_default_visual_projector_keeps_linear_path_and_state_keys() -> None:
    torch.manual_seed(17)
    model = MultiLevelVisualProjector(image_dim=6, embed_dim=4, num_layers=3)
    bank = torch.randn(2, 3, 6)
    global_target, layers = model(bank)
    expected = F.linear(layers.mean(1), model.current_fusion_projector.weight, model.current_fusion_projector.bias)
    torch.testing.assert_close(global_target, expected)
    assert isinstance(model.current_fusion_projector, nn.Linear)
    assert model.calibrate_stage_targets(bank) == (None, None)
    expected_keys = {f"level_projectors.{i}.{field}" for i in range(3) for field in ("weight", "bias")}
    expected_keys.update({"current_fusion_projector.weight", "current_fusion_projector.bias",
                          "stage_calibration_projector.weight", "stage_calibration_projector.bias",
                          "relation_level_logits"})
    assert set(model.state_dict()) == expected_keys


def test_g_bottleneck_matches_original_mean_adapter_computation() -> None:
    torch.manual_seed(23)
    model = MultiLevelVisualProjector(
        image_dim=6, embed_dim=4, num_layers=3, current_target_adapter="bottleneck",
        current_target_bottleneck_dim=2, current_target_dropout=0.1,
    ).eval()
    bank = torch.randn(3, 3, 6)
    actual, layers = model(bank)
    projected = torch.stack([F.linear(bank[:, i], p.weight, p.bias)
                             for i, p in enumerate(model.level_projectors)], dim=1)
    first, _, dropout, last = model.current_fusion_projector
    expected = F.linear(F.gelu(F.linear(projected.mean(1), first.weight, first.bias)), last.weight, last.bias)
    torch.testing.assert_close(layers, projected)
    torch.testing.assert_close(actual, expected)
    assert dropout.p == 0.1
    actual.square().mean().backward()
    assert first.weight.grad is not None and first.weight.grad.abs().sum() > 0
    assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0


def test_h_stage_targets_mix_raw_features_and_match_original_scalar_path() -> None:
    torch.manual_seed(29)
    model = MultiLevelVisualProjector(
        image_dim=6, embed_dim=4, num_layers=3, num_eeg_levels=3,
        parameter_matched_heads=False, stage_calibration_mode="stage_scalar",
        stage_calibration_init_indices=[0, 1, 2], stage_calibration_init_strength=2.0,
        stage_calibration_temperature=1.0,
    )
    bank = torch.randn(2, 3, 6)
    actual, weights = model.calibrate_stage_targets(bank)
    logits = torch.eye(3) * 2.0
    expected_weights = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
    # Explicit stage-wise sums are independent of the implementation's einsum.
    mixed = torch.stack([(bank * row[None, :, None]).sum(1) for row in expected_weights], dim=1)
    projector = model.stage_calibration_projector
    expected = F.linear(mixed, projector.weight, projector.bias)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 3, 4)
    actual.square().mean().backward()
    assert model.stage_calibration_logits.grad is not None
    assert model.stage_calibration_logits.grad.abs().sum() > 0
    assert projector.weight.grad is not None and projector.weight.grad.abs().sum() > 0
    assert all(p.weight.grad is None for p in model.level_projectors)


@pytest.mark.parametrize("kwargs", [
    {"current_target_adapter": "unknown"},
    {"stage_calibration_mode": "unknown"},
    {"stage_calibration_temperature": 0.0},
    {"current_target_bottleneck_dim": 0},
    {"stage_calibration_mode": "stage_scalar", "stage_calibration_init_indices": [4]},
])
def test_invalid_gh_options_fail_loudly(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MultiLevelVisualProjector(image_dim=6, embed_dim=4, num_layers=3, **kwargs)
