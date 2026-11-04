import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relic_sc.full_score import (  # noqa: E402
    BranchScore,
    fuse_branches,
    inverse_mse_layer_weights,
    row_minmax,
    score_branch,
)


def _branch_for_fusion(global_scores, layer_scores) -> BranchScore:
    global_scores = np.asarray(global_scores, dtype=np.float32)
    layer_scores = [np.asarray(score, dtype=np.float32) for score in layer_scores]
    return BranchScore(
        global_score=global_scores,
        layer_scores=layer_scores,
        global_normalized=global_scores,
        layer_normalized=layer_scores,
        layer_weights=np.full(len(layer_scores), 1.0 / len(layer_scores), dtype=np.float32),
        final_score=np.zeros_like(global_scores),
        global_prediction=np.empty((0, 0), dtype=np.float32),
        layer_predictions=[],
    )


@pytest.mark.parametrize(
    ("gamma", "expected"),
    [
        (0.0, [[1.0, 0.0, 1.0]]),
        (0.25, [[13.0 / 14.0, 0.0, 1.0]]),
        (0.5, [[6.0 / 7.0, 0.0, 1.0]]),
        (1.0, [[5.0 / 7.0, 0.0, 1.0]]),
    ],
)
def test_fusion_equalizes_branch_layer_ranges_before_averaging(gamma, expected) -> None:
    # Already-normalized layers can aggregate to different branch ranges:
    # G averages to [0.4, 0.5, 0.6], while H averages to [1.0, 0.0, 0.2].
    branches = {
        "G": _branch_for_fusion([[1.0, 0.0, 1.0]], [[[0.0, 1.0, 0.2]], [[0.8, 0.0, 1.0]]]),
        "H": _branch_for_fusion([[1.0, 0.0, 1.0]], [[[1.0, 0.0, 0.2]], [[1.0, 0.0, 0.2]]]),
    }

    fused = fuse_branches(branches, gamma=gamma)

    np.testing.assert_allclose(fused, expected, rtol=1e-6, atol=1e-7)
    assert fused.dtype == np.float32
    if gamma > 0.0:
        assert fused.argmax(axis=1).tolist() == [2]


def test_fusion_handles_constant_branch_layer_aggregates() -> None:
    branches = {
        "G": _branch_for_fusion(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]],
        ),
        "H": _branch_for_fusion(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[[0.0, 1.0, 0.5], [0.0, 0.5, 1.0]], [[0.0, 1.0, 0.5], [1.0, 0.5, 0.0]]],
        ),
    }

    fused = fuse_branches(branches)

    assert np.isfinite(fused).all()
    np.testing.assert_allclose(fused, [[0.0, 0.5, 0.25], [0.0, 0.0, 0.0]], atol=1e-7)


def test_full_score_shapes_and_fusion() -> None:
    rng = np.random.default_rng(11)
    n_test, n_train, d, d_image = 10, 20, 6, 8
    train_query = rng.normal(size=(n_train, d)).astype(np.float32)
    test_query = rng.normal(size=(n_test, d)).astype(np.float32)
    train_global = rng.normal(size=(n_train, d_image)).astype(np.float32)
    test_global = rng.normal(size=(n_test, d_image)).astype(np.float32)
    train_layers = [rng.normal(size=(n_train, d_image)).astype(np.float32) for _ in range(3)]
    test_layers = [rng.normal(size=(n_test, d_image)).astype(np.float32) for _ in range(3)]

    result = score_branch(
        train_query,
        test_query,
        train_global,
        test_global,
        train_layers,
        test_layers,
    )
    assert result.global_score.shape == (n_test, n_test)
    assert len(result.layer_scores) == 3
    assert result.layer_weights.shape == (3,)
    np.testing.assert_allclose(result.layer_weights.sum(), 1.0, atol=1e-6)

    fused = fuse_branches({"G": result, "H": result})
    assert fused.shape == (n_test, n_test)
    assert np.all((fused >= 0.0) & (fused <= 1.0 + 1e-6))


def test_minmax_and_inverse_mse_regularization() -> None:
    scores = np.array([[2.0, 4.0, 6.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    out = row_minmax(scores)
    np.testing.assert_allclose(out[0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(out[1], [0.0, 0.0, 0.0])

    target = np.zeros((4, 2), dtype=np.float32)
    good = np.zeros_like(target)
    bad = np.ones_like(target)
    weights = inverse_mse_layer_weights(target, [good, bad], kappa=0.25)
    assert weights[0] > weights[1]
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)
