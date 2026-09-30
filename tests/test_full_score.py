import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relic_sc.full_score import (  # noqa: E402
    fuse_branches,
    inverse_mse_layer_weights,
    row_minmax,
    score_branch,
)


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
