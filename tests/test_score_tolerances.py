import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relic_sc import full_score as scorer  # noqa: E402


def test_vector_normalization_zero_tiny_and_regular_rows() -> None:
    vectors = np.array([[0.0, 0.0], [3e-9, 4e-9], [3e-8, 4e-8], [3.0, 4.0]], dtype=np.float32)

    normalized = scorer.l2_normalize(vectors)

    np.testing.assert_allclose(normalized, [[0.0, 0.0], [0.3, 0.4], [0.6, 0.8], [0.6, 0.8]], atol=1e-7)
    np.testing.assert_allclose(scorer.l2_normalize(vectors[2:3], eps=1e-6), [[0.03, 0.04]], atol=1e-7)
    assert np.isfinite(normalized).all()
    assert normalized.dtype == np.float32


def test_calibration_normalizes_tiny_reference_independently_of_variance_epsilon() -> None:
    query = np.zeros((3, 2), dtype=np.float32)
    reference = np.tile(np.array([[3e-8, 4e-8]], dtype=np.float32), (3, 1))

    np.testing.assert_allclose(scorer.loo_variance_calibrate(query, reference), [[0.6, 0.8]] * 3, atol=1e-7)
    np.testing.assert_allclose(
        scorer.loo_variance_calibrate(query, reference, eps=0.01), [[0.6, 0.8]] * 3, atol=1e-7
    )
    np.testing.assert_allclose(
        scorer.calibrate(query, reference, norm_eps=1e-6), [[0.03, 0.04]] * 3, atol=1e-7
    )


def test_calibration_keeps_sample_variance_stabilizer() -> None:
    query = np.array([[0.001, 0.0], [0.0, 0.001], [-0.001, 0.0]], dtype=np.float32)
    reference = np.array([[0.0, 0.002], [0.001, 0.0], [0.0, -0.002]], dtype=np.float32)
    # Evaluate the manuscript formula independently in float64, explicitly
    # excluding each query instead of using the scorer's sum-of-squares shortcut.
    expected = []
    ref = reference.astype(np.float64)
    for index, row in enumerate(query.astype(np.float64)):
        pool = np.delete(query.astype(np.float64), index, axis=0)
        scale = np.clip(np.sqrt((ref.var(axis=0, ddof=1) + 1e-6) / (pool.var(axis=0, ddof=1) + 1e-6)), 0.5, 2.0)
        calibrated = ref.mean(axis=0) + (row - pool.mean(axis=0)) * scale
        expected.append(calibrated / max(np.linalg.norm(calibrated), 1e-8))

    np.testing.assert_allclose(scorer.loo_variance_calibrate(query, reference), expected, atol=1e-6)


def test_neighbour_selection_and_residual_variance_use_normalization_floor() -> None:
    train = np.array([[3e-8, 4e-8], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.array([[3.0, 4.0]], dtype=np.float32)
    residuals = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)

    np.testing.assert_array_equal(scorer.local_indices_by_similarity(train, candidate, k=2), [[0, 2]])
    np.testing.assert_array_equal(
        scorer.local_indices_by_similarity(train, candidate, k=2, norm_eps=1e-6), [[2, 1]]
    )
    np.testing.assert_allclose(
        scorer.candidate_local_variance(train, candidate, residuals, np.zeros_like(residuals), k=2, shrink=0.0),
        [[8.0]],
    )
    np.testing.assert_allclose(
        scorer.candidate_local_variance(
            train, candidate, residuals, np.zeros_like(residuals), k=2, shrink=0.0, norm_eps=1e-6
        ),
        [[2.0]],
    )


def test_layer_shrinkage_default_and_legacy_normalization_override() -> None:
    # With shrink=0, the two normalization passes expose a floor mismatch even
    # after the input has been normalized once; the zero row remains finite.
    layer = np.array([[3e-18, 4e-18], [0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    default = scorer.shrink_layer_features([layer], [layer], shrink=0.0)
    legacy = scorer.shrink_layer_features([layer], [layer], 0.0, 1e-6)
    explicit = scorer.shrink_layer_features([layer], [layer], 0.0, norm_eps=1e-6)
    override = scorer.shrink_layer_features([layer], [layer], 0.0, 1e-6, norm_eps=1e-8)

    for train_or_test in default:
        np.testing.assert_allclose(train_or_test[0], [[0.03, 0.04], [0.0, 0.0], [0.6, 0.8]], atol=1e-7)
    for legacy_side, explicit_side, default_side, override_side in zip(legacy, explicit, default, override):
        np.testing.assert_allclose(legacy_side[0], explicit_side[0])
        np.testing.assert_allclose(legacy_side[0][0], [3e-6, 4e-6], atol=1e-12)
        np.testing.assert_allclose(default_side[0], override_side[0])


def test_variance_mse_and_score_floors_remain_one_micro() -> None:
    np.testing.assert_allclose(scorer.row_minmax(np.array([[0.0, 2e-8, 4e-8]])), [[0.0, 0.02, 0.04]], atol=1e-7)
    features = np.eye(2, dtype=np.float32)
    zeros = np.zeros((2, 2), dtype=np.float32)
    variance = scorer.candidate_local_variance(features, features, zeros, zeros, k=2)
    np.testing.assert_allclose(variance, np.full((2, 2), 1e-6), rtol=1e-7)
    np.testing.assert_allclose(
        scorer.inverse_mse_layer_weights(zeros, [zeros, np.full_like(zeros, 2e-4)], kappa=0.0), [0.5, 0.5]
    )


@pytest.mark.parametrize(
    ("options", "expected_norm", "expected_stat"),
    [({}, 1e-8, 1e-6), ({"eps": 0.003}, 1e-8, 0.003), ({"eps": 0.003, "norm_eps": 7e-8}, 7e-8, 0.003)],
)
def test_score_branch_threads_separate_tolerances(monkeypatch, options, expected_norm, expected_stat) -> None:
    norm_calls = []
    stat_calls = {}
    original_normalize = scorer.l2_normalize

    def record_normalize(x, eps=scorer.NORM_EPS):
        norm_calls.append(eps)
        return original_normalize(x, eps)

    monkeypatch.setattr(scorer, "l2_normalize", record_normalize)
    for name in ("loo_variance_calibrate", "candidate_local_variance", "inverse_mse_layer_weights", "row_minmax"):
        original = getattr(scorer, name)
        signature = inspect.signature(original)
        stat_calls[name] = []

        def record_stat(*args, _name=name, _original=original, _signature=signature, **kwargs):
            bound = _signature.bind(*args, **kwargs)
            bound.apply_defaults()
            stat_calls[_name].append(bound.arguments["eps"])
            return _original(*args, **kwargs)

        monkeypatch.setattr(scorer, name, record_stat)

    rng = np.random.default_rng(17)
    train = rng.normal(size=(8, 3)).astype(np.float32) * 5e-8
    query = rng.normal(size=(3, 3)).astype(np.float32) * 5e-8
    result = scorer.score_branch(train, query, train, query, [train, train[:, ::-1]], [query, query[:, ::-1]], **options)

    assert norm_calls and all(value == expected_norm for value in norm_calls)
    for calls in stat_calls.values():
        assert calls and all(value == expected_stat for value in calls)
    assert np.isfinite(result.final_score).all()


def test_zero_embeddings_are_finite_through_full_scoring() -> None:
    train = np.zeros((4, 3), dtype=np.float32)
    query = np.zeros((3, 3), dtype=np.float32)

    branch = scorer.score_branch(train, query, train, query, [train], [query])
    fused = scorer.fuse_branches({"G": branch, "H": branch})

    np.testing.assert_array_equal(branch.global_prediction, np.zeros((3, 3)))
    np.testing.assert_array_equal(fused, np.zeros((3, 3)))
    assert np.isfinite(fused).all()
