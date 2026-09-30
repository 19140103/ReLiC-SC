"""Full, fixed-configuration ReLiC-SC scoring.

This module is the reference implementation of the scoring configuration used in
the paper.  It keeps the response mappers and all candidate-dependent quantities
separate from per-query calibration and score fusion:

* a ridge mapper predicts one EEG embedding for every candidate;
* leave-one-query-out mean/variance calibration uses those predictions as the
  reference distribution;
* the global input is scored with a variance-weighted residual and each visual
  layer with response agreement;
* scores are min--max normalized per query before branch/layer fusion.

The defaults are intentionally fixed (rather than tuned by this script) so that
the command-line entry point reproduces the shared paper configuration.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


# Shared configuration reported in the manuscript.
RIDGE = 0.1
VARIANCE_POWER = 1.0
VARIANCE_CLIP = 2.0
LOCAL_K = 256
LOCAL_SHRINK = 0.5
LAYER_FEATURE_SHRINK = 0.5
KAPPA = 0.25
GAMMA = 0.5
EPS = 1e-6


def l2_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Normalize rows of an embedding matrix, preserving float32 output."""

    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), float(eps))


def row_minmax(scores: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Min--max normalize each query over its candidate set."""

    scores = np.asarray(scores, dtype=np.float32)
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    return (scores - lo) / np.maximum(hi - lo, float(eps))


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Fit a multi-output ridge mapper with an unregularized intercept."""

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be two-dimensional arrays with matching rows")
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    gram = x_aug.T @ x_aug
    reg = np.eye(gram.shape[0], dtype=np.float32) * float(ridge)
    reg[-1, -1] = 0.0
    # solve is more stable than an explicit inverse and matches the closed form
    # used by the original analysis script.
    return np.linalg.solve(gram + reg, x_aug.T @ y).astype(np.float32)


def apply_ridge(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Apply a mapper returned by :func:`fit_ridge`."""

    x = np.asarray(x, dtype=np.float32)
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    if x_aug.shape[1] != weight.shape[0]:
        raise ValueError("mapper input dimension does not match x")
    return (x_aug @ weight).astype(np.float32)


def loo_variance_calibrate(
    query: np.ndarray,
    reference: np.ndarray,
    power: float = VARIANCE_POWER,
    clip: float = VARIANCE_CLIP,
    eps: float = EPS,
) -> np.ndarray:
    """Align each query's mean and diagonal scale to candidate predictions.

    The query pool is treated leave-one-out: the current query is omitted when
    estimating its observed mean and variance.  ``power=1`` and ``clip=2`` are
    the fixed paper settings.  For a singleton pool the available observation is
    used directly and its variance is set to zero.
    """

    query = np.asarray(query, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query and reference must be [n, d] and [m, d]")
    if reference.shape[0] < 1:
        raise ValueError("reference must contain at least one candidate")

    n = query.shape[0]
    ref_mean = reference.mean(axis=0, keepdims=True)
    ref_var = reference.var(axis=0, ddof=1 if reference.shape[0] > 1 else 0, keepdims=True)
    if n > 1:
        q_sum = query.sum(axis=0, keepdims=True)
        q_sq_sum = (query * query).sum(axis=0, keepdims=True)
        q_mean = (q_sum - query) / float(n - 1)
        q_mean_sq = (q_sq_sum - query * query) / float(n - 1)
        # The expression above is the population second moment.  Convert it to
        # the sample variance, matching np.var(..., ddof=1).
        q_var = np.maximum(q_mean_sq - q_mean * q_mean, 0.0) * float(n - 1) / float(max(1, n - 2))
    else:
        q_mean = query.copy()
        q_var = np.zeros_like(query)

    scale = np.sqrt((ref_var + float(eps)) / (q_var + float(eps)))
    scale = np.clip(scale, 1.0 / float(clip), float(clip)) ** float(power)
    return l2_normalize(ref_mean + (query - q_mean) * scale, eps=eps)


# A descriptive alias used by callers that do not need to mention the LOO detail.
calibrate = loo_variance_calibrate


def local_indices_by_similarity(
    train_features: np.ndarray,
    candidate_features: np.ndarray,
    k: int = LOCAL_K,
    chunk: int = 64,
) -> np.ndarray:
    """Return the top-k training neighbours for every candidate."""

    train_features = l2_normalize(train_features)
    candidate_features = l2_normalize(candidate_features)
    if train_features.shape[0] == 0:
        raise ValueError("train_features must not be empty")
    k = min(max(1, int(k)), train_features.shape[0])
    rows: list[np.ndarray] = []
    for start in range(0, candidate_features.shape[0], max(1, int(chunk))):
        end = min(start + max(1, int(chunk)), candidate_features.shape[0])
        sim = candidate_features[start:end] @ train_features.T
        idx = np.argpartition(sim, kth=sim.shape[1] - k, axis=1)[:, -k:]
        local_rows = np.arange(idx.shape[0])[:, None]
        idx = idx[local_rows, np.argsort(-sim[local_rows, idx], axis=1)]
        rows.append(idx.astype(np.int64, copy=False))
    return np.concatenate(rows, axis=0)


def candidate_local_variance(
    train_features: np.ndarray,
    candidate_features: np.ndarray,
    train_query: np.ndarray,
    train_prediction: np.ndarray,
    k: int = LOCAL_K,
    shrink: float = LOCAL_SHRINK,
    eps: float = EPS,
) -> np.ndarray:
    """Estimate candidate-specific diagonal residual variance.

    Local residual variances are shrunk toward the global training residual
    variance.  The returned array is shaped ``[n_candidates, d]``.
    """

    residual = np.asarray(train_query, dtype=np.float32) - np.asarray(train_prediction, dtype=np.float32)
    if residual.shape[0] < 1:
        raise ValueError("at least one training residual is required")
    ddof = 1 if residual.shape[0] > 1 else 0
    global_var = residual.var(axis=0, ddof=ddof).astype(np.float32)
    idx = local_indices_by_similarity(train_features, candidate_features, k=k)
    local = residual[idx]
    local_var = local.var(axis=1, ddof=1 if local.shape[1] > 1 else 0).astype(np.float32)
    mixed = (1.0 - float(shrink)) * local_var + float(shrink) * global_var[None, :]
    return np.maximum(mixed, float(eps)).astype(np.float32)


def variance_weighted_residual_score(
    query: np.ndarray,
    prediction: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """Compute the global variance-weighted residual score ``B``."""

    query = np.asarray(query, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    variance = np.asarray(variance, dtype=np.float32)
    return -np.sum((query[:, None, :] - prediction[None, :, :]) ** 2 / variance[None, :, :], axis=2)


def response_agreement_score(query: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute layer response agreement ``A`` by inner product."""

    return np.asarray(query, dtype=np.float32) @ np.asarray(prediction, dtype=np.float32).T


def inverse_mse_layer_weights(
    train_query: np.ndarray,
    train_predictions: Sequence[np.ndarray],
    kappa: float = KAPPA,
    eps: float = EPS,
) -> np.ndarray:
    """Compute inverse-MSE layer weights regularized toward uniform weights."""

    if not train_predictions:
        raise ValueError("train_predictions must contain at least one layer")
    mse = np.asarray(
        [np.mean((np.asarray(train_query) - np.asarray(pred)) ** 2) for pred in train_predictions],
        dtype=np.float64,
    )
    reliability = 1.0 / np.maximum(mse, float(eps))
    normalized = reliability / max(float(reliability.sum()), float(eps))
    uniform = np.full_like(normalized, 1.0 / normalized.size)
    weights = (1.0 - float(kappa)) * normalized + float(kappa) * uniform
    return (weights / max(float(weights.sum()), float(eps))).astype(np.float32)


def shrink_layer_features(
    train_layers: Sequence[np.ndarray],
    test_layers: Sequence[np.ndarray],
    shrink: float = LAYER_FEATURE_SHRINK,
    eps: float = EPS,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Shrink each visual layer toward the per-image mean visual feature."""

    if len(train_layers) != len(test_layers) or not train_layers:
        raise ValueError("train_layers and test_layers must have the same non-zero length")
    train_norm = [l2_normalize(x, eps) for x in train_layers]
    test_norm = [l2_normalize(x, eps) for x in test_layers]
    mean_train = l2_normalize(np.mean(train_norm, axis=0), eps)
    mean_test = l2_normalize(np.mean(test_norm, axis=0), eps)
    train_out = [l2_normalize((1.0 - float(shrink)) * x + float(shrink) * mean_train, eps) for x in train_norm]
    test_out = [l2_normalize((1.0 - float(shrink)) * x + float(shrink) * mean_test, eps) for x in test_norm]
    return train_out, test_out


@dataclass
class BranchScore:
    """All fixed-configuration scores for one EEG branch."""

    global_score: np.ndarray
    layer_scores: list[np.ndarray]
    global_normalized: np.ndarray
    layer_normalized: list[np.ndarray]
    layer_weights: np.ndarray
    final_score: np.ndarray
    global_prediction: np.ndarray
    layer_predictions: list[np.ndarray]


def score_branch(
    train_query: np.ndarray,
    test_query: np.ndarray,
    train_global: np.ndarray,
    test_global: np.ndarray,
    train_layers: Sequence[np.ndarray],
    test_layers: Sequence[np.ndarray],
    *,
    ridge: float = RIDGE,
    variance_power: float = VARIANCE_POWER,
    variance_clip: float = VARIANCE_CLIP,
    local_k: int = LOCAL_K,
    local_shrink: float = LOCAL_SHRINK,
    layer_feature_shrink: float = LAYER_FEATURE_SHRINK,
    kappa: float = KAPPA,
    eps: float = EPS,
) -> BranchScore:
    """Compute the complete score for one branch.

    ``train_*`` arrays are paired rows.  ``test_query[i]`` is assumed to have
    candidate ``i`` as its ground-truth image, which is the convention used by
    the released evaluation files; the function itself does not use labels.
    """

    train_query = l2_normalize(train_query, eps)
    test_query = l2_normalize(test_query, eps)
    train_global = l2_normalize(train_global, eps)
    test_global = l2_normalize(test_global, eps)

    global_weight = fit_ridge(train_global, train_query, ridge)
    train_global_pred = l2_normalize(apply_ridge(train_global, global_weight), eps)
    test_global_pred = l2_normalize(apply_ridge(test_global, global_weight), eps)
    calibrated_global = loo_variance_calibrate(test_query, test_global_pred, variance_power, variance_clip, eps)
    global_var = candidate_local_variance(
        train_global,
        test_global,
        train_query,
        train_global_pred,
        local_k,
        local_shrink,
        eps,
    )
    global_score = variance_weighted_residual_score(calibrated_global, test_global_pred, global_var)
    global_normalized = row_minmax(global_score, eps)

    shrunk_train, shrunk_test = shrink_layer_features(train_layers, test_layers, layer_feature_shrink, eps)
    layer_predictions_train: list[np.ndarray] = []
    layer_predictions_test: list[np.ndarray] = []
    layer_scores: list[np.ndarray] = []
    for layer_train, layer_test in zip(shrunk_train, shrunk_test):
        weight = fit_ridge(layer_train, train_query, ridge)
        pred_train = l2_normalize(apply_ridge(layer_train, weight), eps)
        pred_test = l2_normalize(apply_ridge(layer_test, weight), eps)
        calibrated_layer = loo_variance_calibrate(test_query, pred_test, variance_power, variance_clip, eps)
        layer_scores.append(response_agreement_score(calibrated_layer, pred_test))
        layer_predictions_train.append(pred_train)
        layer_predictions_test.append(pred_test)

    layer_weights = inverse_mse_layer_weights(train_query, layer_predictions_train, kappa, eps)
    layer_normalized = [row_minmax(score, eps) for score in layer_scores]
    layer_aggregate = np.sum(np.stack([w * score for w, score in zip(layer_weights, layer_normalized)]), axis=0)
    # The final mix is applied after branch/layer normalization.  For one branch
    # this still returns a useful score and makes the function easy to test.
    final_score = (1.0 - float(GAMMA)) * row_minmax(global_normalized, eps) + float(GAMMA) * row_minmax(layer_aggregate, eps)
    return BranchScore(
        global_score=global_score,
        layer_scores=layer_scores,
        global_normalized=global_normalized,
        layer_normalized=layer_normalized,
        layer_weights=layer_weights,
        final_score=final_score.astype(np.float32),
        global_prediction=test_global_pred,
        layer_predictions=layer_predictions_test,
    )


def fuse_branches(branch_scores: Mapping[str, BranchScore], gamma: float = GAMMA, eps: float = EPS) -> np.ndarray:
    """Fuse G/H (or D4/D12) scores using the paper's final equation."""

    if not branch_scores:
        raise ValueError("at least one branch is required")
    ordered = list(branch_scores.values())
    global_part = np.mean([score.global_normalized for score in ordered], axis=0)
    layer_part = np.mean(
        [np.sum(np.stack([w * s for w, s in zip(score.layer_weights, score.layer_normalized)]), axis=0) for score in ordered],
        axis=0,
    )
    return ((1.0 - float(gamma)) * row_minmax(global_part, eps) + float(gamma) * row_minmax(layer_part, eps)).astype(np.float32)


def ranks(scores: np.ndarray) -> np.ndarray:
    """Return one-based rank of the diagonal candidate for each query."""

    labels = np.arange(scores.shape[0])
    order = np.argsort(-scores, axis=1)
    return np.asarray([int(np.where(order[i] == labels[i])[0][0]) + 1 for i in range(scores.shape[0])], dtype=np.int64)


def metrics(scores: np.ndarray) -> dict[str, float]:
    r = ranks(scores)
    return {
        "top1": float(np.mean(r <= 1) * 100.0),
        "top5": float(np.mean(r <= 5) * 100.0),
        "top10": float(np.mean(r <= 10) * 100.0),
        "mrr": float(np.mean(1.0 / r)),
        "mean_rank": float(np.mean(r)),
    }


def gather_layer(bank_train: np.ndarray, bank_test: np.ndarray, data: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Gather cached object/image banks using the indices in an embedding NPZ."""

    train = bank_train[data["train_object_idx"].astype(np.int64), data["train_image_idx"].astype(np.int64)]
    test = bank_test[data["test_object_idx"].astype(np.int64), data["test_image_idx"].astype(np.int64)]
    return np.asarray(train, dtype=np.float32), np.asarray(test, dtype=np.float32)


def load_layer_banks(root: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    names: list[str] = []
    for level in sorted(root.glob("level-*")):
        train_path = level / "image_train.npy"
        test_path = level / "image_test.npy"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"missing image bank under {level}")
        train.append(np.load(train_path, mmap_mode="r"))
        test.append(np.load(test_path, mmap_mode="r"))
        names.append(level.name)
    if not train:
        raise FileNotFoundError(f"no level-* image banks under {root}")
    return train, test, names


def _load_branch(data: Mapping[str, np.ndarray], bank_train: Sequence[np.ndarray], bank_test: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    required = ("train_pred", "test_pred", "train_image", "test_image", "train_object_idx", "train_image_idx", "test_object_idx", "test_image_idx")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"embedding file is missing {', '.join(missing)}")
    layers_train, layers_test = [], []
    for tr_bank, te_bank in zip(bank_train, bank_test):
        tr, te = gather_layer(tr_bank, te_bank, data)
        layers_train.append(tr)
        layers_test.append(te)
    return data["train_pred"], data["test_pred"], data["train_image"], data["test_image"], layers_train, layers_test


def _parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"run specification must be label=run_name: {spec}")
    return tuple(spec.split("=", 1))  # type: ignore[return-value]


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed full ReLiC-SC scoring from saved embeddings.")
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--image-bank-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--run-spec", dest="run_specs", action="append", required=True, help="label=run_name")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _cli()
    train_banks, test_banks, _ = load_layer_banks(args.image_bank_root)
    specs = [_parse_run_spec(spec) for spec in args.run_specs]
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for subject in args.subjects:
            branch_results: dict[str, BranchScore] = {}
            for label, run_name in specs:
                path = args.embedding_root / subject / run_name / f"seed-{seed}" / "train_test_embeddings.npz"
                if not path.exists():
                    print(f"[warn] missing {path}")
                    continue
                with np.load(path) as npz:
                    data = {key: npz[key] for key in npz.files}
                train_q, test_q, train_g, test_g, train_l, test_l = _load_branch(data, train_banks, test_banks)
                result = score_branch(train_q, test_q, train_g, test_g, train_l, test_l)
                branch_results[label] = result
                for method, score in (("global_residual", result.global_score), ("layer_agreement", np.sum(np.stack([w * s for w, s in zip(result.layer_weights, result.layer_normalized)]), axis=0)), ("full", result.final_score)):
                    row = {"seed": seed, "subject": subject, "scope": "branch", "branch": label, "method": method}
                    row.update(metrics(score))
                    rows.append(row)
            if branch_results:
                fused = fuse_branches(branch_results)
                row = {"seed": seed, "subject": subject, "scope": "fusion", "branch": "+".join(branch_results), "method": "full"}
                row.update(metrics(fused))
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "subject", "scope", "branch", "method", "top1", "top5", "top10", "mrr", "mean_rank"]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
