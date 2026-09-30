"""Leakage-safe utilities for the NeuroTriage short-calibration gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationSplit:
    """Chronological calibration pool and a common image-disjoint evaluation set."""

    calibration_pool: pd.DataFrame
    evaluation: pd.DataFrame
    cutoff_trial_order: int


def build_calibration_split(
    trials: pd.DataFrame,
    heldout_session: int,
    max_budget: int,
) -> CalibrationSplit:
    """Build one common evaluation set for all calibration budgets.

    The first ``max_budget`` retained trials form the largest calibration pool.
    Evaluation occurs strictly later and removes every image identity seen in
    that pool.  Smaller budgets are prefixes of the same pool, so budget
    comparisons cannot be caused by evaluating on different examples.
    """

    required = {"array_row", "session", "image_id", "trial_order_index"}
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"Trial table is missing calibration columns: {missing}")
    if max_budget < 1:
        raise ValueError("max_budget must be positive")
    session = trials.loc[trials["session"].eq(int(heldout_session))].copy()
    session.sort_values("trial_order_index", kind="stable", inplace=True)
    if len(session) <= max_budget:
        raise ValueError("Held-out session is too small for calibration and evaluation")

    pool = session.iloc[:max_budget].copy()
    cutoff = int(pool["trial_order_index"].max())
    calibration_images = set(pool["image_id"])
    evaluation = session.loc[
        session["trial_order_index"].gt(cutoff)
        & ~session["image_id"].isin(calibration_images)
    ].copy()
    if evaluation.empty:
        raise ValueError("Image-disjoint calibration evaluation set is empty")
    if set(pool["image_id"]).intersection(evaluation["image_id"]):
        raise RuntimeError("Calibration image identities leaked into evaluation")
    if int(evaluation["trial_order_index"].min()) <= cutoff:
        raise RuntimeError("Calibration evaluation is not strictly chronological")
    return CalibrationSplit(
        calibration_pool=pool.reset_index(drop=True),
        evaluation=evaluation.reset_index(drop=True),
        cutoff_trial_order=cutoff,
    )


def fit_residual_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    base_weight: np.ndarray,
    base_bias: np.ndarray,
    ridge_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a closed-form residual map tethered to the frozen base adapter.

    Solving in the sample-space dual keeps the largest inverse at ``budget x
    budget`` (at most 180 in the pilot), rather than 513 x 513.
    """

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    weight = np.asarray(base_weight, dtype=np.float64)
    bias = np.asarray(base_bias, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("Features and targets must be aligned matrices")
    if weight.shape != (y.shape[1], x.shape[1]) or bias.shape != (y.shape[1],):
        raise ValueError("Base adapter dimensions do not match features and targets")
    if ridge_ratio <= 0:
        raise ValueError("ridge_ratio must be positive")

    augmented = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    base_output = x @ weight.T + bias
    residual = y - base_output
    gram = augmented @ augmented.T
    feature_energy = max(float(np.trace(gram) / len(x)), 1e-12)
    ridge = float(ridge_ratio * feature_energy)
    dual = np.linalg.solve(
        gram + ridge * np.eye(len(x), dtype=gram.dtype), residual
    )
    delta = augmented.T @ dual
    adapted_weight = weight + delta[:-1].T
    adapted_bias = bias + delta[-1]
    report = {
        "ridge": ridge,
        "feature_energy": feature_energy,
        "delta_weight_frobenius": float(np.linalg.norm(delta[:-1])),
        "delta_bias_l2": float(np.linalg.norm(delta[-1])),
    }
    return (
        adapted_weight.astype(np.float32),
        adapted_bias.astype(np.float32),
        report,
    )


def normalized_adapter_output(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    output = (
        np.asarray(features, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T
        + np.asarray(bias, dtype=np.float32)
    )
    return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-8)


def fit_semantic_residual_ridge(
    base_predictions: np.ndarray,
    targets: np.ndarray,
    ridge_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Learn a low-rank, identity-tethered correction in normalized space.

    The correction rank is at most the number of calibration examples.  This
    avoids allowing 30--180 labels to freely rewrite a 512 x 512 EEG adapter.
    """

    prediction = np.asarray(base_predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError("Base predictions and targets must be aligned matrices")
    if ridge_ratio <= 0:
        raise ValueError("ridge_ratio must be positive")
    prediction /= np.maximum(np.linalg.norm(prediction, axis=1, keepdims=True), 1e-12)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    residual = target - prediction
    gram = prediction @ prediction.T
    feature_energy = max(float(np.trace(gram) / len(prediction)), 1e-12)
    ridge = float(ridge_ratio * feature_energy)
    dual = np.linalg.solve(
        gram + ridge * np.eye(len(prediction), dtype=gram.dtype), residual
    )
    correction = prediction.T @ dual
    report = {
        "ridge": ridge,
        "feature_energy": feature_energy,
        "correction_frobenius": float(np.linalg.norm(correction)),
        "maximum_rank": int(len(prediction)),
    }
    return correction.astype(np.float32), report


def apply_semantic_residual(
    base_predictions: np.ndarray,
    correction: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(base_predictions, dtype=np.float32)
    adapted = prediction + prediction @ np.asarray(correction, dtype=np.float32)
    return adapted / np.maximum(np.linalg.norm(adapted, axis=1, keepdims=True), 1e-8)


def paired_target_similarity(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    prediction = np.asarray(predictions, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    return np.sum(prediction * target, axis=1)
