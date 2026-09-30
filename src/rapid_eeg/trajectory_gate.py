"""Utilities for the cheap CEVR evidence-trajectory Go/No-Go experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RepeatPairSplit:
    """Chronological adjacent-repeat pairs split above the trial level."""

    train_pairs: pd.DataFrame
    evaluation_pairs: pd.DataFrame


def build_repeat_pair_split(
    trials: pd.DataFrame,
    heldout_session: int,
    exclude_evaluation_images_from_train: bool = True,
) -> RepeatPairSplit:
    """Pair each retained view with the next same-image view in its session.

    Evaluation uses only the held-out session.  By default, images present in
    that evaluation session are also excluded from the probe-training pairs,
    preventing an image-identity shortcut in the early trajectory gate.
    """

    required = {"array_row", "session", "image_id", "trial_order_index"}
    missing = sorted(required.difference(trials.columns))
    if missing:
        raise ValueError(f"Trial table is missing repeat-pair columns: {missing}")
    ordered = trials.sort_values(
        ["session", "image_id", "trial_order_index"], kind="stable"
    ).copy()
    grouped = ordered.groupby(["session", "image_id"], sort=False, observed=True)
    ordered["next_array_row"] = grouped["array_row"].shift(-1)
    ordered["next_trial_order_index"] = grouped["trial_order_index"].shift(-1)
    pairs = ordered.loc[ordered["next_array_row"].notna()].copy()
    pairs.rename(columns={"array_row": "base_array_row"}, inplace=True)
    pairs["base_array_row"] = pairs["base_array_row"].astype("int64")
    pairs["next_array_row"] = pairs["next_array_row"].astype("int64")

    session = int(heldout_session)
    evaluation = pairs.loc[pairs["session"].eq(session)].copy()
    training = pairs.loc[pairs["session"].ne(session)].copy()
    if exclude_evaluation_images_from_train:
        evaluation_images = set(evaluation["image_id"])
        training = training.loc[~training["image_id"].isin(evaluation_images)].copy()
    if training.empty or evaluation.empty:
        raise ValueError("Repeat split requires non-empty training and held-out pairs")
    if exclude_evaluation_images_from_train and set(training["image_id"]).intersection(
        evaluation["image_id"]
    ):
        raise RuntimeError("Evaluation image identities leaked into probe training")
    return RepeatPairSplit(
        train_pairs=training.reset_index(drop=True),
        evaluation_pairs=evaluation.reset_index(drop=True),
    )


def realized_repeat_gain(
    fused_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    base_positions: np.ndarray,
    next_positions: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    """Cosine-to-target gain after precision-neutral two-trial mean fusion."""

    fused = np.asarray(fused_embeddings, dtype=np.float32)
    targets = np.asarray(target_embeddings, dtype=np.float32)
    base = fused[np.asarray(base_positions, dtype=np.int64)]
    following = fused[np.asarray(next_positions, dtype=np.int64)]
    target = targets[np.asarray(target_positions, dtype=np.int64)]
    base /= np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-8)
    following /= np.maximum(np.linalg.norm(following, axis=1, keepdims=True), 1e-8)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    repeated = base + following
    repeated /= np.maximum(np.linalg.norm(repeated, axis=1, keepdims=True), 1e-8)
    return np.sum(repeated * target, axis=1) - np.sum(base * target, axis=1)


def prediction_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    costs: np.ndarray,
) -> dict[str, object]:
    """Return gain regression and diagnostic repeat-vs-stop regret metrics."""

    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("Actual and predicted gains must be aligned one-dimensional arrays")
    error = predicted - actual
    pearson = float(np.corrcoef(actual, predicted)[0, 1]) if actual.size > 1 else float("nan")
    actual_rank = pd.Series(actual).rank(method="average").to_numpy()
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy()
    spearman = (
        float(np.corrcoef(actual_rank, predicted_rank)[0, 1])
        if actual.size > 1
        else float("nan")
    )
    regrets: list[float] = []
    repeat_rates: list[float] = []
    oracle_rates: list[float] = []
    for cost in np.asarray(costs, dtype=np.float64):
        oracle_utility = np.maximum(actual - cost, 0.0)
        choose_repeat = predicted > cost
        policy_utility = np.where(choose_repeat, actual - cost, 0.0)
        regrets.append(float(np.mean(oracle_utility - policy_utility)))
        repeat_rates.append(float(np.mean(choose_repeat)))
        oracle_rates.append(float(np.mean(actual > cost)))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": pearson,
        "spearman": spearman,
        "mean_action_regret": float(np.mean(regrets)),
        "costs": [float(value) for value in costs],
        "action_regret_by_cost": regrets,
        "predicted_repeat_rate_by_cost": repeat_rates,
        "oracle_repeat_rate_by_cost": oracle_rates,
    }
