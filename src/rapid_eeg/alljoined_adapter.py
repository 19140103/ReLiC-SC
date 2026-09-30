"""Leakage-safe Alljoined-1.6M trial tables for CEVR-derived experiments.

This module is intentionally independent from the author's original CEVR
repository.  It keeps every retained EEG epoch as one observation, reconstructs
the surrounding RSVP stream from the unfiltered metadata, and creates the
three disjoint roles required by a leave-one-session-out experiment:

* decoder training from non-held-out ``stim_train`` sessions;
* chronological calibration from held-out-session ``stim_train`` trials;
* evaluation from held-out-session ``stim_test`` trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRIAL_ORDER_COLUMNS = ("session", "block_id", "sequence_id", "sequence_image_id")
SEQUENCE_COLUMNS = ("subject", "session", "block_id", "sequence_id")
DEFAULT_NEIGHBOR_OFFSETS = (-4, -3, -2, -1, 1, 2, 3, 4)

_REQUIRED_COLUMNS = {
    "image_path",
    "subject",
    "session",
    "block_id",
    "sequence_id",
    "sequence_image_id",
    "partition",
    "dropped",
}


def image_id(image_path: object) -> str:
    """Return a platform-independent image identifier from an official path."""

    return str(image_path).replace("\\", "/").rsplit("/", 1)[-1]


def _neighbor_prefix(offset: int) -> str:
    direction = "next" if offset > 0 else "prev"
    return f"neighbor_{direction}_{abs(offset)}"


def enrich_trial_metadata(
    raw_metadata: pd.DataFrame,
    partition: str,
    neighbor_offsets: Iterable[int] = DEFAULT_NEIGHBOR_OFFSETS,
) -> pd.DataFrame:
    """Build an EEG-array-aligned table while retaining RSVP neighbors.

    Neighbors are calculated before filtering dropped and oddball rows.  This is
    essential: shifting only the retained partition would silently jump across
    stimuli that were present in the RSVP stream but absent from the EEG array.
    The returned ``array_row`` is aligned with the corresponding filtered EEG
    array produced by ``scripts/prepare_pilot.py``.
    """

    missing = sorted(_REQUIRED_COLUMNS.difference(raw_metadata.columns))
    if missing:
        raise ValueError(f"Alljoined metadata is missing required columns: {missing}")
    if raw_metadata.index.has_duplicates:
        raise ValueError("Alljoined metadata index must be unique for EEG alignment")

    offsets = tuple(int(value) for value in neighbor_offsets)
    if any(value == 0 for value in offsets) or len(set(offsets)) != len(offsets):
        raise ValueError("Neighbor offsets must be unique non-zero integers")

    frame = raw_metadata.copy()
    frame.insert(0, "source_row_index", frame.index.to_numpy(dtype=np.int64))
    frame["image_id"] = frame["image_path"].map(image_id)
    frame.sort_values(list(TRIAL_ORDER_COLUMNS), kind="stable", inplace=True)

    if frame.duplicated(list(SEQUENCE_COLUMNS) + ["sequence_image_id"]).any():
        raise ValueError("Duplicate RSVP positions found within a sequence")

    frame["trial_order_index"] = (
        frame.groupby(["subject", "session"], sort=False).cumcount().astype("int32")
    )
    grouped = frame.groupby(list(SEQUENCE_COLUMNS), sort=False, observed=True)
    for offset in offsets:
        prefix = _neighbor_prefix(offset)
        shifted_position = grouped["sequence_image_id"].shift(-offset)
        expected_position = frame["sequence_image_id"] + offset
        valid = shifted_position.eq(expected_position)
        for source_column, suffix in (
            ("image_id", "image_id"),
            ("partition", "partition"),
            ("dropped", "dropped"),
            ("source_row_index", "source_row_index"),
        ):
            shifted = grouped[source_column].shift(-offset)
            frame[f"{prefix}_{suffix}"] = shifted.where(valid)

    selected = frame.loc[
        frame["partition"].eq(partition) & ~frame["dropped"].astype(bool)
    ].copy()
    selected.sort_values("source_row_index", kind="stable", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    selected.insert(0, "array_row", np.arange(len(selected), dtype=np.int64))
    selected["repetition_id"] = (
        selected.groupby(["subject", "image_id"], sort=False)
        .cumcount()
        .astype("int32")
    )
    selected["session_repetition_id"] = (
        selected.groupby(["subject", "session", "image_id"], sort=False)
        .cumcount()
        .astype("int16")
    )

    if not selected["source_row_index"].is_monotonic_increasing:
        raise RuntimeError("Filtered metadata no longer matches official EEG row order")
    return selected


def validate_eeg_alignment(
    eeg_path: Path,
    trial_table: pd.DataFrame,
    expected_channels: int = 32,
    expected_samples: int = 250,
) -> tuple[int, int, int]:
    """Validate table-to-array alignment without loading the full EEG array."""

    eeg = np.load(eeg_path, mmap_mode="r")
    expected_shape = (len(trial_table), expected_channels, expected_samples)
    if eeg.shape != expected_shape:
        raise ValueError(
            f"EEG/table mismatch for {eeg_path}: got {eeg.shape}, expected {expected_shape}"
        )
    if not np.array_equal(
        trial_table["array_row"].to_numpy(dtype=np.int64),
        np.arange(len(trial_table), dtype=np.int64),
    ):
        raise ValueError("array_row must be a contiguous zero-based EEG index")
    return tuple(int(value) for value in eeg.shape)


@dataclass(frozen=True)
class AlljoinedLOSOFold:
    """Indices and repeat groups for one leakage-safe session holdout."""

    heldout_session: int
    decoder_train_indices: np.ndarray
    calibration_pool_indices: np.ndarray
    evaluation_indices: np.ndarray
    calibration_prefixes: dict[int, np.ndarray]
    evaluation_repeat_groups: dict[str, np.ndarray]

    def summary(self) -> dict[str, object]:
        return {
            "heldout_session": self.heldout_session,
            "decoder_train_trials": int(len(self.decoder_train_indices)),
            "calibration_pool_trials": int(len(self.calibration_pool_indices)),
            "evaluation_trials": int(len(self.evaluation_indices)),
            "evaluation_images": int(len(self.evaluation_repeat_groups)),
            "minimum_evaluation_repeats": int(
                min(map(len, self.evaluation_repeat_groups.values()), default=0)
            ),
            "maximum_evaluation_repeats": int(
                max(map(len, self.evaluation_repeat_groups.values()), default=0)
            ),
            "calibration_budgets": sorted(self.calibration_prefixes),
        }


def build_loso_fold(
    train_trials: pd.DataFrame,
    test_trials: pd.DataFrame,
    heldout_session: int,
    calibration_budgets: Iterable[int] = (0, 30, 60, 180),
) -> AlljoinedLOSOFold:
    """Create decoder-train, calibration, and evaluation roles for one fold."""

    required = {"array_row", "session", "partition", "image_id", "trial_order_index"}
    for name, frame in (("train", train_trials), ("test", test_trials)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} trial table is missing columns: {missing}")

    if set(train_trials["partition"].unique()) != {"stim_train"}:
        raise ValueError("Decoder/calibration table must contain only stim_train trials")
    if set(test_trials["partition"].unique()) != {"stim_test"}:
        raise ValueError("Evaluation table must contain only stim_test trials")
    image_overlap = set(train_trials["image_id"]).intersection(test_trials["image_id"])
    if image_overlap:
        raise ValueError(f"Official train/test image leakage detected: {sorted(image_overlap)[:5]}")

    session = int(heldout_session)
    decoder = train_trials.loc[train_trials["session"].ne(session)]
    calibration = train_trials.loc[train_trials["session"].eq(session)].sort_values(
        "trial_order_index", kind="stable"
    )
    evaluation = test_trials.loc[test_trials["session"].eq(session)].sort_values(
        "trial_order_index", kind="stable"
    )
    if decoder.empty or calibration.empty or evaluation.empty:
        raise ValueError(f"Held-out session {session} does not define all three data roles")

    decoder_indices = decoder["array_row"].to_numpy(dtype=np.int64)
    calibration_indices = calibration["array_row"].to_numpy(dtype=np.int64)
    evaluation_indices = evaluation["array_row"].to_numpy(dtype=np.int64)
    budgets = sorted(set(int(value) for value in calibration_budgets))
    if not budgets or budgets[0] < 0 or budgets[-1] > len(calibration_indices):
        raise ValueError(
            f"Calibration budgets {budgets} are invalid for {len(calibration_indices)} trials"
        )
    prefixes = {budget: calibration_indices[:budget].copy() for budget in budgets}
    repeat_groups = {
        str(key): group["array_row"].to_numpy(dtype=np.int64)
        for key, group in evaluation.groupby("image_id", sort=True)
    }

    return AlljoinedLOSOFold(
        heldout_session=session,
        decoder_train_indices=decoder_indices,
        calibration_pool_indices=calibration_indices,
        evaluation_indices=evaluation_indices,
        calibration_prefixes=prefixes,
        evaluation_repeat_groups=repeat_groups,
    )
