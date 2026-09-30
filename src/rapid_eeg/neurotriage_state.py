"""De-identification helpers for NeuroTriage session-state metadata."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd


SUBJECT_COLUMN = "AJ 1.6M subject #"
TIMESTAMP_COLUMN = "Timestamp"
DATE_COLUMN = "Date"
CAFFEINE_COLUMN = (
    "Did you consume caffeine (coffee, tea, energy drinks) within the last "
    "24 hours?"
)
SLEEP_COLUMN = "How many hours of sleep did you get last night?"
RESTED_COLUMN = "Did you feel rested upon waking?"
FATIGUED_COLUMN = "Are you currently feeling tired or fatigued?"
EATEN_COLUMN = "Have you eaten in the past 4 hours?"

STATE_COLUMNS = [
    "subject",
    "subject_id",
    "session",
    "sleep_hours",
    "rested",
    "fatigued",
    "caffeine_24h",
    "eaten_4h",
    "state_complete",
]


def parse_yes_no(value: Any) -> float:
    """Convert a questionnaire yes/no response to 1/0 or NaN."""
    if pd.isna(value):
        return np.nan
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return 1.0
    if normalized in {"no", "n", "false", "0"}:
        return 0.0
    return np.nan


def parse_sleep_hours(value: Any) -> float:
    """Parse the small set of free-text sleep-hour formats in the source."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        return result if 0.0 <= result <= 24.0 else np.nan

    normalized = str(value).strip().lower()
    if not normalized or normalized in {"yes", "i don't know", "unknown"}:
        return np.nan
    word_numbers = {
        "zero": 0.0,
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
        "six": 6.0,
        "seven": 7.0,
        "eight": 8.0,
        "nine": 9.0,
        "ten": 10.0,
    }
    if normalized in word_numbers:
        return word_numbers[normalized]

    fraction_match = re.search(r"(\d+(?:\.\d+)?)\s+1/2", normalized)
    if fraction_match:
        result = float(fraction_match.group(1)) + 0.5
        return result if result <= 24.0 else np.nan

    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", normalized)]
    if not numbers:
        return np.nan
    result = sum(numbers[:2]) / min(len(numbers), 2)
    return result if 0.0 <= result <= 24.0 and math.isfinite(result) else np.nan


def build_session_state(source: pd.DataFrame) -> pd.DataFrame:
    """Return a strictly de-identified subject/session state table.

    Session numbers are assigned by questionnaire date and submission timestamp
    within subject. This handles the one participant with two sessions on the
    same calendar date without retaining either date in the output.
    """
    required = {
        SUBJECT_COLUMN,
        TIMESTAMP_COLUMN,
        DATE_COLUMN,
        CAFFEINE_COLUMN,
        SLEEP_COLUMN,
        RESTED_COLUMN,
        FATIGUED_COLUMN,
        EATEN_COLUMN,
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing questionnaire columns: {sorted(missing)}")

    work = source.loc[:, sorted(required)].copy()
    work["_source_order"] = np.arange(len(work))
    work["subject"] = pd.to_numeric(work[SUBJECT_COLUMN], errors="raise").astype(int)
    work["_date"] = pd.to_datetime(work[DATE_COLUMN], errors="coerce")
    work["_timestamp"] = pd.to_datetime(work[TIMESTAMP_COLUMN], errors="coerce")
    work = work.sort_values(
        ["subject", "_date", "_timestamp", "_source_order"],
        kind="stable",
        na_position="last",
    )
    work["session"] = work.groupby("subject", sort=False).cumcount() + 1

    counts = work.groupby("subject").size()
    if not counts.eq(4).all():
        raise ValueError(f"Expected four questionnaire rows per subject: {counts.to_dict()}")

    output = pd.DataFrame(
        {
            "subject": work["subject"].astype("int8"),
            "subject_id": work["subject"].map(lambda value: f"sub-{value:02d}"),
            "session": work["session"].astype("int8"),
            "sleep_hours": work[SLEEP_COLUMN].map(parse_sleep_hours).astype("Float32"),
            "rested": work[RESTED_COLUMN].map(parse_yes_no).astype("Int8"),
            "fatigued": work[FATIGUED_COLUMN].map(parse_yes_no).astype("Int8"),
            "caffeine_24h": work[CAFFEINE_COLUMN].map(parse_yes_no).astype("Int8"),
            "eaten_4h": work[EATEN_COLUMN].map(parse_yes_no).astype("Int8"),
        }
    )
    targets = ["sleep_hours", "rested", "fatigued", "caffeine_24h", "eaten_4h"]
    output["state_complete"] = output[targets].notna().all(axis=1)
    output = output.sort_values(["subject", "session"], kind="stable").reset_index(drop=True)
    if list(output.columns) != STATE_COLUMNS:
        raise AssertionError("Unexpected output schema; de-identification guard failed")
    return output


def state_audit(state: pd.DataFrame) -> dict[str, Any]:
    """Summarize usable state coverage without exposing participant responses."""
    binary_targets = ["rested", "fatigued", "caffeine_24h", "eaten_4h"]
    targets = ["sleep_hours", *binary_targets]
    variation = {}
    for column in targets:
        unique_per_subject = state.groupby("subject")[column].nunique(dropna=True)
        variation[column] = int(unique_per_subject.gt(1).sum())
    return {
        "rows": int(len(state)),
        "subjects": int(state["subject"].nunique()),
        "sessions_per_subject": {
            str(int(key)): int(value)
            for key, value in state.groupby("subject")["session"].nunique().items()
        },
        "complete_binary_state_rows": int(
            state[binary_targets].notna().all(axis=1).sum()
        ),
        "complete_state_rows": int(state["state_complete"].sum()),
        "available_by_field": {
            column: int(state[column].notna().sum()) for column in targets
        },
        "missing_by_field": {
            column: int(state[column].isna().sum()) for column in targets
        },
        "subjects_with_within_subject_variation": variation,
        "output_columns": list(state.columns),
        "direct_identifiers_retained": False,
    }
