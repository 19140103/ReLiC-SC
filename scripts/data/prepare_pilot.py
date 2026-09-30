"""Convert an audited pilot subject into model-ready memory-mapped arrays."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "Alljoined-1.6M" / "preprocessed_eeg"
PROCESSED_ROOT = ROOT / "data" / "processed" / "alljoined_1_6m"


def aligned_times(payload: dict[str, Any], n_timepoints: int) -> np.ndarray:
    times = np.asarray(payload["times"])
    if times.shape[0] == n_timepoints:
        return times

    nonnegative = times[times >= 0]
    if nonnegative.shape[0] >= n_timepoints:
        return nonnegative[:n_timepoints]

    if times.shape[0] > n_timepoints:
        return times[:n_timepoints]

    raise ValueError(
        f"Cannot align time axis of length {times.shape[0]} to EEG data with "
        f"{n_timepoints} timepoints."
    )


def convert_split(
    source: Path,
    destination: Path,
    dtype: np.dtype,
    chunk_trials: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    data = np.asarray(payload["preprocessed_eeg_data"])
    temporary = destination.with_name(destination.stem + ".part.npy")

    if destination.exists():
        if temporary.exists():
            raise FileExistsError(
                f"Both final and temporary processed outputs exist for {destination}; "
                f"refusing to guess which one to keep."
            )
        existing = np.load(destination, mmap_mode="r")
        if existing.shape != data.shape or existing.dtype != dtype:
            raise ValueError(
                f"Existing processed output does not match expected shape/dtype: "
                f"{destination} has {existing.shape}/{existing.dtype}, expected {data.shape}/{dtype}"
            )
        summary = {
            "source": str(source.relative_to(ROOT)),
            "destination": str(destination.relative_to(ROOT)),
            "shape": list(existing.shape),
            "source_dtype": str(data.dtype),
            "destination_dtype": str(existing.dtype),
            "source_bytes": source.stat().st_size,
            "destination_bytes": destination.stat().st_size,
            "max_absolute_cast_error": None,
            "mean_absolute_cast_error": None,
            "reused_existing": True,
        }
        shared = {
            "channels": list(payload["ch_names"]),
            "times": aligned_times(payload, data.shape[-1]),
            "configs": payload.get("configs", {}),
        }
        del existing, data, payload
        gc.collect()
        return summary, shared

    if temporary.exists():
        existing = np.load(temporary, mmap_mode="r")
        if existing.shape != data.shape or existing.dtype != dtype:
            raise ValueError(
                f"Temporary processed output is not safely reusable: "
                f"{temporary} has {existing.shape}/{existing.dtype}, expected {data.shape}/{dtype}"
            )
        del existing
        temporary.replace(destination)
        summary = {
            "source": str(source.relative_to(ROOT)),
            "destination": str(destination.relative_to(ROOT)),
            "shape": list(data.shape),
            "source_dtype": str(data.dtype),
            "destination_dtype": str(dtype),
            "source_bytes": source.stat().st_size,
            "destination_bytes": destination.stat().st_size,
            "max_absolute_cast_error": None,
            "mean_absolute_cast_error": None,
            "resumed_temporary": True,
        }
        shared = {
            "channels": list(payload["ch_names"]),
            "times": aligned_times(payload, data.shape[-1]),
            "configs": payload.get("configs", {}),
        }
        del data, payload
        gc.collect()
        return summary, shared

    mapped = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=dtype, shape=data.shape
    )
    max_abs_error = 0.0
    sum_abs_error = 0.0
    value_count = 0
    for start in range(0, len(data), chunk_trials):
        stop = min(start + chunk_trials, len(data))
        converted = data[start:stop].astype(dtype, copy=False)
        mapped[start:stop] = converted
        error = np.abs(data[start:stop] - converted.astype(data.dtype, copy=False))
        max_abs_error = max(max_abs_error, float(error.max(initial=0.0)))
        sum_abs_error += float(error.sum(dtype=np.float64))
        value_count += int(error.size)
    mapped.flush()
    del mapped
    temporary.replace(destination)

    summary = {
        "source": str(source.relative_to(ROOT)),
        "destination": str(destination.relative_to(ROOT)),
        "shape": list(data.shape),
        "source_dtype": str(data.dtype),
        "destination_dtype": str(dtype),
        "source_bytes": source.stat().st_size,
        "destination_bytes": destination.stat().st_size,
        "max_absolute_cast_error": max_abs_error,
        "mean_absolute_cast_error": sum_abs_error / value_count,
    }
    shared = {
        "channels": list(payload["ch_names"]),
        "times": aligned_times(payload, data.shape[-1]),
        "configs": payload.get("configs", {}),
    }
    del data, payload
    gc.collect()
    return summary, shared


def prepare_metadata(source: Path, destination_dir: Path) -> dict[str, Any]:
    frame = pd.read_parquet(source)
    results: dict[str, Any] = {}
    for short_name, partition in [("train", "stim_train"), ("test", "stim_test")]:
        selected = frame.loc[
            (frame["partition"] == partition) & ~frame["dropped"].astype(bool)
        ].copy()
        selected.insert(0, "source_row_index", selected.index.astype("int64"))
        selected.reset_index(drop=True, inplace=True)
        selected["repetition_id"] = (
            selected.groupby("image_path", sort=False).cumcount().astype("int32")
        )
        selected["session_repetition_id"] = (
            selected.groupby(["image_path", "session"], sort=False)
            .cumcount()
            .astype("int16")
        )
        output = destination_dir / f"{short_name}_metadata.parquet"
        selected.to_parquet(output, index=False)
        results[short_name] = {
            "path": str(output.relative_to(ROOT)),
            "rows": int(len(selected)),
            "columns": list(selected.columns),
            "unique_images": int(selected["image_path"].nunique()),
            "file_bytes": output.stat().st_size,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="sub-09")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--chunk-trials", type=int, default=256)
    args = parser.parse_args()

    destination_dir = PROCESSED_ROOT / args.subject
    destination_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(args.dtype)
    split_sources = {
        "test": RAW_ROOT / args.subject / "preprocessed_eeg_test_flat.npy",
        "train": RAW_ROOT / args.subject / "preprocessed_eeg_training_flat.npy",
    }

    split_summaries: dict[str, Any] = {}
    shared_reference: dict[str, Any] | None = None
    for split in ["test", "train"]:
        summary, shared = convert_split(
            split_sources[split],
            destination_dir / f"{split}_eeg.npy",
            dtype,
            args.chunk_trials,
        )
        if shared_reference is None:
            shared_reference = shared
        else:
            if shared["channels"] != shared_reference["channels"] or not np.array_equal(
                shared["times"], shared_reference["times"]
            ):
                raise RuntimeError("Train/test channel order or time axis differs")
        split_summaries[split] = summary

    assert shared_reference is not None
    np.save(destination_dir / "times.npy", shared_reference["times"], allow_pickle=False)
    (destination_dir / "channels.json").write_text(
        json.dumps(shared_reference["channels"], indent=2), encoding="utf-8"
    )
    (destination_dir / "preprocessing_config.json").write_text(
        json.dumps(shared_reference["configs"], indent=2), encoding="utf-8"
    )
    metadata = prepare_metadata(
        RAW_ROOT / args.subject / "experiment_metadata_categories.parquet",
        destination_dir,
    )

    manifest_path = ROOT / "data" / "metadata" / "alljoined_hf_manifest.json"
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "subject": args.subject,
        "source_repo": source_manifest.get("repo_id"),
        "source_revision": source_manifest.get("resolved_commit"),
        "dtype": str(dtype),
        "arrays": split_summaries,
        "metadata": metadata,
        "channels": shared_reference["channels"],
        "times_path": str((destination_dir / "times.npy").relative_to(ROOT)),
        "metadata_alignment": {
            split: "PASS"
            if metadata[split]["rows"] == split_summaries[split]["shape"][0]
            else "FAIL"
            for split in ["train", "test"]
        },
    }
    report_path = destination_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if set(report["metadata_alignment"].values()) == {"PASS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
