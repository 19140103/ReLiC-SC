"""Bridge processed Alljoined arrays and a CLIP depth cache to encoder inputs.

The bridge preserves the official train/test image split and stores the available
repeat mean in one repeat slot. It is for the paper's image-level averaged-EEG
protocol, not single-trial or leave-one-session-out training. No EEG or visual
features are downloaded. All paths are supplied by the caller.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd


VISUAL_LAYERS = (1, 2, 4, 5, 7, 8, 10, 12)
VISUAL_BANK_NAME = "ViT-B32-depth-bank-8"


def image_id(value: object) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _read_split(processed_subject: Path, split: str) -> dict[str, Any]:
    eeg = np.load(processed_subject / f"{split}_eeg.npy", mmap_mode="r", allow_pickle=False)
    frame = pd.read_parquet(processed_subject / f"{split}_metadata.parquet")
    required = {"category_num", "category_img_num", "image_path", "source_row_index", "partition"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{split} metadata is missing columns: {missing}")
    if eeg.ndim != 3 or min(eeg.shape) <= 0 or len(frame) != eeg.shape[0]:
        raise ValueError(f"{split} EEG must be nonempty [trials, channels, time] and row-aligned with metadata")
    if not np.issubdtype(eeg.dtype, np.floating):
        raise ValueError(f"{split} EEG must have floating-point values")
    if frame["source_row_index"].isna().any() or frame["source_row_index"].duplicated().any():
        raise ValueError(f"{split} source_row_index must be unique and non-null")
    if not frame["source_row_index"].is_monotonic_increasing:
        raise ValueError(f"{split} metadata is not in the original EEG row order")
    if "array_row" in frame and not np.array_equal(frame["array_row"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"{split} array_row does not match EEG row order")
    if set(frame["partition"]) != {f"stim_{split}"}:
        raise ValueError(f"{split} metadata does not preserve the official stim_{split} partition")
    if frame["image_path"].isna().any():
        raise ValueError(f"{split} image paths must be non-null")
    frame = frame.copy()
    frame["image_id"] = frame["image_path"].map(image_id)
    if frame["image_id"].eq("").any():
        raise ValueError(f"{split} image identifiers must be nonempty")
    for column in ("category_num", "category_img_num"):
        values = frame[column].to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
            raise ValueError(f"{split} {column} must contain nonnegative integer indices")
    grouped = frame.groupby(["category_num", "category_img_num"], sort=True)
    if grouped["image_id"].nunique().ne(1).any():
        raise ValueError(f"{split} category/image key maps to multiple image identifiers")
    keys = grouped["image_id"].first().reset_index()
    if keys["image_id"].duplicated().any():
        raise ValueError(f"{split} image identifier maps to multiple category/image keys")
    objects = int(frame["category_num"].max()) + 1
    images = int(frame["category_img_num"].max()) + 1
    if len(keys) != objects * images:
        raise ValueError(f"{split} category/image keys must form a complete zero-based rectangular grid")
    keys["repeat_count"] = grouped.size().to_numpy(dtype=np.int64)
    return {"eeg": eeg, "frame": frame, "keys": keys, "groups": grouped.indices,
            "objects": objects, "images": images}


def _read_visual_cache(image_cache: Path) -> tuple[np.ndarray, dict[str, int]]:
    embeddings = np.load(image_cache / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 3 or embeddings.shape[1] != len(VISUAL_LAYERS) or min(embeddings.shape) <= 0:
        raise ValueError("Visual cache must have shape [images, 8 levels, feature_dim]")
    report = json.loads((image_cache / "cache_report.json").read_text(encoding="utf-8"))
    if tuple(report.get("visual_layers", [])) != VISUAL_LAYERS:
        raise ValueError(f"cache_report.json must identify visual_layers as {list(VISUAL_LAYERS)} in this order")
    if report.get("normalized", False) is not False:
        raise ValueError("Alljoined visual cache must contain unnormalized intermediate features")
    if "model" in report and report["model"] != "ViT-B-32-quickgelu":
        raise ValueError("Alljoined visual cache must use the ViT-B-32-quickgelu backbone")
    index = pd.read_parquet(image_cache / "image_index.parquet")
    if not {"image_id", "embedding_row"}.issubset(index.columns):
        raise ValueError("Visual cache index needs image_id and embedding_row columns")
    if index[["image_id", "embedding_row"]].isna().any().any():
        raise ValueError("Visual cache identifiers and rows must be non-null")
    ids = index["image_id"].map(image_id)
    rows = index["embedding_row"].to_numpy()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Visual cache image identifiers must be unique and nonempty")
    if not np.issubdtype(rows.dtype, np.integer) or (rows < 0).any() or (rows >= embeddings.shape[0]).any():
        raise ValueError("Visual cache embedding_row is outside the embedding array")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Visual cache embedding_row must be unique")
    return embeddings, dict(zip(ids, rows.astype(int)))


def _atomic_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp.npy", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_encoder_data(processed_subject: Path, image_cache: Path, output_root: Path,
                         subject: str) -> dict[str, Any]:
    """Create encoder inputs; reject ambiguous identity and conflicting outputs."""
    if not re.fullmatch(r"sub-\d{2}", subject):
        raise ValueError("subject must have the form sub-XX")
    processed_subject, image_cache, output_root = map(Path, (processed_subject, image_cache, output_root))
    splits = {name: _read_split(processed_subject, name) for name in ("train", "test")}
    overlap = set(splits["train"]["keys"]["image_id"]) & set(splits["test"]["keys"]["image_id"])
    if overlap:
        raise ValueError("Train/test image identifiers overlap; refusing image leakage")
    for split in splits.values():
        if "subject" in split["frame"]:
            observed = split["frame"]["subject"].astype(str).str.replace("sub-", "", regex=False)
            try:
                matching_subject = set(observed.astype(int)) == {int(subject[4:])}
            except ValueError:
                matching_subject = False
            if not matching_subject:
                raise ValueError("Requested subject differs from metadata subject")
    channels = json.loads((processed_subject / "channels.json").read_text(encoding="utf-8"))
    times = np.load(processed_subject / "times.npy", allow_pickle=False)
    if not isinstance(channels, list) or any(not isinstance(c, str) or not c for c in channels) or len(set(channels)) != len(channels):
        raise ValueError("channels.json must contain unique nonempty channel names")
    if times.ndim != 1 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("times.npy must contain a finite, strictly increasing time axis")
    if any(s["eeg"].shape[1:] != (len(channels), len(times)) for s in splits.values()):
        raise ValueError("Train/test EEG shapes disagree with channel names or time axis")
    embeddings, lookup = _read_visual_cache(image_cache)
    for split in splits.values():
        missing = set(split["keys"]["image_id"]) - lookup.keys()
        if missing:
            raise ValueError(f"{len(missing)} metadata image identifiers are missing from the visual cache")

    eeg_root = output_root / "preprocessed_eeg_npy"
    subject_root = eeg_root / subject
    report_path = output_root / "reports" / f"{subject}_encoder_data.json"
    output_files = [subject_root / f"{split}{suffix}" for split in splits
                    for suffix in (".npy", "_repeat_counts.npy", "_image_keys.csv")]
    if any(p.exists() for p in [report_path, *output_files]):
        raise FileExistsError("Subject output already exists; use a new output root instead of overwriting data")
    info = {"ch_names": channels, "times": times.astype(float).tolist(),
            "source": "Alljoined-1.6M available-repeat means"}
    info_path = eeg_root / "info.json"
    if info_path.exists():
        existing = json.loads(info_path.read_text(encoding="utf-8"))
        if existing.get("ch_names") != channels or existing.get("times") != info["times"]:
            raise ValueError("Existing shared info.json has different channels or times")

    # Validate all requested EEG groups and shared banks before creating outputs.
    # A bank is only reused if every feature and image key matches, not merely its shape.
    visual_root = output_root / "image_feature" / VISUAL_BANK_NAME
    for name, split in splits.items():
        for indices in split["groups"].values():
            if not np.isfinite(split["eeg"][indices]).all():
                raise ValueError(f"{name} EEG contains non-finite values")
        rows = np.asarray([lookup[key] for key in split["keys"]["image_id"]], dtype=np.int64)
        split["visual_rows"] = rows
        key_path = visual_root / f"{name}_image_keys.csv"
        key_csv = split["keys"][["category_num", "category_img_num", "image_id"]].to_csv(index=False, lineterminator="\n")
        split["visual_key_csv"] = key_csv
        if key_path.exists() and key_path.read_text(encoding="utf-8") != key_csv:
            raise ValueError(f"Existing {name} visual bank image keys differ across subjects")
        for level in range(len(VISUAL_LAYERS)):
            features = np.asarray(embeddings[rows, level], dtype=np.float32).reshape(split["objects"], split["images"], -1)
            if not np.isfinite(features).all():
                raise ValueError(f"{name} visual features contain non-finite values")
            path = visual_root / f"level-{100 + level:03d}" / f"image_{name}.npy"
            if path.exists():
                if not key_path.exists():
                    raise ValueError("Existing visual bank lacks an image-key manifest; use a new output root")
                if not np.array_equal(np.load(path, mmap_mode="r", allow_pickle=False), features):
                    raise ValueError(f"Existing {name} visual bank differs from the requested cache")

    if not info_path.exists():
        _atomic_text(info_path, json.dumps(info, indent=2))
    summary: dict[str, Any] = {"subject": subject, "protocol": "available-repeat mean; official image split",
                              "visual_layers": list(VISUAL_LAYERS), "channels": channels, "splits": {}}
    for name, split in splits.items():
        shape = (split["objects"], split["images"], 1, len(channels), len(times))
        subject_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=subject_root, suffix=".tmp.npy", delete=False) as handle:
            temporary = Path(handle.name)
        out = None
        try:
            out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
            counts = np.zeros(shape[:2], dtype=np.int64)
            for key, indices in split["groups"].items():
                # Float32 accumulation matches the original available-repeat bridge.
                out[key[0], key[1], 0] = np.asarray(split["eeg"][indices], dtype=np.float32).mean(axis=0, dtype=np.float32)
                counts[key] = len(indices)
            out.flush()
            del out
            out = None
            temporary.replace(subject_root / f"{name}.npy")
        finally:
            if out is not None:
                del out
            temporary.unlink(missing_ok=True)
        _atomic_array(subject_root / f"{name}_repeat_counts.npy", counts)
        _atomic_text(subject_root / f"{name}_image_keys.csv", split["keys"].to_csv(index=False, lineterminator="\n"))
        for level in range(len(VISUAL_LAYERS)):
            path = visual_root / f"level-{100 + level:03d}" / f"image_{name}.npy"
            if not path.exists():
                features = np.asarray(embeddings[split["visual_rows"], level], dtype=np.float32)
                _atomic_array(path, features.reshape(shape[0], shape[1], -1))
        _atomic_text(visual_root / f"{name}_image_keys.csv", split["visual_key_csv"])
        summary["splits"][name] = {"shape": list(shape), "trials": len(split["frame"]),
                                    "images": len(split["keys"]), "repeat_min": int(counts.min()),
                                    "repeat_max": int(counts.max())}
    summary["encoder_inputs"] = {
        "eeg_data_dir": "preprocessed_eeg_npy",
        "visual_feature_root": f"image_feature/{VISUAL_BANK_NAME}",
        "visual_levels": [f"level-{100 + level:03d}" for level in range(len(VISUAL_LAYERS))],
    }
    _atomic_text(report_path, json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-subject", type=Path, required=True,
                        help="Directory containing prepare_pilot train/test arrays, metadata, channels and times")
    parser.add_argument("--image-cache", type=Path, required=True,
                        help="Directory containing embeddings.npy, image_index.parquet and cache_report.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    report = prepare_encoder_data(args.processed_subject, args.image_cache, args.output_root, args.subject)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
