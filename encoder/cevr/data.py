from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import write_json


def load_visual_representations(feature_dirs: list[str | Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_arrays = []
    test_arrays = []
    labels = []
    for feature_dir in feature_dirs:
        feature_path = Path(feature_dir)
        train_path = feature_path / "image_train.npy"
        test_path = feature_path / "image_test.npy"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing image_train.npy/image_test.npy in {feature_path}")
        train_arrays.append(np.load(train_path, mmap_mode="r"))
        test_arrays.append(np.load(test_path, mmap_mode="r"))
        labels.append(feature_path.name)

    train_shapes = {tuple(array.shape) for array in train_arrays}
    test_shapes = {tuple(array.shape) for array in test_arrays}
    if len(train_shapes) != 1 or len(test_shapes) != 1:
        raise ValueError(f"Inconsistent visual feature shapes: train={train_shapes}, test={test_shapes}")

    image_train = np.stack([np.asarray(array, dtype=np.float32) for array in train_arrays], axis=0)
    image_test = np.stack([np.asarray(array, dtype=np.float32) for array in test_arrays], axis=0)
    return image_train, image_test, labels


def load_things_eeg(
    eeg_dir: str | Path,
    subject: str,
    selected_channels: list[str],
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eeg_dir = Path(eeg_dir)
    subject_id = int(subject.replace("sub-", ""))
    info_path = eeg_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing THINGS-EEG info file: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    channel_indices = [info["ch_names"].index(channel) for channel in selected_channels]

    subject_dir = eeg_dir / f"sub-{subject_id:02d}"
    train_path = subject_dir / "train.npy"
    test_path = subject_dir / "test.npy"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train.npy/test.npy for {subject}: {subject_dir}")

    def average_selected_channels(eeg: np.ndarray, chunk_size: int = 64) -> np.ndarray:
        if eeg.ndim != 5:
            raise ValueError(
                "Expected THINGS-EEG shape "
                "(objects, images, repeats, channels, time), "
                f"got {eeg.shape}."
            )
        averaged = np.empty(
            (eeg.shape[0], eeg.shape[1], len(channel_indices), eeg.shape[-1]),
            dtype=np.float32,
        )
        for start in range(0, eeg.shape[0], chunk_size):
            end = min(start + chunk_size, eeg.shape[0])
            selected = np.take(eeg[start:end], channel_indices, axis=3)
            averaged[start:end] = selected.mean(axis=2, dtype=np.float32)
        return averaged

    cache_key = hashlib.sha1("\0".join(selected_channels).encode("utf-8")).hexdigest()[:12]
    cache_dir = subject_dir / "cevr_averaged_cache"
    cache_train_path = cache_dir / f"train_{cache_key}.npy"
    cache_test_path = cache_dir / f"test_{cache_key}.npy"
    cache_meta_path = cache_dir / f"meta_{cache_key}.json"
    expected_cache_meta = {
        "selected_channels": list(selected_channels),
        "train_size": train_path.stat().st_size,
        "train_mtime_ns": train_path.stat().st_mtime_ns,
        "test_size": test_path.stat().st_size,
        "test_mtime_ns": test_path.stat().st_mtime_ns,
    }

    cache_valid = False
    if cache:
        cache_valid = cache_train_path.exists() and cache_test_path.exists() and cache_meta_path.exists()
        if cache_valid:
            try:
                cache_valid = json.loads(cache_meta_path.read_text(encoding="utf-8")) == expected_cache_meta
            except (OSError, ValueError, json.JSONDecodeError):
                cache_valid = False

    if cache_valid:
        train_avg = np.load(cache_train_path, mmap_mode="r")
        test_avg = np.load(cache_test_path, mmap_mode="r")
    else:
        train = np.load(train_path, mmap_mode="r")
        test = np.load(test_path, mmap_mode="r")
        train_avg = average_selected_channels(train)
        test_avg = average_selected_channels(test)
        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            train_tmp = cache_dir / f"train_{cache_key}.tmp.npy"
            test_tmp = cache_dir / f"test_{cache_key}.tmp.npy"
            np.save(train_tmp, train_avg)
            np.save(test_tmp, test_avg)
            train_tmp.replace(cache_train_path)
            test_tmp.replace(cache_test_path)
            write_json(cache_meta_path, expected_cache_meta)

    train_keys = np.array(
        [
            (object_idx, image_idx)
            for object_idx in range(train_avg.shape[0])
            for image_idx in range(train_avg.shape[1])
        ],
        dtype=np.int64,
    )
    test_keys = np.array(
        [
            (object_idx, image_idx)
            for object_idx in range(test_avg.shape[0])
            for image_idx in range(test_avg.shape[1])
        ],
        dtype=np.int64,
    )
    train_eeg = train_avg.reshape(-1, train_avg.shape[-2], train_avg.shape[-1])
    test_eeg = test_avg.reshape(-1, test_avg.shape[-2], test_avg.shape[-1])
    return train_eeg, train_keys, test_eeg, test_keys


def visual_targets_from_keys(image_features: np.ndarray, keys: np.ndarray) -> np.ndarray:
    targets = np.empty((keys.shape[0], image_features.shape[0], image_features.shape[-1]), dtype=np.float32)
    for idx, (object_idx, image_idx) in enumerate(keys):
        targets[idx] = image_features[:, int(object_idx), int(image_idx), :]
    return targets


def standardize_by_train(
    train_eeg: np.ndarray,
    test_eeg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_eeg.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(train_eeg.std(axis=(0, 2), keepdims=True), 1e-6)
    return (
        ((train_eeg - mean) / std).astype(np.float32, copy=False),
        ((test_eeg - mean) / std).astype(np.float32, copy=False),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


class PairDataset(Dataset):
    def __init__(self, eeg: np.ndarray, visual_features: np.ndarray, keys: np.ndarray):
        if eeg.shape[0] != visual_features.shape[0]:
            raise ValueError(f"EEG/visual size mismatch: {eeg.shape[0]} vs {visual_features.shape[0]}")
        self.eeg = eeg
        self.visual_features = visual_features
        self.keys = keys

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        object_idx, image_idx = self.keys[index]
        group = int(object_idx) * 1000 + int(image_idx)
        return (
            torch.from_numpy(self.eeg[index].astype(np.float32, copy=False)),
            torch.from_numpy(self.visual_features[index].astype(np.float32, copy=False)),
            torch.tensor(group, dtype=torch.long),
        )
