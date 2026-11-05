"""Tiny synthetic checks for the Alljoined image-mean encoder bridge."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "encoder"))
spec = importlib.util.spec_from_file_location("prepare_encoder_data", ROOT / "scripts/data/prepare_encoder_data.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

from cevr.data import load_things_eeg, load_visual_representations, visual_targets_from_keys


def _fixture(tmp_path: Path):
    processed, cache, out = (tmp_path / name for name in ("processed", "cache", "prepared"))
    processed.mkdir()
    cache.mkdir()
    (processed / "channels.json").write_text(json.dumps(["Cz", "Pz"]))
    np.save(processed / "times.npy", np.array([0.0, 0.004, 0.008]))
    # Array order intentionally differs from category order, and repeats are unequal.
    frames = {}
    eeg_arrays = {}
    for split, prefix in (("train", "a"), ("test", "b")):
        frames[split] = pd.DataFrame({
            "source_row_index": [10, 12, 15], "array_row": [0, 1, 2],
            "subject": [1, 1, 1], "partition": [f"stim_{split}"] * 3,
            "category_num": [1, 0, 1], "category_img_num": [0, 0, 0],
            "image_path": [f"folder/{prefix}1.jpg", f"folder/{prefix}0.jpg", f"folder/{prefix}1.jpg"],
        })
        frames[split].to_parquet(processed / f"{split}_metadata.parquet", index=False)
        eeg_arrays[split] = np.arange(18, dtype=np.float32).reshape(3, 2, 3) + (0 if split == "train" else 100)
        np.save(processed / f"{split}_eeg.npy", eeg_arrays[split])
    embeddings = np.arange(4 * 8 * 5, dtype=np.float32).reshape(4, 8, 5)
    np.save(cache / "embeddings.npy", embeddings)
    # Cache row order also differs from both trial and category order.
    pd.DataFrame({"image_id": ["b1.jpg", "a0.jpg", "b0.jpg", "a1.jpg"],
                  "embedding_row": [0, 1, 2, 3]}).to_parquet(cache / "image_index.parquet", index=False)
    (cache / "cache_report.json").write_text(json.dumps({"visual_layers": list(bridge.VISUAL_LAYERS)}))
    return processed, cache, out, frames, eeg_arrays, embeddings


def test_bridge_means_keys_visual_alignment_and_encoder_load(tmp_path):
    processed, cache, out, _, eeg, embeddings = _fixture(tmp_path)
    report = bridge.prepare_encoder_data(processed, cache, out, "sub-01")
    assert report["splits"]["train"]["shape"] == [2, 1, 1, 2, 3]
    assert report["splits"]["train"]["repeat_min"] == 1
    assert report["splits"]["train"]["repeat_max"] == 2
    train, train_keys, test, test_keys = load_things_eeg(out / "preprocessed_eeg_npy", "sub-01", ["Pz", "Cz"], cache=False)
    np.testing.assert_array_equal(train_keys, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(test_keys, train_keys)
    np.testing.assert_array_equal(train[0], eeg["train"][1, [1, 0]])
    np.testing.assert_array_equal(train[1], eeg["train"][[0, 2]].mean(0)[[1, 0]])
    np.testing.assert_array_equal(test[1], eeg["test"][[0, 2]].mean(0)[[1, 0]])
    np.testing.assert_array_equal(np.load(out / "preprocessed_eeg_npy/sub-01/train_repeat_counts.npy"), [[1], [2]])
    keys = pd.read_csv(out / "preprocessed_eeg_npy/sub-01/train_image_keys.csv")
    assert keys["image_id"].tolist() == ["a0.jpg", "a1.jpg"]
    visual_root = out / "image_feature" / bridge.VISUAL_BANK_NAME
    levels = [visual_root / f"level-{100 + n:03d}" for n in range(8)]
    visual_train, visual_test, _ = load_visual_representations(levels)
    np.testing.assert_array_equal(visual_targets_from_keys(visual_train, train_keys), embeddings[[1, 3]])
    np.testing.assert_array_equal(visual_targets_from_keys(visual_test, test_keys), embeddings[[2, 0]])
    with pytest.raises(FileExistsError, match="already exists"):
        bridge.prepare_encoder_data(processed, cache, out, "sub-01")


@pytest.mark.parametrize("failure", ["overlap", "shuffled_rows", "missing_cell", "ambiguous_image", "wrong_subject", "nonfinite"])
def test_bridge_rejects_misalignment_before_outputs(tmp_path, failure):
    processed, cache, out, frames, eeg, _ = _fixture(tmp_path)
    split = "train"
    if failure == "overlap":
        split = "test"
        frames[split].loc[1, "image_path"] = "folder/a0.jpg"
    elif failure == "shuffled_rows":
        frames[split] = frames[split].iloc[[1, 0, 2]]
    elif failure == "missing_cell":
        frames[split].loc[frames[split]["category_num"].eq(1), "category_num"] = 2
    elif failure == "ambiguous_image":
        frames[split].loc[2, "image_path"] = "folder/a0.jpg"
    elif failure == "wrong_subject":
        frames[split]["subject"] = 2
    elif failure == "nonfinite":
        eeg[split][0, 0, 0] = np.nan
        np.save(processed / f"{split}_eeg.npy", eeg[split])
    frames[split].to_parquet(processed / f"{split}_metadata.parquet", index=False)
    with pytest.raises(ValueError):
        bridge.prepare_encoder_data(processed, cache, out, "sub-01")
    assert not out.exists()


def test_bridge_requires_correct_visual_layers_and_unique_cache_ids(tmp_path):
    processed, cache, out, _, _, _ = _fixture(tmp_path)
    (cache / "cache_report.json").write_text(json.dumps({"visual_layers": list(range(8))}))
    with pytest.raises(ValueError, match="visual_layers"):
        bridge.prepare_encoder_data(processed, cache, out, "sub-01")
    (cache / "cache_report.json").write_text(json.dumps({"visual_layers": list(bridge.VISUAL_LAYERS)}))
    index = pd.read_parquet(cache / "image_index.parquet")
    index.loc[0, "image_id"] = "a0.jpg"
    index.to_parquet(cache / "image_index.parquet", index=False)
    with pytest.raises(ValueError, match="unique"):
        bridge.prepare_encoder_data(processed, cache, out, "sub-01")
    assert not out.exists()


def test_bridge_shared_bank_reuse_checks_content_and_metadata(tmp_path):
    processed, cache, out, frames, _, embeddings = _fixture(tmp_path)
    bridge.prepare_encoder_data(processed, cache, out, "sub-01")
    for split in frames:
        frames[split]["subject"] = 2
        frames[split].to_parquet(processed / f"{split}_metadata.parquet", index=False)
    # Same-shaped changed data must not be silently accepted as a shared bank.
    changed = embeddings.copy()
    changed[1, 0, 0] += 1
    np.save(cache / "embeddings.npy", changed)
    with pytest.raises(ValueError, match="differs"):
        bridge.prepare_encoder_data(processed, cache, out, "sub-02")
    assert not (out / "preprocessed_eeg_npy/sub-02").exists()
    np.save(cache / "embeddings.npy", embeddings)
    bridge.prepare_encoder_data(processed, cache, out, "sub-02")
    assert (out / "preprocessed_eeg_npy/sub-02/train.npy").exists()


@pytest.mark.parametrize("bad_field", [{"normalized": True}, {"model": "ViT-B-32"}])
def test_bridge_rejects_explicit_wrong_visual_protocol(tmp_path, bad_field):
    processed, cache, out, _, _, _ = _fixture(tmp_path)
    report = {"visual_layers": list(bridge.VISUAL_LAYERS), **bad_field}
    (cache / "cache_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Alljoined visual cache"):
        bridge.prepare_encoder_data(processed, cache, out, "sub-01")
    assert not out.exists()
