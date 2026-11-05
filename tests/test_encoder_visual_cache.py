"""Small offline extraction tests; no CLIP checkpoint or dataset is downloaded."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/data/extract_encoder_visual_cache.py"
spec = importlib.util.spec_from_file_location("extract_encoder_visual_cache", SCRIPT)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)
bridge_spec = importlib.util.spec_from_file_location("cache_test_bridge", ROOT / "scripts/data/prepare_encoder_data.py")
bridge = importlib.util.module_from_spec(bridge_spec)
bridge_spec.loader.exec_module(bridge)


class _Block(nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, x, attn_mask=None):
        return x + self.increment


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv1.weight.copy_(torch.eye(3).reshape(3, 3, 1, 1))
        self.class_embedding = nn.Parameter(torch.zeros(3))
        self.positional_embedding = nn.Parameter(torch.zeros(5, 3))
        self.patch_dropout = nn.Identity()
        self.ln_pre = nn.Identity()
        self.ln_post = nn.Identity()
        self.proj = nn.Parameter(torch.diag(torch.tensor([1.0, 2.0, 3.0])))
        self.transformer = nn.Module()
        self.transformer.batch_first = True
        self.transformer.resblocks = nn.ModuleList([_Block((idx + 1) / 100) for idx in range(12)])


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()


def _preprocess(image):
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255


def _inputs(tmp_path):
    processed, images, out = (tmp_path / name for name in ("processed", "images", "cache"))
    processed.mkdir()
    images.mkdir()
    for split, color in (("train", 80), ("test", 160)):
        Image.new("RGB", (2, 2), (color, color, color)).save(images / f"{split}.png")
        pd.DataFrame({"image_path": [f"stimuli\\images\\{split}.png"], "partition": [f"stim_{split}"],
                      "source_row_index": [1], "subject": [1], "category_num": [0], "category_img_num": [0]
                      }).to_parquet(processed / f"{split}_metadata.parquet", index=False)
        np.save(processed / f"{split}_eeg.npy", np.ones((1, 1, 2), dtype=np.float32))
    (processed / "channels.json").write_text(json.dumps(["Cz"]))
    np.save(processed / "times.npy", np.array([0.0, 0.004]))
    return processed, images, out


def test_cache_reuses_shared_forward_and_round_trips_bridge(tmp_path):
    processed, images, out = _inputs(tmp_path)
    report = extractor.extract_visual_cache(processed, images, out, _Model(), _preprocess,
                                           torch.device("cpu"), batch_size=1)
    assert report["model"] == "ViT-B-32-quickgelu"
    assert report["normalized"] is False
    assert report["dtype"] == "float16"
    assert report["visual_layers"] == [1, 2, 4, 5, 7, 8, 10, 12]
    index = pd.read_parquet(out / "image_index.parquet")
    assert index["image_id"].tolist() == ["test.png", "train.png"]
    np.testing.assert_array_equal(index["embedding_row"], [0, 1])
    values = np.load(out / "embeddings.npy")
    assert values.shape == (2, 8, 3)
    # Independent arithmetic for fake residual blocks: exclude CLS, mean patches,
    # then project. The cache preserves amplitude instead of L2 normalization.
    increments = np.cumsum(np.arange(1, 13) / 100)[np.asarray(extractor.VISUAL_LAYERS) - 1]
    expected = ((np.array([160, 80])[:, None, None] / 255 + increments[None, :, None]) * np.array([1, 2, 3])).astype(np.float16)
    np.testing.assert_allclose(values, expected, atol=0.002, rtol=0)
    prepared = tmp_path / "prepared"
    bridge.prepare_encoder_data(processed, out, prepared, "sub-01")
    visual_root = prepared / "image_feature" / bridge.VISUAL_BANK_NAME
    np.testing.assert_array_equal(np.load(visual_root / "level-100/image_train.npy")[0, 0], values[1, 0].astype(np.float32))
    with pytest.raises(FileExistsError, match="already exists"):
        extractor.extract_visual_cache(processed, images, out, _Model(), _preprocess, torch.device("cpu"))


def test_missing_images_fail_without_creating_cache(tmp_path):
    processed, images, out = _inputs(tmp_path)
    # Point at an empty image root rather than deleting a fixture file.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="Missing"):
        extractor.extract_visual_cache(processed, empty, out, _Model(), _preprocess, torch.device("cpu"))
    assert not out.exists()


def test_metadata_overlap_rejected(tmp_path):
    processed, _, _ = _inputs(tmp_path)
    metadata = pd.read_parquet(processed / "test_metadata.parquet")
    metadata["image_path"] = "train.png"
    metadata.to_parquet(processed / "test_metadata.parquet", index=False)
    with pytest.raises(ValueError, match="overlap"):
        extractor.collect_image_ids(processed)


def test_cli_help_and_no_implicit_checkpoint_download(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--pretrained" in result.stdout and "--image-root" in result.stdout
    result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--processed-subject", str(tmp_path),
                             "--image-root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
                             "--pretrained", str(tmp_path / "not-downloaded.pt"), "--device", "cpu"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "no downloads are attempted" in result.stderr
    assert not (tmp_path / "out").exists()
