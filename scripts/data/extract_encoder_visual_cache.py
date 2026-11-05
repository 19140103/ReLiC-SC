"""Create the Alljoined CLIP depth cache consumed by prepare_encoder_data.py.

Use already-downloaded official stimulus images and a local, trusted OpenAI CLIP
ViT-B/32 checkpoint. This command never downloads images or model weights.
It follows the historical Alljoined feature protocol: QuickGELU, eight depths,
patch-token mean after layer normalization and projection, no L2 normalization,
FP32 inference and float16 cache storage.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


BACKBONE = "ViT-B-32-quickgelu"
VISUAL_LAYERS = (1, 2, 4, 5, 7, 8, 10, 12)


@lru_cache(maxsize=1)
def _shared_extractor():
    """Reuse the released extractor's token/block path, not a second implementation."""
    path = Path(__file__).resolve().parents[2] / "encoder/scripts/extract_multilevel_visual_features.py"
    spec = importlib.util.spec_from_file_location("relic_shared_visual_extractor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_image_ids(processed_subject: Path) -> list[str]:
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "test"):
        metadata = pd.read_parquet(processed_subject / f"{split}_metadata.parquet")
        if not {"image_path", "partition"}.issubset(metadata.columns):
            raise ValueError(f"{split} metadata needs image_path and partition columns")
        if metadata.empty or metadata["image_path"].isna().any():
            raise ValueError(f"{split} metadata must contain non-null image paths")
        if set(metadata["partition"]) != {f"stim_{split}"}:
            raise ValueError(f"{split} metadata must preserve the official image partition")
        ids = {str(value).replace("\\", "/").rsplit("/", 1)[-1] for value in metadata["image_path"]}
        if any(key in {"", ".", ".."} for key in ids):
            raise ValueError("Invalid image identifier in metadata")
        split_ids[split] = ids
    if split_ids["train"] & split_ids["test"]:
        raise ValueError("Train/test image identifiers overlap")
    return sorted(split_ids["train"] | split_ids["test"])


class _ImageDataset(Dataset):
    def __init__(self, paths: list[Path], preprocess):
        self.paths, self.preprocess = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.preprocess(image.convert("RGB"))


def extract_visual_cache(processed_subject: Path, image_root: Path, output_dir: Path,
                         model, preprocess, device: torch.device, batch_size: int = 128,
                         num_workers: int = 0, checkpoint_sha256: str | None = None) -> dict:
    """Extract a deterministic metadata-indexed cache from an already-loaded model."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    processed_subject, image_root, output_dir = map(Path, (processed_subject, image_root, output_dir))
    image_ids = collect_image_ids(processed_subject)
    image_root = image_root.resolve()
    paths = [(image_root / key).resolve() for key in image_ids]
    if any(not path.is_relative_to(image_root) for path in paths):
        raise ValueError("Image paths must remain within the supplied image root")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} official stimulus images; first: {missing[0]}")
    destinations = [output_dir / filename for filename in ("embeddings.npy", "image_index.parquet", "cache_report.json")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Cache output already exists; use a new output directory")
    shared = _shared_extractor()
    if not hasattr(model.visual, "transformer") or not hasattr(model.visual.transformer, "resblocks"):
        raise TypeError("Expected an OpenCLIP ViT visual encoder")
    selected = shared.selected_block_indices(len(model.visual.transformer.resblocks), list(VISUAL_LAYERS))
    model = model.to(device=device, dtype=torch.float32).eval()
    loader = DataLoader(_ImageDataset(paths, preprocess), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            levels = shared.block_forward_with_intermediates(
                model.visual, batch.to(device=device, dtype=torch.float32), selected)
            features = torch.stack(levels, dim=1).cpu().numpy().astype(np.float16)
            if not np.isfinite(features).all():
                raise ValueError("Non-finite visual features after float16 cache conversion")
            chunks.append(features)
    embeddings = np.concatenate(chunks, axis=0)
    report = {"model": BACKBONE, "pretrained": "local OpenAI CLIP checkpoint",
              "checkpoint_sha256": checkpoint_sha256,
              "images": len(image_ids), "embedding_shape": list(embeddings.shape),
              "dtype": str(embeddings.dtype), "normalized": False,
              "visual_layers": list(VISUAL_LAYERS), "inference_dtype": "float32",
              "pooling": "ln_post then mean of patch tokens (excluding CLS), then visual projection",
              "source_implementation": "encoder/scripts/extract_multilevel_visual_features.py",
              "embedding_path": "embeddings.npy", "index_path": "image_index.parquet"}
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage all three files before publishing them. Report is written last so an
    # interrupted cache cannot masquerade as a completed extraction.
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".extract-") as temporary:
        staging = Path(temporary)
        np.save(staging / "embeddings.npy", embeddings, allow_pickle=False)
        pd.DataFrame({"embedding_row": np.arange(len(image_ids), dtype=np.int32),
                      "image_id": image_ids}).to_parquet(staging / "image_index.parquet", index=False)
        (staging / "cache_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for path in destinations:
            (staging / path.name).replace(path)
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-subject", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Official stimuli/images directory containing files named by metadata basenames")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True,
                        help="Local official OpenAI CLIP ViT-B/32 checkpoint; model names/URLs are not accepted")
    parser.add_argument("--device", required=True, help="Explicit device, e.g. cpu or cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--trusted-torchscript-checkpoint", action="store_true",
                        help="Allow a trusted official TorchScript checkpoint rather than weights-only loading")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if not args.pretrained.is_file():
        raise FileNotFoundError("--pretrained must be an existing local OpenAI CLIP checkpoint; no downloads are attempted")
    shared = _shared_extractor()
    device = shared.resolve_device(args.device)
    # Built-in metadata lookup only: it does not download the checkpoint. Pass
    # the OpenAI preprocessing explicitly because a local path has no tag metadata.
    import open_clip
    from open_clip.pretrained import get_pretrained_cfg

    config = get_pretrained_cfg(BACKBONE, "openai")
    preprocess_kwargs = {"image_mean": config.get("mean"), "image_std": config.get("std"),
                         "image_interpolation": config.get("interpolation"),
                         "image_resize_mode": config.get("resize_mode")}
    model, _, preprocess = open_clip.create_model_and_transforms(
        BACKBONE, pretrained=str(args.pretrained.resolve()), precision="fp32", device=device,
        load_weights_only=not args.trusted_torchscript_checkpoint,
        **{key: value for key, value in preprocess_kwargs.items() if value is not None})
    with args.pretrained.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = extract_visual_cache(args.processed_subject, args.image_root, args.output_dir,
                                  model, preprocess, device, args.batch_size, args.num_workers,
                                  checkpoint_hash)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
