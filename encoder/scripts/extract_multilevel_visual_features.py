from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def resolve_device(device_arg: str) -> torch.device:
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{device_arg}' was requested, but CUDA is unavailable.")
    return requested


def resolve_pretrained(
    backbone: str,
    pretrained: str | None,
    prefer_openai_url: bool,
    cache_dir: str | None,
) -> tuple[str | None, dict[str, object]]:
    preprocess_kwargs: dict[str, object] = {}
    if pretrained is None or str(pretrained).lower() in {"", "none", "null"}:
        return None, preprocess_kwargs
    if prefer_openai_url:
        from open_clip.pretrained import download_pretrained, get_pretrained_cfg

        cfg = get_pretrained_cfg(backbone, pretrained)
        if not cfg:
            raise RuntimeError(f"No pretrained config found for {backbone}/{pretrained}.")
        checkpoint_path = download_pretrained(cfg, prefer_hf_hub=False, cache_dir=cache_dir)
        preprocess_kwargs = {
            "image_mean": tuple(cfg["mean"]) if "mean" in cfg else None,
            "image_std": tuple(cfg["std"]) if "std" in cfg else None,
            "image_interpolation": cfg.get("interpolation"),
            "image_resize_mode": cfg.get("resize_mode"),
        }
        return checkpoint_path, preprocess_kwargs
    return pretrained, preprocess_kwargs


def build_image_list(image_dir: str | Path) -> list[str]:
    image_dir = Path(image_dir)
    nested = image_dir / image_dir.name.replace("train_", "training_")
    if nested.is_dir():
        image_dir = nested
    image_list: list[str] = []
    for image_class in sorted(image_dir.iterdir()):
        if not image_class.is_dir():
            continue
        for image_file in sorted(image_class.iterdir()):
            if image_file.is_file():
                image_list.append(str(image_file))
    return image_list


class ImagePathDataset(Dataset):
    def __init__(self, image_list: list[str], preprocess):
        self.image_list = image_list
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_list[index]).convert("RGB")
        return self.preprocess(image)


def selected_block_indices(num_blocks: int, requested: list[int] | None) -> list[int]:
    if requested:
        indices = [idx - 1 if idx >= 1 else idx for idx in requested]
    elif num_blocks == 12:
        indices = [0, 1, 3, 4, 6, 7, 9, 11]
    else:
        positions = np.linspace(0, num_blocks - 1, 8)
        indices = sorted(dict.fromkeys(int(round(pos)) for pos in positions))
        while len(indices) < 8:
            for idx in range(num_blocks):
                if idx not in indices:
                    indices.append(idx)
                    indices = sorted(indices)
                    break
    if len(set(indices)) != len(indices):
        raise ValueError(f"Transformer block depths must be unique, got {indices}.")
    for idx in indices:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"Layer index {idx + 1} is out of range 1..{num_blocks}.")
    return indices


def make_tokens(visual, image_batch: torch.Tensor) -> torch.Tensor:
    x = visual.conv1(image_batch)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    class_embedding = visual.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
    x = torch.cat([class_embedding, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.patch_dropout(x)
    x = visual.ln_pre(x)
    return x


def block_forward_with_intermediates(
    visual,
    image_batch: torch.Tensor,
    selected_indices: list[int],
) -> list[torch.Tensor]:
    x = make_tokens(visual, image_batch)
    transformer = visual.transformer
    if not transformer.batch_first:
        x = x.transpose(0, 1).contiguous()
    selected = set(selected_indices)
    features = []
    attn_mask = getattr(transformer, "attn_mask", None)
    for block_idx, block in enumerate(transformer.resblocks):
        x = block(x, attn_mask=attn_mask)
        if block_idx in selected:
            x_tokens = x.transpose(0, 1) if not transformer.batch_first else x
            x_norm = visual.ln_post(x_tokens)
            pooled = x_norm[:, 1:, :].mean(dim=1)
            if visual.proj is not None:
                pooled = pooled @ visual.proj
            features.append(pooled)
    if len(features) != len(selected_indices):
        raise RuntimeError(f"Collected {len(features)} features for {len(selected_indices)} layers.")
    return features


def extract_features_for_split(
    image_dir: str | Path,
    num_images_per_object: int,
    preprocess,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_images: int,
    selected_indices: list[int],
) -> list[np.ndarray]:
    image_list = build_image_list(image_dir)
    if max_images > 0:
        image_list = image_list[:max_images]
    dataset = ImagePathDataset(image_list, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    per_level: list[list[np.ndarray]] = [[] for _ in selected_indices]
    visual = model.visual
    for image_batch in tqdm(loader, desc=f"extract {Path(image_dir).name}"):
        image_batch = image_batch.to(device)
        with torch.no_grad():
            batch_features = block_forward_with_intermediates(visual, image_batch, selected_indices)
        for level_idx, feature in enumerate(batch_features):
            per_level[level_idx].append(feature.detach().cpu().numpy())

    arrays = []
    for chunks in per_level:
        all_features = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if all_features.shape[0] % num_images_per_object != 0:
            raise ValueError(
                f"Feature count {all_features.shape[0]} is not divisible by "
                f"num_images_per_object={num_images_per_object}."
            )
        arrays.append(all_features.reshape(-1, num_images_per_object, all_features.shape[-1]))
    return arrays


def save_split(arrays: list[np.ndarray], output_dir: str | Path, label_start: int, split_name: str) -> None:
    output_dir = Path(output_dir)
    for level_idx, array in enumerate(arrays):
        level_dir = output_dir / f"level-{label_start + level_idx:03d}"
        level_dir.mkdir(parents=True, exist_ok=True)
        output_path = level_dir / f"image_{split_name}.npy"
        tmp_output = level_dir / f"image_{split_name}.tmp.npy"
        np.save(tmp_output, array)
        os.replace(tmp_output, output_path)
        print(f"Saved {split_name} level {label_start + level_idx:03d}: {array.shape}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract multi-level CLIP visual features.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--prefer-openai-url", action="store_true")
    parser.add_argument(
        "--trusted-torchscript-checkpoint",
        action="store_true",
        help="Allow loading a trusted TorchScript checkpoint such as the official OpenAI CLIP weights.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--image-set-dir", default="data/things_eeg/image_set")
    parser.add_argument("--output-dir", default="data/things_eeg/image_feature/ViT-B32-multilevel-8")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="1-based transformer block depths. Default for ViT-B/32 is 1 2 4 5 7 8 10 12.",
    )
    parser.add_argument("--label-start", type=int, default=100)
    parser.add_argument("--num-images-per-object-train", type=int, default=10)
    parser.add_argument("--num-images-per-object-test", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    import open_clip

    print("Input arguments:")
    for key, value in vars(args).items():
        print(f"{key:32} {value}")
    sys.stdout.flush()

    device = resolve_device(args.device)
    pretrained, preprocess_kwargs = resolve_pretrained(
        args.backbone,
        args.pretrained,
        args.prefer_openai_url,
        args.cache_dir,
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.backbone,
        pretrained=pretrained,
        precision="fp32",
        device=device,
        load_weights_only=not args.trusted_torchscript_checkpoint,
        **{key: value for key, value in preprocess_kwargs.items() if value is not None},
    )
    model.eval()
    if not hasattr(model.visual, "transformer") or not hasattr(model.visual.transformer, "resblocks"):
        raise TypeError("This extractor currently supports OpenCLIP ViT-style visual backbones.")

    selected_indices = selected_block_indices(len(model.visual.transformer.resblocks), args.layers)
    print("Selected 1-based transformer depths:", [idx + 1 for idx in selected_indices], flush=True)

    complete = True
    for level_idx in range(len(selected_indices)):
        level_dir = Path(args.output_dir) / f"level-{args.label_start + level_idx:03d}"
        complete = complete and (level_dir / "image_train.npy").exists()
        complete = complete and (level_dir / "image_test.npy").exists()
    if complete and not args.overwrite:
        print(f"Output directory {args.output_dir} is complete, skipping extraction.")
        return

    train_arrays = extract_features_for_split(
        Path(args.image_set_dir) / "train_images",
        args.num_images_per_object_train,
        preprocess,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.max_images_per_split,
        selected_indices,
    )
    save_split(train_arrays, args.output_dir, args.label_start, "train")

    test_arrays = extract_features_for_split(
        Path(args.image_set_dir) / "test_images",
        args.num_images_per_object_test,
        preprocess,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.max_images_per_split,
        selected_indices,
    )
    save_split(test_arrays, args.output_dir, args.label_start, "test")


if __name__ == "__main__":
    main()
