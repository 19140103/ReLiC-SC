#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def import_train_module(train_script: Path):
    spec = importlib.util.spec_from_file_location("chronotrace_set_align_train_export_current", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import training script: {train_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def visual_group_label(path: str | Path) -> str:
    name = Path(path).name
    if "-" in name:
        return name.split("-", 1)[0]
    return "visual"


def standardize_with_checkpoint(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32, copy=False) - mean.astype(np.float32)) / np.maximum(std.astype(np.float32), 1e-6)).astype(
        np.float32,
        copy=False,
    )


def current_bank_from_keys(image_bank: np.ndarray, keys: np.ndarray) -> np.ndarray:
    object_idx = keys[:, 0].astype(np.int64, copy=False)
    image_idx = keys[:, 1].astype(np.int64, copy=False)
    return image_bank[:, object_idx, image_idx, :].transpose(1, 0, 2).astype(np.float32, copy=False)


def empty_targets(num_samples: int, num_levels: int, image_dim: int, num_relations: int, current_relation_index: int) -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((num_samples, num_relations, num_levels, image_dim), dtype=np.float32)
    masks = np.zeros((num_samples, num_relations), dtype=np.float32)
    masks[:, current_relation_index] = 1.0
    return targets, masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export current-stimulus train/test embeddings from a THINGS-EEG2 ChronoTRACE checkpoint without requiring chronological metadata."
    )
    parser.add_argument(
        "--cevr-root",
        type=Path,
        required=True,
        help="Path to the encoder training repository containing the saved checkpoint and training script.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args_cli = parse_args()
    module = import_train_module(args_cli.cevr_root / "scripts" / "chronotrace_set_align_train.py")

    requested_device = args_cli.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    checkpoint = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
    ckpt_args = SimpleNamespace(**checkpoint["args"])

    image_train_bank, image_test_bank, visual_labels = module.load_image_bank(ckpt_args.image_bank_dirs)
    visual_group_labels = [visual_group_label(bank_dir) for bank_dir in ckpt_args.image_bank_dirs]
    visual_group_name_to_id = {name: idx for idx, name in enumerate(dict.fromkeys(visual_group_labels))}
    visual_group_ids = [visual_group_name_to_id[name] for name in visual_group_labels]

    train_eeg, train_keys, test_eeg, test_keys = module.load_project_format_eeg(
        Path(ckpt_args.eeg_data_dir),
        ckpt_args.subject,
        ckpt_args.selected_channels,
    )
    eeg_mean = checkpoint.get("eeg_mean")
    eeg_std = checkpoint.get("eeg_std")
    if eeg_mean is None or eeg_std is None:
        raise RuntimeError("Checkpoint does not contain eeg_mean/eeg_std.")
    train_eeg = standardize_with_checkpoint(train_eeg, eeg_mean, eeg_std)
    test_eeg = standardize_with_checkpoint(test_eeg, eeg_mean, eeg_std)

    current_train_bank = current_bank_from_keys(image_train_bank, train_keys)
    current_test_bank = current_bank_from_keys(image_test_bank, test_keys)
    train_targets, train_masks = empty_targets(
        train_eeg.shape[0],
        image_train_bank.shape[0],
        image_train_bank.shape[-1],
        len(module.RELATIONS),
        module.CURRENT_RELATION_INDEX,
    )
    test_targets, test_masks = empty_targets(
        test_eeg.shape[0],
        image_test_bank.shape[0],
        image_test_bank.shape[-1],
        len(module.RELATIONS),
        module.CURRENT_RELATION_INDEX,
    )
    train_targets[:, module.CURRENT_RELATION_INDEX] = current_train_bank
    test_targets[:, module.CURRENT_RELATION_INDEX] = current_test_bank

    train_ds = module.ChronoSetDataset(train_eeg, train_targets, train_masks, train_keys)
    test_ds = module.ChronoSetDataset(test_eeg, test_targets, test_masks, test_keys)

    channels = int(train_eeg.shape[1])
    image_dim = int(image_train_bank.shape[-1])
    model = module.ChronoTraceSetModel(
        channels=channels,
        image_dim=image_dim,
        eeg_model_dim=ckpt_args.eeg_model_dim,
        embed_dim=ckpt_args.embed_dim,
        window_ends=ckpt_args.eeg_window_ends,
        window_mode=ckpt_args.eeg_window_mode,
        eeg_fusion_mode=ckpt_args.eeg_fusion_mode,
        eeg_channel_gate=ckpt_args.eeg_channel_gate,
        num_relations=len(module.RELATIONS),
        num_heads=ckpt_args.num_heads,
        dropout=ckpt_args.dropout,
        current_slot_weight=ckpt_args.current_slot_weight,
        eeg_encoder_type=ckpt_args.eeg_encoder_type,
        causal_tcn_dim=ckpt_args.causal_tcn_dim,
        causal_tcn_layers=ckpt_args.causal_tcn_layers,
        causal_tcn_kernel_size=ckpt_args.causal_tcn_kernel_size,
        causal_horizon_pool=ckpt_args.causal_horizon_pool,
        causal_gru_input_dim=ckpt_args.causal_gru_input_dim,
        causal_gru_hidden_dim=ckpt_args.causal_gru_hidden_dim,
        causal_gru_layers=ckpt_args.causal_gru_layers,
        causal_additive_hidden_dim=ckpt_args.causal_additive_hidden_dim,
    ).to(device)
    visual_projector = module.VisualEvidenceProjector(
        image_dim=image_dim,
        embed_dim=ckpt_args.embed_dim,
        num_levels=int(image_train_bank.shape[0]),
        num_relations=len(module.RELATIONS),
        num_eeg_levels=len(ckpt_args.eeg_window_ends),
        fixed_current_level=ckpt_args.fixed_current_level,
        current_fusion_indices=ckpt_args.current_fusion_indices,
        current_fusion_mode=ckpt_args.current_fusion_mode,
        current_fusion_group_ids=visual_group_ids,
        current_target_adapter=ckpt_args.current_target_adapter,
        current_target_bottleneck_dim=ckpt_args.current_target_bottleneck_dim,
        current_target_dropout=ckpt_args.current_target_dropout,
        stage_calibration_mode=ckpt_args.stage_calibration_mode,
        stage_calibration_group_ids=visual_group_ids,
        stage_calibration_init_indices=ckpt_args.stage_calibration_init_indices,
        stage_calibration_init_strength=ckpt_args.stage_calibration_init_strength,
        stage_calibration_temperature=ckpt_args.stage_calibration_temperature,
        init_strength=ckpt_args.visual_init_strength,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    visual_projector.load_state_dict(checkpoint["visual_projector"])
    model.eval()
    visual_projector.eval()

    train_pred = module.predict_current(model, train_ds, device, args_cli.batch_size)
    train_image = visual_projector.combine_current_np(current_train_bank, device, args_cli.batch_size)
    test_pred = module.predict_current(model, test_ds, device, args_cli.batch_size)
    test_image = visual_projector.combine_current_np(current_test_bank, device, args_cli.batch_size)

    args_cli.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args_cli.out,
        train_pred=train_pred,
        train_image=train_image,
        train_object_idx=train_keys[:, 0],
        train_image_idx=train_keys[:, 1],
        test_pred=test_pred,
        test_image=test_image,
        test_object_idx=test_keys[:, 0],
        test_image_idx=test_keys[:, 1],
    )
    summary = {
        "checkpoint": str(args_cli.checkpoint),
        "out": str(args_cli.out),
        "subject": ckpt_args.subject,
        "run_name": ckpt_args.run_name,
        "device": str(device),
        "train_pred_shape": list(train_pred.shape),
        "train_image_shape": list(train_image.shape),
        "test_pred_shape": list(test_pred.shape),
        "test_image_shape": list(test_image.shape),
        "train_key_shape": list(train_keys.shape),
        "test_key_shape": list(test_keys.shape),
        "visual_labels": visual_labels,
        "note": "Current-stimulus-only export; chronological prev/next metadata is not required for ReLiC-SC.",
    }
    args_cli.out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
