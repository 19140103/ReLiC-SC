from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (
    PairDataset,
    load_things_eeg,
    load_visual_representations,
    standardize_by_train,
    visual_targets_from_keys,
)
from .model import CEVRModel, MultiLevelVisualProjector
from .utils import DEFAULT_CHANNELS, count_parameters, l2_normalize_np, seed_everything, write_csv, write_json


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_visual_dirs(config: dict[str, Any]) -> list[str]:
    data_cfg = config["data"]
    if data_cfg.get("visual_feature_dirs"):
        return [str(Path(path)) for path in data_cfg["visual_feature_dirs"]]
    root = Path(data_cfg["visual_feature_root"])
    levels = [str(level) for level in data_cfg.get("visual_levels", [])]
    if not levels:
        raise ValueError("Set data.visual_feature_dirs or data.visual_feature_root plus data.visual_levels.")
    return [str(root / f"level-{level}") for level in levels]


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = l2_normalize_np(pred.astype(np.float32, copy=False))
    target = l2_normalize_np(target.astype(np.float32, copy=False))
    sim = pred @ target.T
    labels = np.arange(sim.shape[0])
    top1 = float((np.argmax(sim, axis=1) == labels).mean() * 100.0)
    k = min(5, sim.shape[1])
    top5_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    top5 = float((top5_idx == labels[:, None]).any(axis=1).mean() * 100.0)
    matched = np.diag(sim)
    if sim.shape[0] > 1:
        mismatched = sim[~np.eye(sim.shape[0], dtype=bool)].reshape(sim.shape[0], sim.shape[0] - 1).mean(axis=1)
    else:
        mismatched = np.zeros_like(matched)
    return {
        "top1": top1,
        "top5": top5,
        "matched_mean": float(matched.mean()),
        "mismatched_mean": float(mismatched.mean()),
        "matched_gap": float((matched - mismatched).mean()),
    }


@torch.no_grad()
def predict(
    model: CEVRModel,
    dataset: PairDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for eeg_batch, _visual, _group in loader:
        eeg_batch = eeg_batch.to(device)
        outputs.append(model(eeg_batch)["current"].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    raise ValueError(f"Unknown scheduler: {name}")


def train(config: dict[str, Any]) -> dict[str, Any]:
    run_cfg = config["run"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    seed = int(run_cfg.get("seed", 20260815))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(run_cfg.get("cpu", False)) else "cpu")

    subject = str(run_cfg["subject"])
    run_name = str(run_cfg.get("run_name", "cevr"))
    output_dir = Path(run_cfg.get("output_dir", "results")) / subject / run_name / f"seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", config)

    image_train_features, image_test_features, visual_labels = load_visual_representations(resolve_visual_dirs(config))
    selected_channels = data_cfg.get("selected_channels", DEFAULT_CHANNELS)
    train_eeg, train_keys, test_eeg, test_keys = load_things_eeg(
        data_cfg["eeg_data_dir"],
        subject,
        selected_channels,
        cache=bool(data_cfg.get("cache_averaged_eeg", True)),
    )
    if data_cfg.get("eeg_standardization", "none") == "train":
        train_eeg, test_eeg, eeg_mean, eeg_std = standardize_by_train(train_eeg, test_eeg)
    else:
        eeg_mean = np.zeros((1, train_eeg.shape[1], 1), dtype=np.float32)
        eeg_std = np.ones((1, train_eeg.shape[1], 1), dtype=np.float32)

    train_visual = visual_targets_from_keys(image_train_features, train_keys)
    test_visual = visual_targets_from_keys(image_test_features, test_keys)
    train_ds = PairDataset(train_eeg, train_visual, train_keys)
    test_ds = PairDataset(test_eeg, test_visual, test_keys)

    batch_size = int(train_cfg.get("batch_size", 1024))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = CEVRModel(
        channels=int(train_eeg.shape[1]),
        embed_dim=int(model_cfg.get("embed_dim", 512)),
        window_ends=[int(x) for x in model_cfg["window_ends"]],
        additive_hidden_dim=int(model_cfg.get("additive_hidden_dim", 1024)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        eeg_fusion_mode=str(model_cfg.get("eeg_fusion_mode", "mean")),
        current_slot_weight=float(model_cfg.get("current_slot_weight", 0.0)),
        parameter_matched_heads=bool(model_cfg.get("parameter_matched_heads", True)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        num_slots=int(model_cfg.get("num_slots", 5)),
    ).to(device)
    visual_projector = MultiLevelVisualProjector(
        image_dim=int(image_train_features.shape[-1]),
        embed_dim=int(model_cfg.get("embed_dim", 512)),
        num_layers=int(image_train_features.shape[0]),
        fusion_indices=list(range(int(image_train_features.shape[0]))),
        parameter_matched_heads=bool(model_cfg.get("parameter_matched_heads", True)),
        num_relations=int(model_cfg.get("num_slots", 5)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(visual_projector.parameters()),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = make_scheduler(
        optimizer,
        str(train_cfg.get("lr_scheduler", "cosine")),
        int(train_cfg.get("epochs", 50)),
        float(train_cfg.get("min_lr", 1e-6)),
    )

    current_test_features = test_visual
    summary: dict[str, Any] = {
        "subject": subject,
        "run_name": run_name,
        "device": str(device),
        "train_eeg_shape": list(train_eeg.shape),
        "test_eeg_shape": list(test_eeg.shape),
        "image_train_features_shape": list(image_train_features.shape),
        "image_test_features_shape": list(image_test_features.shape),
        "visual_level_labels": visual_labels,
        "selected_channels": list(selected_channels),
        "window_ends": list(model.window_ends),
        "eeg_fusion_mode": model.eeg_fusion_mode,
        "visual_fusion_mode": "mean",
        "final_checkpoint_protocol": True,
        "parameter_matched_heads": bool(model_cfg.get("parameter_matched_heads", True)),
        "current_slot_weight": float(model_cfg.get("current_slot_weight", 0.0)),
        **count_parameters(model, visual_projector),
    }
    if bool(run_cfg.get("dry_run", False)):
        write_json(output_dir / "dry_run_summary.json", summary)
        return summary

    history: list[dict[str, Any]] = []
    best_top1 = 0.0
    best_top5_at_top1 = 0.0
    best_epoch = 0
    start_time = time.time()
    epochs = int(train_cfg.get("epochs", 50))
    eval_every = int(train_cfg.get("eval_every", 5))
    log_every = int(train_cfg.get("log_every", 5))
    temperature = float(train_cfg.get("temperature", 0.07))

    for epoch in range(1, epochs + 1):
        model.train()
        visual_projector.train()
        total_loss = 0.0
        batches = 0
        for eeg_batch, visual_features, _group in train_loader:
            eeg_batch = eeg_batch.to(device)
            visual_features = visual_features.to(device)
            eeg_output = model(eeg_batch)["current"]
            visual_target, _projected = visual_projector(visual_features)
            loss = contrastive_loss(eeg_output, visual_target, temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            batches += 1

        if scheduler is not None:
            scheduler.step()

        row: dict[str, Any] = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / max(batches, 1),
        }
        if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
            pred = predict(model, test_ds, device, batch_size)
            image_embed = visual_projector.encode_numpy(current_test_features, device, batch_size)
            metrics = retrieval_metrics(pred, image_embed)
            row.update(metrics)
            if metrics["top1"] > best_top1:
                best_top1 = float(metrics["top1"])
                best_top5_at_top1 = float(metrics["top5"])
                best_epoch = epoch
        history.append(row)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"[{run_name}] {subject} epoch={epoch:03d} "
                f"loss={row['train_loss']:.4f} top1={row.get('top1', float('nan')):.3f}",
                flush=True,
            )

    pred = predict(model, test_ds, device, batch_size)
    image_embed = visual_projector.encode_numpy(current_test_features, device, batch_size)
    metrics = retrieval_metrics(pred, image_embed)
    final_metrics = {
        **summary,
        "epochs": epochs,
        "train_time_seconds": float(time.time() - start_time),
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "matched_gap": metrics["matched_gap"],
        "best_diagnostic_top1": best_top1,
        "best_diagnostic_top5_at_top1": best_top5_at_top1,
        "best_diagnostic_epoch": best_epoch,
        "eeg_component_weights": model.eeg_component_weights(),
        "visual_fusion_weights": visual_projector.fusion_weights(),
    }
    write_csv(output_dir / "history.csv", history)
    write_json(output_dir / "final_metrics.json", final_metrics)
    np.savez_compressed(
        output_dir / "final_embeddings.npz",
        current_pred=pred,
        current_image=image_embed,
        current_object_idx=test_keys[:, 0],
        current_image_idx=test_keys[:, 1],
    )
    torch.save(
        {
            "model": model.state_dict(),
            "visual_projector": visual_projector.state_dict(),
            "config": config,
            "final_metrics": final_metrics,
            "eeg_mean": eeg_mean,
            "eeg_std": eeg_std,
        },
        output_dir / "final_checkpoint.pt",
    )
    return final_metrics


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    overrides: dict[str, Any] = {"run": {}, "data": {}}
    if args.subject is not None:
        overrides["run"]["subject"] = args.subject
    if args.run_name is not None:
        overrides["run"]["run_name"] = args.run_name
    if args.output_dir is not None:
        overrides["run"]["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides["run"]["seed"] = args.seed
    if args.cpu:
        overrides["run"]["cpu"] = True
    if args.dry_run:
        overrides["run"]["dry_run"] = True
    if args.eeg_data_dir is not None:
        overrides["data"]["eeg_data_dir"] = args.eeg_data_dir
    if args.visual_feature_root is not None:
        overrides["data"]["visual_feature_root"] = args.visual_feature_root
    if args.visual_feature_dirs is not None:
        overrides["data"]["visual_feature_dirs"] = args.visual_feature_dirs
    return deep_update(config, overrides)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CEVR for one THINGS-EEG subject.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument("--subject", default=None, help="Subject id, e.g. sub-01.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--eeg-data-dir", default=None, help="Directory with info.json and sub-xx/train.npy/test.npy.")
    parser.add_argument("--visual-feature-root", default=None, help="Directory containing level-xxx visual feature folders.")
    parser.add_argument("--visual-feature-dirs", nargs="+", default=None, help="Explicit visual feature level directories.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = train(config_from_args(args))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
