#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def row_minmax(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    return (scores - lo) / np.maximum(hi - lo, eps)


def ranks(scores: np.ndarray) -> np.ndarray:
    labels = np.arange(scores.shape[0])
    order = np.argsort(-scores, axis=1)
    out = np.empty(scores.shape[0], dtype=np.int64)
    for i in range(scores.shape[0]):
        out[i] = int(np.where(order[i] == labels[i])[0][0]) + 1
    return out


def metrics(scores: np.ndarray) -> dict[str, float]:
    r = ranks(scores)
    return {
        "top1": float((r <= 1).mean() * 100.0),
        "top5": float((r <= 5).mean() * 100.0),
        "top10": float((r <= 10).mean() * 100.0),
        "mrr": float((1.0 / r).mean()),
        "mean_rank": float(r.mean()),
    }


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float, device: torch.device) -> np.ndarray:
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
    ones = torch.ones((x_t.shape[0], 1), dtype=x_t.dtype, device=device)
    x_aug = torch.cat([x_t, ones], dim=1)
    xtx = x_aug.T @ x_aug
    reg = torch.eye(xtx.shape[0], dtype=x_t.dtype, device=device) * float(ridge)
    reg[-1, -1] = 0.0
    xty = x_aug.T @ y_t
    weight = torch.linalg.solve(xtx + reg, xty)
    return weight.detach().cpu().numpy().astype(np.float32)


def apply_ridge(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    return (x_aug @ weight).astype(np.float32)


def loo_background_correct(query: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    if query.shape[0] <= 1:
        bias = query.mean(axis=0, keepdims=True) - mu.mean(axis=0, keepdims=True)
        return l2_normalize(query - float(alpha) * bias)
    loo_query_mean = (query.sum(axis=0, keepdims=True) - query) / float(query.shape[0] - 1)
    bias = loo_query_mean - mu.mean(axis=0, keepdims=True)
    return l2_normalize(query - float(alpha) * bias)


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_name: str


def parse_run_spec(spec: str) -> RunSpec:
    label, run_name = spec.split("=", 1)
    return RunSpec(label=label, run_name=run_name)


def split_train_indices(n: int, seed: str, subject: str, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    rng_seed = int(seed) + 1009 * sum(ord(c) for c in subject)
    rng = np.random.default_rng(rng_seed)
    indices = rng.permutation(n)
    n_val = max(200, int(round(n * float(val_fraction))))
    n_val = min(max(1, n_val), n - 1)
    val_idx = np.sort(indices[:n_val])
    fit_idx = np.sort(indices[n_val:])
    return fit_idx, val_idx


def choose_alpha_on_train_validation(
    branch_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    alpha_grid: list[float],
    ridge: float,
    device: torch.device,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    fused_scores_by_alpha: dict[float, list[np.ndarray]] = {alpha: [] for alpha in alpha_grid}

    for _label, (train_q, train_i) in branch_arrays.items():
        fit_q = train_q[fit_idx]
        fit_i = train_i[fit_idx]
        val_q = train_q[val_idx]
        val_i = train_i[val_idx]
        weight = fit_ridge(fit_i, fit_q, ridge, device)
        val_mu = l2_normalize(apply_ridge(val_i, weight))
        for alpha in alpha_grid:
            val_q_cal = loo_background_correct(val_q, val_mu, alpha)
            fused_scores_by_alpha[alpha].append(row_minmax(val_q_cal @ val_mu.T))

    scored: list[tuple[float, dict[str, float]]] = []
    for alpha in alpha_grid:
        fused = np.mean(fused_scores_by_alpha[alpha], axis=0)
        m = metrics(fused)
        scored.append((alpha, m))
        out = {"alpha": alpha}
        out.update(m)
        rows.append(out)

    # Tie-breaker: prefer higher metrics, then alpha closest to the theory-derived value 1.0.
    best_alpha, _ = sorted(
        scored,
        key=lambda item: (
            -item[1]["top1"],
            -item[1]["top5"],
            -item[1]["top10"],
            -item[1]["mrr"],
            abs(item[0] - 1.0),
            item[0],
        ),
    )[0]
    return float(best_alpha), rows


def bootstrap_ci(values: list[float], seed: int = 20260830, n_boot: int = 10000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal ReLiC-SC evaluation from saved train/test embeddings.")
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--run-spec", dest="run_specs", action="append", required=True, help="label=run_name")
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--fixed-alpha", type=float, default=1.0)
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [parse_run_spec(s) for s in args.run_specs]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        for subject in args.subjects:
            raw_train: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            raw_test: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for spec in specs:
                path = args.embedding_root / subject / spec.run_name / f"seed-{seed}" / "train_test_embeddings.npz"
                if not path.exists():
                    print(f"[warn] missing {path}")
                    continue
                data = np.load(path)
                raw_train[spec.label] = (l2_normalize(data["train_pred"]), l2_normalize(data["train_image"]))
                raw_test[spec.label] = (l2_normalize(data["test_pred"]), l2_normalize(data["test_image"]))

            if not raw_train:
                continue

            n_train = next(iter(raw_train.values()))[0].shape[0]
            fit_idx, val_idx = split_train_indices(n_train, seed, subject, args.val_fraction)
            selected_alpha, selection_rows = choose_alpha_on_train_validation(
                raw_train,
                fit_idx,
                val_idx,
                [float(a) for a in args.alpha_grid],
                args.ridge,
                device,
            )
            for row in selection_rows:
                alpha_rows.append(
                    {
                        "seed": seed,
                        "subject": subject,
                        "scope": "fusion_trainval",
                        "branch_combo": "+".join(raw_train.keys()),
                        "alpha": row["alpha"],
                        "selected_alpha": selected_alpha,
                        "selected": int(float(row["alpha"]) == selected_alpha),
                        "ridge": args.ridge,
                        "top1": row["top1"],
                        "top5": row["top5"],
                        "top10": row["top10"],
                        "mrr": row["mrr"],
                        "mean_rank": row["mean_rank"],
                    }
                )

            score_store: dict[str, dict[str, np.ndarray]] = {}
            for label in raw_train.keys():
                train_q, train_i = raw_train[label]
                test_q, test_i = raw_test[label]
                weight = fit_ridge(train_i, train_q, args.ridge, device)
                test_mu = l2_normalize(apply_ridge(test_i, weight))

                rng = np.random.default_rng(int(seed) + sum(ord(c) for c in subject + label) * 97)
                shuffled_train_q = train_q.copy()
                rng.shuffle(shuffled_train_q, axis=0)
                shuffle_weight = fit_ridge(train_i, shuffled_train_q, args.ridge, device)
                shuffled_mu = l2_normalize(apply_ridge(test_i, shuffle_weight))

                branch_scores = {
                    "cosine": test_q @ test_i.T,
                    "relic": test_q @ test_mu.T,
                    "relic_sc_fixed_alpha1_loo": loo_background_correct(test_q, test_mu, args.fixed_alpha) @ test_mu.T,
                    "relic_sc_trainval_alpha": loo_background_correct(test_q, test_mu, selected_alpha) @ test_mu.T,
                    "shuffled_relic": test_q @ shuffled_mu.T,
                    "shuffled_relic_sc_fixed_alpha1_loo": loo_background_correct(test_q, shuffled_mu, args.fixed_alpha) @ shuffled_mu.T,
                }
                score_store[label] = branch_scores
                for method, score in branch_scores.items():
                    row = {
                        "seed": seed,
                        "subject": subject,
                        "scope": "branch",
                        "branch_combo": label,
                        "method": method,
                        "ridge": args.ridge,
                        "fixed_alpha": args.fixed_alpha if "fixed_alpha1" in method else "",
                        "selected_alpha": selected_alpha if method == "relic_sc_trainval_alpha" else "",
                    }
                    row.update(metrics(score))
                    rows.append(row)

            labels = list(score_store.keys())
            if len(labels) >= 2:
                methods = sorted(set().union(*(score_store[label].keys() for label in labels)))
                for method in methods:
                    if not all(method in score_store[label] for label in labels):
                        continue
                    fused = np.mean([row_minmax(score_store[label][method]) for label in labels], axis=0)
                    row = {
                        "seed": seed,
                        "subject": subject,
                        "scope": "fusion",
                        "branch_combo": "+".join(labels),
                        "method": method,
                        "ridge": args.ridge,
                        "fixed_alpha": args.fixed_alpha if "fixed_alpha1" in method else "",
                        "selected_alpha": selected_alpha if method == "relic_sc_trainval_alpha" else "",
                    }
                    row.update(metrics(fused))
                    rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "subject",
        "scope",
        "branch_combo",
        "method",
        "ridge",
        "fixed_alpha",
        "selected_alpha",
        "top1",
        "top5",
        "top10",
        "mrr",
        "mean_rank",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for key in sorted({(r["scope"], r["branch_combo"], r["method"]) for r in rows}):
        subset = [r for r in rows if (r["scope"], r["branch_combo"], r["method"]) == key]
        summary_rows.append(
            {
                "scope": key[0],
                "branch_combo": key[1],
                "method": key[2],
                "n": len(subset),
                "mean_top1": float(np.mean([float(r["top1"]) for r in subset])),
                "mean_top5": float(np.mean([float(r["top5"]) for r in subset])),
                "mean_top10": float(np.mean([float(r["top10"]) for r in subset])),
                "mean_mrr": float(np.mean([float(r["mrr"]) for r in subset])),
                "mean_rank": float(np.mean([float(r["mean_rank"]) for r in subset])),
                "min_top1": float(np.min([float(r["top1"]) for r in subset])),
                "max_top1": float(np.max([float(r["top1"]) for r in subset])),
            }
        )
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "branch_combo",
                "method",
                "n",
                "mean_top1",
                "mean_top5",
                "mean_top10",
                "mean_mrr",
                "mean_rank",
                "min_top1",
                "max_top1",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    alpha_path = args.out.with_name(args.out.stem + "_alpha_selection.csv")
    with alpha_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "subject",
                "scope",
                "branch_combo",
                "alpha",
                "selected_alpha",
                "selected",
                "ridge",
                "top1",
                "top5",
                "top10",
                "mrr",
                "mean_rank",
            ],
        )
        writer.writeheader()
        writer.writerows(alpha_rows)

    delta_rows: list[dict[str, object]] = []
    for scope, branch_combo in sorted({(r["scope"], r["branch_combo"]) for r in rows}):
        keys = sorted({(r["seed"], r["subject"]) for r in rows if r["scope"] == scope and r["branch_combo"] == branch_combo})
        for method in sorted({r["method"] for r in rows if r["scope"] == scope and r["branch_combo"] == branch_combo and r["method"] != "relic"}):
            deltas: list[float] = []
            for seed, subject in keys:
                subset = [
                    r
                    for r in rows
                    if r["scope"] == scope and r["branch_combo"] == branch_combo and r["seed"] == seed and r["subject"] == subject
                ]
                by_method = {r["method"]: r for r in subset}
                if "relic" not in by_method or method not in by_method:
                    continue
                delta = float(by_method[method]["top1"]) - float(by_method["relic"]["top1"])
                deltas.append(delta)
                delta_rows.append(
                    {
                        "seed": seed,
                        "subject": subject,
                        "scope": scope,
                        "branch_combo": branch_combo,
                        "method": method,
                        "base_method": "relic",
                        "delta_top1": delta,
                        "delta_top5": float(by_method[method]["top5"]) - float(by_method["relic"]["top5"]),
                        "delta_top10": float(by_method[method]["top10"]) - float(by_method["relic"]["top10"]),
                    }
                )
            if deltas and scope == "fusion":
                wins = int(sum(d > 1e-9 for d in deltas))
                ties = int(sum(abs(d) <= 1e-9 for d in deltas))
                losses = int(sum(d < -1e-9 for d in deltas))
                lo, hi = bootstrap_ci(deltas)
                print(
                    f"[delta] {branch_combo} {method} vs relic: "
                    f"mean={np.mean(deltas):.2f}, wins/ties/losses={wins}/{ties}/{losses}, ci95=[{lo:.2f}, {hi:.2f}]"
                )

    delta_path = args.out.with_name(args.out.stem + "_paired_deltas.csv")
    with delta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "subject",
                "scope",
                "branch_combo",
                "method",
                "base_method",
                "delta_top1",
                "delta_top5",
                "delta_top10",
            ],
        )
        writer.writeheader()
        writer.writerows(delta_rows)

    print(f"[saved] metrics: {args.out}")
    print(f"[saved] summary: {summary_path}")
    print(f"[saved] alpha selection: {alpha_path}")
    print(f"[saved] paired deltas: {delta_path}")
    print("[fusion summary]")
    for row in sorted([r for r in summary_rows if r["scope"] == "fusion"], key=lambda r: (r["mean_top1"], r["mean_top5"]), reverse=True):
        print(
            f"{row['mean_top1']:6.2f} top1 | {row['mean_top5']:6.2f} top5 | "
            f"{row['mean_top10']:6.2f} top10 | {row['branch_combo']} | {row['method']}"
        )


if __name__ == "__main__":
    main()
