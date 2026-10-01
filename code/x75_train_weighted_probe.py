"""Train one x75 weighted refusal-rate probe and its permutation control."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

from edit_common import atomic_json_dump, file_record, load_config, set_seed  # noqa: E402
from x75_weighting import (  # noqa: E402
    ARMS,
    WEIGHT_FORMULAS,
    normalize_weights,
    raw_weights,
    refusal_rate,
    weight_audit,
    weighted_bce_with_logits,
    weighted_standardized,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weight-audit", required=True)
    parser.add_argument("--out-dir", default="data/runs/x75/probes")
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-indices", nargs="*", type=int)
    return parser.parse_args()


def weighted_nll_numpy(logits: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    logits_t = torch.from_numpy(logits.astype(np.float64))
    target_t = torch.from_numpy(target.astype(np.float64))
    weight_t = torch.from_numpy(weight.astype(np.float64))
    loss = F.binary_cross_entropy_with_logits(logits_t, target_t, reduction="none")
    return float((loss * weight_t).mean())


def metrics(
    logits: np.ndarray,
    target: np.ndarray,
    count: np.ndarray,
    total: np.ndarray,
    weight: np.ndarray,
) -> dict:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    majority = count * 2 >= total
    auc = None
    if len(np.unique(majority)) == 2:
        auc = float(roc_auc_score(majority, probability))
    rho = spearmanr(probability, target).statistic
    unweighted_nll = F.binary_cross_entropy_with_logits(
        torch.from_numpy(logits.astype(np.float64)),
        torch.from_numpy(target.astype(np.float64)),
    )
    return {
        "n": int(len(target)),
        "weighted_nll": weighted_nll_numpy(logits, target, weight),
        "unweighted_nll": float(unweighted_nll),
        "unweighted_brier": float(np.mean((probability - target) ** 2)),
        "unweighted_spearman": None if not np.isfinite(rho) else float(rho),
        "unweighted_auc_majority_refusal": auc,
        "mean_target": float(target.mean()),
        "weighted_mean_target": float(np.mean(weight * target)),
        "mean_prediction": float(probability.mean()),
        "weighted_mean_prediction": float(np.mean(weight * probability)),
        "n_positive_weight": int((weight > 0).sum()),
    }


def fit_logistic(
    x: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    l2: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros(x.shape[1], device=x.device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), device=x.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [w, b], max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = x @ w + b
        loss = weighted_bce_with_logits(logits, target, weight)
        if l2:
            loss = loss + 0.5 * l2 * w.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach(), b.detach()


def atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_probe(
    path: Path,
    *,
    w: torch.Tensor,
    b: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    state_index: int,
    l2: float,
    kind: str,
    arm: str,
    metadata: dict,
) -> None:
    q_raw = w.cpu() / std.cpu()
    q_norm = q_raw.norm()
    if not torch.isfinite(q_norm) or q_norm <= 0:
        raise ValueError(f"invalid raw probe direction norm for {kind}")
    artifact = {
        "schema_version": 1,
        "experiment": "x75",
        "kind": kind,
        "arm": arm,
        "weight_formula": WEIGHT_FORMULAS[arm],
        "target_definition": "r = refusal_count / num_responses",
        "state_index": int(state_index),
        "block_index": int(state_index - 1),
        "l2": float(l2),
        "w_standardized": w.cpu(),
        "bias": b.cpu(),
        "mean": mean.cpu(),
        "std": std.cpu(),
        "q_raw": q_raw,
        "q_unit": q_raw / q_norm,
        "metadata": metadata,
    }
    atomic_torch_save(artifact, path)


def _verify_audit(path: str, dataset_path: str, arm: str) -> dict:
    audit = json.loads(Path(path).read_text())
    if audit.get("experiment") != "x75" or arm not in audit.get("arms", []):
        raise ValueError("weight audit does not cover this x75 arm")
    recorded = audit["inputs"]["dataset"]
    actual = file_record(dataset_path, hash_contents=False)
    for key in ("path", "size_bytes", "mtime_ns"):
        if recorded.get(key) != actual.get(key):
            raise ValueError(f"dataset changed since weight audit: {key}")
    return audit


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    probe_cfg = config["probe"]
    seed = int(probe_cfg["permutation_seed"])
    set_seed(seed)
    audit = _verify_audit(args.weight_audit, args.dataset, args.arm)
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    labels = dataset["labels"]
    target = {
        split: refusal_rate(
            labels[split]["refusal_count"], labels[split]["num_responses"]
        )
        for split in ("train", "validation", "test")
    }
    raw = {split: raw_weights(target[split], args.arm) for split in target}
    weight = {split: normalize_weights(raw[split]) for split in target}
    device = torch.device(args.device)
    state_indices = args.state_indices or list(range(1, dataset["metadata"]["num_states"]))
    l2_grid = [float(value) for value in probe_cfg["l2_grid"]]
    max_iter = int(probe_cfg["max_iter"])
    search_rows = []
    winner = None

    for state_index in state_indices:
        train_cpu = dataset["states"]["train"][state_index]
        val_cpu = dataset["states"]["validation"][state_index]
        mean, std, x_train, x_val = weighted_standardized(
            train_cpu, weight["train"], val_cpu
        )
        x_train = x_train.to(device)
        x_val = x_val.to(device)
        y_train = target["train"].to(device)
        w_train = weight["train"].to(device)
        for l2 in l2_grid:
            probe_w, probe_b = fit_logistic(x_train, y_train, w_train, l2, max_iter)
            val_logits = (x_val @ probe_w + probe_b).float().cpu().numpy()
            row = {
                "state_index": int(state_index),
                "block_index": int(state_index - 1),
                "l2": l2,
                "validation": metrics(
                    val_logits,
                    target["validation"].numpy(),
                    labels["validation"]["refusal_count"].numpy(),
                    labels["validation"]["num_responses"].numpy(),
                    weight["validation"].numpy(),
                ),
            }
            search_rows.append(row)
            if winner is None or (
                row["validation"]["weighted_nll"]
                < winner["row"]["validation"]["weighted_nll"]
            ):
                winner = {
                    "row": row,
                    "w": probe_w.cpu(),
                    "b": probe_b.cpu(),
                    "mean": mean,
                    "std": std,
                }
        del x_train, x_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if winner is None:
        raise RuntimeError("probe search produced no candidate")
    best_state = int(winner["row"]["state_index"])
    best_l2 = float(winner["row"]["l2"])
    train_cpu = dataset["states"]["train"][best_state]
    val_cpu = dataset["states"]["validation"][best_state]
    test_cpu = dataset["states"]["test"][best_state]
    mean, std, x_train, x_val, x_test = weighted_standardized(
        train_cpu, weight["train"], val_cpu, test_cpu
    )
    x_train, x_val, x_test = x_train.to(device), x_val.to(device), x_test.to(device)
    probe_w, probe_b = fit_logistic(
        x_train,
        target["train"].to(device),
        weight["train"].to(device),
        best_l2,
        max_iter,
    )
    logits = {
        "train": (x_train @ probe_w + probe_b).cpu().numpy(),
        "validation": (x_val @ probe_w + probe_b).cpu().numpy(),
        "test": (x_test @ probe_w + probe_b).cpu().numpy(),
    }
    true_metrics = {
        split: metrics(
            logits[split],
            target[split].numpy(),
            labels[split]["refusal_count"].numpy(),
            labels[split]["num_responses"].numpy(),
            weight[split].numpy(),
        )
        for split in logits
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"x75_{args.arm}"
    true_path = out_dir / f"{prefix}_true.pt"
    control_path = out_dir / f"{prefix}_control.pt"
    metadata = {
        **dataset["metadata"],
        "dataset_file": file_record(args.dataset, hash_contents=False),
        "weight_audit": file_record(args.weight_audit),
        "permutation_seed": seed,
    }
    save_probe(
        true_path,
        w=probe_w,
        b=probe_b,
        mean=mean,
        std=std,
        state_index=best_state,
        l2=best_l2,
        kind="true",
        arm=args.arm,
        metadata=metadata,
    )
    np.save(out_dir / f"{prefix}_true_test_logits.npy", logits["test"].astype(np.float32))

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(target["train"]))
    perm_target = target["train"][permutation]
    perm_raw = raw["train"][permutation]
    perm_weight = normalize_weights(perm_raw)
    control_mean, control_std, cx_train, cx_test = weighted_standardized(
        train_cpu, perm_weight, test_cpu
    )
    cx_train, cx_test = cx_train.to(device), cx_test.to(device)
    control_w, control_b = fit_logistic(
        cx_train,
        perm_target.to(device),
        perm_weight.to(device),
        best_l2,
        max_iter,
    )
    control_logits = (cx_test @ control_w + control_b).cpu().numpy()
    control_metrics = metrics(
        control_logits,
        target["test"].numpy(),
        labels["test"]["refusal_count"].numpy(),
        labels["test"]["num_responses"].numpy(),
        weight["test"].numpy(),
    )
    save_probe(
        control_path,
        w=control_w,
        b=control_b,
        mean=control_mean,
        std=control_std,
        state_index=best_state,
        l2=best_l2,
        kind="control",
        arm=args.arm,
        metadata={**metadata, "permutation_index": 0},
    )
    np.save(
        out_dir / f"{prefix}_control_test_logits.npy",
        control_logits.astype(np.float32),
    )

    weighted_train_mean = float((weight["train"] * target["train"]).mean())
    clipped_mean = float(np.clip(weighted_train_mean, 1e-8, 1 - 1e-8))
    intercept_logit = float(np.log(clipped_mean / (1 - clipped_mean)))
    baseline_logits = np.full(len(target["test"]), intercept_logit, dtype=np.float64)
    result = {
        "schema_version": 1,
        "experiment": "x75",
        "arm": args.arm,
        "target": "r = refusal_count / num_responses",
        "weight_formula": WEIGHT_FORMULAS[args.arm],
        "selection_criterion": "minimum weighted validation binomial NLL",
        "search": search_rows,
        "selected": winner["row"],
        "true_probe": true_metrics,
        "control_probe_test": control_metrics,
        "weighted_intercept_baseline_test": metrics(
            baseline_logits,
            target["test"].numpy(),
            labels["test"]["refusal_count"].numpy(),
            labels["test"]["num_responses"].numpy(),
            weight["test"].numpy(),
        ),
        "weight_audits": {split: weight_audit(target[split], args.arm) for split in target},
        "artifacts": {
            "true": file_record(true_path),
            "control": file_record(control_path),
            "true_test_logits": file_record(out_dir / f"{prefix}_true_test_logits.npy"),
            "control_test_logits": file_record(
                out_dir / f"{prefix}_control_test_logits.npy"
            ),
        },
        "inputs": {
            "dataset": file_record(args.dataset, hash_contents=False),
            "weight_audit": file_record(args.weight_audit),
        },
        "sealed_test_used_for_selection": False,
    }
    result_path = out_dir / f"{prefix}_probe_results.json"
    atomic_json_dump(result, result_path)
    print(json.dumps({"arm": args.arm, "selected": result["selected"], "test": true_metrics["test"]}, indent=2))


if __name__ == "__main__":
    main()
