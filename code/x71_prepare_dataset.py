"""Build a compact TBG-only artifact from existing Llama caches and labels.

This script performs no model inference and no response generation. It uses the
historical x66 seed-42 train/test row indices as the sealed outer split and
creates a validation split only inside the historical training rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

from edit_common import atomic_json_dump, file_record, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-states", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--source-labels", required=True)
    parser.add_argument("--clustered-order", required=True)
    parser.add_argument("--refusal-labels", required=True)
    parser.add_argument("--train-indices", required=True)
    parser.add_argument("--test-indices", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--hash-hidden", action="store_true")
    return parser.parse_args()


def _load_indices(path: str) -> np.ndarray:
    values = np.asarray(json.loads(Path(path).read_text()), dtype=np.int64)
    if values.ndim != 1 or len(np.unique(values)) != len(values):
        raise ValueError(f"Invalid or duplicate split indices in {path}")
    return values


def _strata(k: np.ndarray, total: np.ndarray) -> np.ndarray:
    return np.where(k == 0, 0, np.where(k == total, 2, 1))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = json.loads(Path(args.prompts).read_text())
    source = np.load(args.source_labels)
    clustered = json.loads(Path(args.clustered_order).read_text())
    labels_obj = json.loads(Path(args.refusal_labels).read_text())
    labels = labels_obj["rows"]
    train_outer = _load_indices(args.train_indices)
    test_outer = _load_indices(args.test_indices)
    frozen_cfg = config["frozen_dataset"]

    if len(prompts) != len(source):
        raise ValueError("prompts/source-label length mismatch")
    if len(prompts) != int(frozen_cfg["source_cache_rows"]):
        raise ValueError(
            "source cache row count does not match the frozen dataset contract: "
            f"{len(prompts)} != {frozen_cfg['source_cache_rows']}"
        )
    if not set(train_outer).isdisjoint(set(test_outer)):
        raise ValueError("outer train/test indices overlap")
    if max(train_outer.max(), test_outer.max()) >= len(prompts):
        raise ValueError("outer split index exceeds cache row count")

    harmful_order = [int(row["id"]) for row in clustered]
    n_harmful_full = len(harmful_order)
    if n_harmful_full != int(frozen_cfg["harmful_pool_rows"]):
        raise ValueError("harmful source-pool size disagrees with frozen dataset contract")
    if not np.all(source[:n_harmful_full] == 1) or not np.all(source[n_harmful_full:] == 0):
        raise ValueError("cache violates harmful-first, benign-second positional contract")

    # The physical cache retains all 30k source rows.  The frozen x66/x67
    # dataset is instead the exact union of the 13,527 labeled harmful rows
    # and the first 10,500 rows of the fixed seed-42 benign permutation.
    # Fail closed here so the 30k source cache can never be mistaken for the
    # experiment's final 24,027-row outer dataset.
    row_of_prompt_id = {prompt_id: row for row, prompt_id in enumerate(harmful_order)}
    if len(row_of_prompt_id) != n_harmful_full:
        raise ValueError("clustered harmful order contains duplicate prompt ids")
    try:
        expected_harmful = np.asarray(
            [row_of_prompt_id[int(prompt_id)] for prompt_id in labels], dtype=np.int64
        )
    except KeyError as exc:
        raise ValueError(f"refusal label prompt id absent from harmful source pool: {exc}") from exc
    if len(expected_harmful) != int(frozen_cfg["retained_harmful_rows"]):
        raise ValueError("refusal-label count disagrees with frozen harmful count")
    benign_rows = np.flatnonzero(source == 0)
    benign_rng = np.random.RandomState(int(frozen_cfg["benign_permutation_seed"]))
    expected_benign = benign_rng.permutation(benign_rows)[
        : int(frozen_cfg["retained_benign_rows"])
    ]
    expected_outer = np.sort(np.concatenate((expected_harmful, expected_benign)))
    actual_outer = np.sort(np.concatenate((train_outer, test_outer)))
    if len(actual_outer) != int(frozen_cfg["total_rows"]):
        raise ValueError("outer split size disagrees with frozen dataset total")
    if not np.array_equal(actual_outer, expected_outer):
        missing = np.setdiff1d(expected_outer, actual_outer)
        extra = np.setdiff1d(actual_outer, expected_outer)
        raise ValueError(
            "outer split rows do not exactly realize the frozen dataset "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    def labeled_harmful(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        selected = indices[source[indices] == 1]
        pids, counts, totals = [], [], []
        for row_index in selected:
            if row_index >= n_harmful_full:
                raise ValueError(f"harmful source row {row_index} lies beyond harmful order")
            prompt_id = harmful_order[int(row_index)]
            record = labels.get(str(prompt_id))
            if record is None:
                raise ValueError(
                    f"split contains harmful prompt id {prompt_id} without v4 refusal labels"
                )
            refusal = np.asarray(record["refusal"], dtype=np.int64)
            if refusal.ndim != 1 or len(refusal) == 0 or not np.isin(refusal, [0, 1]).all():
                raise ValueError(f"invalid refusal vector for prompt id {prompt_id}")
            pids.append(prompt_id)
            counts.append(int(refusal.sum()))
            totals.append(int(len(refusal)))
        return (
            selected,
            np.asarray(pids, dtype=np.int64),
            np.asarray(counts, dtype=np.int64),
            np.asarray(totals, dtype=np.int64),
        )

    tr_rows_all, tr_pids_all, tr_k_all, tr_n_all = labeled_harmful(train_outer)
    te_rows, te_pids, te_k, te_n = labeled_harmful(test_outer)
    if len(np.unique(tr_n_all)) != 1 or len(np.unique(te_n)) != 1:
        raise ValueError("binomial response count K is not constant within a split")

    split_cfg = config["split"]
    local = np.arange(len(tr_rows_all))
    train_local, val_local = train_test_split(
        local,
        test_size=float(split_cfg["validation_fraction"]),
        random_state=int(split_cfg["validation_seed"]),
        stratify=_strata(tr_k_all, tr_n_all),
    )

    split_rows = {
        "train": tr_rows_all[train_local],
        "validation": tr_rows_all[val_local],
        "test": te_rows,
    }
    split_pids = {
        "train": tr_pids_all[train_local],
        "validation": tr_pids_all[val_local],
        "test": te_pids,
    }
    split_k = {
        "train": tr_k_all[train_local],
        "validation": tr_k_all[val_local],
        "test": te_k,
    }
    split_n = {
        "train": tr_n_all[train_local],
        "validation": tr_n_all[val_local],
        "test": te_n,
    }

    benign_train = train_outer[source[train_outer] == 0]
    benign_test = test_outer[source[test_outer] == 0]
    rng = np.random.default_rng(int(split_cfg["validation_seed"]))
    rng.shuffle(benign_train)
    rng.shuffle(benign_test)
    benign_retain = benign_train[: int(split_cfg["benign_retain_n"])]
    benign_validation = benign_test[: int(split_cfg["benign_validation_n"])]

    cache = torch.load(args.hidden_states, map_location="cpu", weights_only=False)
    tbg_states = cache["tbg_states"]
    if len(tbg_states) != int(cache["num_layers"]):
        raise ValueError("hidden-state list length disagrees with num_layers")
    for state_index, state in enumerate(tbg_states):
        if state.ndim != 2 or state.shape[0] != len(prompts):
            raise ValueError(f"bad tbg state shape at index {state_index}: {tuple(state.shape)}")

    artifact = {
        "schema_version": 1,
        "metadata": {
            "model_id": config["model_id"],
            "probe_chat_date": config["probe_chat_date"],
            "outer_split_seed": split_cfg["outer_seed"],
            "validation_seed": split_cfg["validation_seed"],
            "num_states": len(tbg_states),
            "hidden_dim": int(tbg_states[0].shape[1]),
            "cache_metadata": cache.get("metadata", {}),
        },
        "states": {},
        "labels": {},
    }
    for name in ("train", "validation", "test"):
        row_tensor = torch.as_tensor(split_rows[name], dtype=torch.long)
        artifact["states"][name] = [state[row_tensor].clone() for state in tbg_states]
        artifact["labels"][name] = {
            "full_rows": torch.as_tensor(split_rows[name], dtype=torch.long),
            "prompt_ids": torch.as_tensor(split_pids[name], dtype=torch.long),
            "refusal_count": torch.as_tensor(split_k[name], dtype=torch.long),
            "num_responses": torch.as_tensor(split_n[name], dtype=torch.long),
        }
    torch.save(artifact, out_dir / "probe_dataset.pt")

    prompt_splits: dict[str, list[dict[str, object]]] = {}
    for name in ("train", "validation", "test"):
        prompt_splits[name] = [
            {
                "full_row": int(row),
                "prompt_id": int(pid),
                "prompt": prompts[int(row)],
                "refusal_count": int(k),
                "num_responses": int(n),
            }
            for row, pid, k, n in zip(
                split_rows[name], split_pids[name], split_k[name], split_n[name]
            )
        ]
    for name, rows in (
        ("benign_retain", benign_retain),
        ("benign_validation", benign_validation),
    ):
        prompt_splits[name] = [
            {"full_row": int(row), "prompt": prompts[int(row)]} for row in rows
        ]
    atomic_json_dump(prompt_splits, out_dir / "prompt_splits.json")

    records = {
        "hidden_states": file_record(args.hidden_states, hash_contents=args.hash_hidden),
        "prompts": file_record(args.prompts),
        "source_labels": file_record(args.source_labels),
        "clustered_order": file_record(args.clustered_order),
        "refusal_labels": file_record(args.refusal_labels),
        "train_indices": file_record(args.train_indices),
        "test_indices": file_record(args.test_indices),
    }
    manifest = {
        "schema_version": 1,
        "inputs": records,
        "outputs": {
            "probe_dataset": file_record(out_dir / "probe_dataset.pt", hash_contents=False),
            "prompt_splits": file_record(out_dir / "prompt_splits.json"),
        },
        "counts": {
            name: len(prompt_splits[name]) for name in prompt_splits
        },
        "frozen_outer_dataset": {
            "source_cache_rows": len(prompts),
            "total_rows": len(actual_outer),
            "harmful_rows": int((source[actual_outer] == 1).sum()),
            "benign_rows": int((source[actual_outer] == 0).sum()),
            "exact_contract_match": True,
        },
        "historical_sampling_reused": True,
        "model_inference_performed": False,
    }
    atomic_json_dump(manifest, out_dir / "prepare_manifest.json")
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    print(f"saved compact artifact -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
