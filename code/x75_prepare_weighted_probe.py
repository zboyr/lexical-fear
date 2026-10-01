"""Audit x75 weights and inherited x71 split membership without copying states."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

from edit_common import atomic_json_dump, file_record  # noqa: E402
from x75_weighting import ARMS, refusal_rate, subgroup_name, weight_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Inherited x71 probe_dataset.pt")
    parser.add_argument("--prompt-splits", required=True)
    parser.add_argument("--out", default="data/runs/x75/x75_weight_audit.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = torch.load(args.dataset, map_location="cpu", weights_only=False)
    prompt_splits = json.loads(Path(args.prompt_splits).read_text())
    labels = dataset["labels"]
    splits = {}
    subgroup_membership = {}
    for split in ("train", "validation", "test"):
        count = labels[split]["refusal_count"]
        total = labels[split]["num_responses"]
        rate = refusal_rate(count, total)
        if not torch.all(total == 10):
            raise ValueError(f"{split} does not contain exactly ten responses per prompt")
        prompt_ids = labels[split]["prompt_ids"].long()
        expected_ids = torch.tensor(
            [int(row["prompt_id"]) for row in prompt_splits[split]], dtype=torch.long
        )
        if not torch.equal(prompt_ids, expected_ids):
            raise ValueError(f"{split} prompt ids differ between dataset and prompt_splits")
        subgroup_membership[split] = {
            str(int(prompt_id)): subgroup_name(float(value))
            for prompt_id, value in zip(prompt_ids, rate)
        }
        splits[split] = {
            "n": int(rate.numel()),
            "mean_r": float(rate.mean()),
            "subgroups": {
                name: sum(value == name for value in subgroup_membership[split].values())
                for name in (
                    "stable_comply",
                    "low_refusal_mixed",
                    "high_refusal_mixed",
                    "stable_refuse",
                )
            },
            "arms": {arm: weight_audit(rate, arm) for arm in ARMS},
        }

    result = {
        "schema_version": 1,
        "experiment": "x75",
        "target": "r = refusal_count / 10",
        "arms": list(ARMS),
        "splits": splits,
        "subgroup_membership": subgroup_membership,
        "inputs": {
            "dataset": file_record(args.dataset, hash_contents=False),
            "prompt_splits": file_record(args.prompt_splits),
        },
        "model_inference_performed": False,
        "sealed_test_used_for_selection": False,
    }
    atomic_json_dump(result, args.out)
    print(json.dumps({name: block["subgroups"] for name, block in splits.items()}, indent=2))


if __name__ == "__main__":
    main()
