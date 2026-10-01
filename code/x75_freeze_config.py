"""Freeze x75 settings, inherited input hashes, subgroups, and code hashes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

from edit_common import atomic_json_dump, file_record  # noqa: E402
from x75_dose_kl import CONTROL_REFERENCE_DOSES, TRUE_REFERENCE_DOSES  # noqa: E402
from x75_weighting import ARMS, WEIGHT_FORMULAS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x71-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt-splits", required=True)
    parser.add_argument("--refusal-labels", required=True)
    parser.add_argument("--weight-audit", required=True)
    parser.add_argument("--x71-probe", required=True)
    parser.add_argument("--x71-adapter", required=True)
    parser.add_argument("--x71-dose-kl", required=True)
    parser.add_argument("--x71-base-score", required=True)
    parser.add_argument("--x71-true-score", action="append", required=True)
    parser.add_argument("--x70-input", action="append", required=True)
    parser.add_argument("--capability-chat-date")
    parser.add_argument("--out", default="data/runs/x75/x75_config.json")
    return parser.parse_args()


def _score_doses(paths: list[str]) -> dict[str, dict]:
    records = {}
    doses = set()
    for path in paths:
        obj = json.loads(Path(path).read_text())
        if obj.get("kind") != "true":
            raise ValueError(f"x71 comparison score is not true-kind: {path}")
        dose = float(obj["dose"])
        if dose in doses:
            raise ValueError(f"duplicate x71 comparison dose {dose}")
        doses.add(dose)
        records[str(dose)] = file_record(path)
    if doses != set(TRUE_REFERENCE_DOSES):
        raise ValueError(f"x71 scores must cover {TRUE_REFERENCE_DOSES}; got {sorted(doses)}")
    return records


def _capability_date(paths: list[str], requested: str | None) -> str:
    dates = set()
    for path in paths:
        obj = json.loads(Path(path).read_text())
        if obj.get("condition") == "base" and obj.get("chat_date"):
            dates.add(str(obj["chat_date"]))
    if requested:
        dates.add(requested)
    if len(dates) != 1:
        raise ValueError(f"cannot freeze one capability chat date from {sorted(dates)}")
    return next(iter(dates))


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(json.loads(Path(args.x71_config).read_text()))
    weight_audit = json.loads(Path(args.weight_audit).read_text())
    if weight_audit.get("experiment") != "x75":
        raise ValueError("weight audit is not an x75 artifact")
    prompt_splits = json.loads(Path(args.prompt_splits).read_text())
    if len(prompt_splits.get("test", [])) != len(weight_audit["subgroup_membership"]["test"]):
        raise ValueError("weight-audit test membership disagrees with prompt splits")
    dose_kl = json.loads(Path(args.x71_dose_kl).read_text())
    available = {float(row["dose"]) for row in dose_kl["measurements"]}
    if not set(TRUE_REFERENCE_DOSES).issubset(available):
        raise ValueError("x71 dose-KL artifact lacks one or more frozen anchors")

    code_records = {
        path.name: file_record(path)
        for path in sorted(HERE.glob("x75_*.py"))
    }
    config["experiment_name"] = "x75_boundary_weighted_refusal_edit"
    config["x75"] = {
        "schema_version": 1,
        "design_frozen": True,
        "number_claimed": "2026-08-20",
        "arms": list(ARMS),
        "target": "r = refusal_count / 10",
        "weight_formulas": WEIGHT_FORMULAS,
        "weight_normalization": "divide by full-split raw-weight mean; preserve exact zeros",
        "weighted_standardization": "train-only weighted population mean/std",
        "reference_true_doses": list(TRUE_REFERENCE_DOSES),
        "reference_control_doses": list(CONTROL_REFERENCE_DOSES),
        "calibration": {
            "absolute_tolerance": 5e-5,
            "relative_tolerance": 0.02,
            "max_iterations": 14,
            "max_abs_dose": 16.0,
            "bracket_growth": 2.0,
            "monotonic_slack": 1e-6,
            "primary_axis": "x71 sign-specific realized benign-validation KL",
        },
        "primary_endpoint": {
            "reference_doses": [-1.0, 1.0],
            "effect": "0.5 * (refusal_rate_positive - refusal_rate_negative)",
            "bootstrap": 5000,
            "bootstrap_seed": 20260819,
            "family_alpha": 0.05,
            "multiplicity": "Holm over three arm-vs-x71 tests",
        },
        "capability_reference_doses": [-3.0, 3.0],
        "capability_chat_date": _capability_date(args.x70_input, args.capability_chat_date),
        "subgroup_membership_test": weight_audit["subgroup_membership"]["test"],
        "subgroup_counts": {
            split: weight_audit["splits"][split]["subgroups"]
            for split in ("train", "validation", "test")
        },
        "artifact_paths": {
            "weight_audit": "data/runs/x75/x75_weight_audit.json",
            "probes": "data/runs/x75/probes/x75_{arm}_{true,control}.pt",
            "edits": "data/runs/x75/edits/x75_{arm}_{true,control}.pt",
            "dose_kl": "data/runs/x75/dose_kl/x75_{arm}_{true,control}.json",
            "generations": "data/generations/x75_{arm}_{true,control}_{anchor}.json",
            "scores": "data/runs/x75/scores/x75_{arm}_{true,control}_{anchor}.json",
            "refusal_analysis": "data/results/x75_refusal_analysis.json",
            "capability_analysis": "data/results/x75_capability_analysis.json",
        },
        "inherited_inputs": {
            "x71_config": file_record(args.x71_config),
            "probe_dataset": file_record(args.dataset),
            "prompt_splits": file_record(args.prompt_splits),
            "refusal_labels": file_record(args.refusal_labels),
            "weight_audit": file_record(args.weight_audit),
            "x71_probe": file_record(args.x71_probe),
            "x71_adapter": file_record(args.x71_adapter),
            "x71_dose_kl": file_record(args.x71_dose_kl),
            "x71_base_score": file_record(args.x71_base_score),
            "x71_true_scores": _score_doses(args.x71_true_score),
            "x70_inputs": [file_record(path) for path in args.x70_input],
        },
        "code": code_records,
    }
    atomic_json_dump(config, args.out)
    print(json.dumps({
        "out": str(Path(args.out).resolve()),
        "arms": config["x75"]["arms"],
        "capability_chat_date": config["x75"]["capability_chat_date"],
        "n_code_files": len(code_records),
    }, indent=2))


if __name__ == "__main__":
    main()
