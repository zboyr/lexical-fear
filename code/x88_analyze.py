"""Descriptive per-model paired deltas for x88; imports the qwen row from x87.

Writes data/results/x88_cross_model_deep_dose.json. Exploratory only: no
margins, no gates, claimable false everywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from x88_common import (
    ALPHA,
    CONDITIONS,
    DATA,
    MODELS,
    RESULTS,
    RUN,
    atomic_json,
    file_record,
    paired_arrays,
    paired_bootstrap,
    read_json,
    stable_seed,
    tag_dir,
)

IFEVAL_KEYS = ("prompt_level_strict_acc", "prompt_level_strict_acc,none")
EDITED = tuple(c for c in CONDITIONS if c != "base")
# Extension conditions run without the lm_eval utility tasks (user scope):
SAFETY_ONLY = {"x75_m3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument(
        "--x87-results", default=str(RESULTS / "x87_external_safety_utility.json")
    )
    parser.add_argument("--x87-gsm8k", default=str(RESULTS / "x87_gsm8k_posthoc.json"))
    parser.add_argument("--out", default=str(RESULTS / "x88_cross_model_deep_dose.json"))
    return parser.parse_args()


def indexed(path: Path, benchmark: str, field: str) -> dict[str, float]:
    payload = read_json(path)
    return {
        row["id"]: float(row[field])
        for row in payload["rows"]
        if row["benchmark"] == benchmark
    }


def ifeval_scores(path: Path) -> dict[str, float]:
    payload = read_json(path)
    scores = {}
    for sample in payload["samples"]:
        value = None
        metrics = sample.get("metrics")
        containers = [sample] + ([metrics] if isinstance(metrics, dict) else [])
        for container in containers:
            for key in IFEVAL_KEYS:
                if key in container:
                    value = float(container[key])
        if value is None:
            raise KeyError(f"IFEval sample lacks strict prompt metric in {path}")
        scores[str(sample["doc_id"])] = value
    if not scores:
        raise ValueError(f"no IFEval samples in {path}")
    return scores


def gsm8k_scores(path: Path, filter_name: str) -> dict[str, float]:
    payload = read_json(path)
    scores = {}
    for sample in payload["samples"]:
        if sample.get("filter") != filter_name:
            continue
        value = sample.get("exact_match")
        if value is None:
            metrics = sample.get("metrics")
            if isinstance(metrics, dict):
                value = metrics.get("exact_match")
        if value is None:
            raise KeyError(f"GSM8K sample lacks exact_match in {path}")
        scores[str(sample["doc_id"])] = float(value)
    if not scores:
        raise ValueError(f"no GSM8K samples for {filter_name} in {path}")
    return scores


def comparison(base, edited, *, seed):
    _, base_arr, edited_arr = paired_arrays(base, edited)
    return paired_bootstrap(base_arr, edited_arr, alpha=ALPHA, seed=seed)


def analyze_tag(run: Path, tag: str) -> tuple[dict, list[dict]]:
    d = run / tag
    inputs = []
    endpoints: dict[str, dict[str, dict[str, float]]] = {}
    for condition in CONDITIONS:
        paths = {
            "orbench": d / f"x88_orbench_{condition}.json",
            "strongreject": d / f"x88_strongreject_{condition}.json",
        }
        if condition not in SAFETY_ONLY:
            paths["ifeval"] = d / f"x88_ifeval_{condition}.json"
            paths["gsm8k"] = d / f"x88_gsm8k_{condition}.json"
        for path in paths.values():
            inputs.append(file_record(path))
        endpoints.setdefault("orbench_hard_refusal", {})[condition] = indexed(
            paths["orbench"], "orbench_hard", "refusal"
        )
        endpoints.setdefault("orbench_toxic_refusal", {})[condition] = indexed(
            paths["orbench"], "orbench_toxic", "refusal"
        )
        endpoints.setdefault("strongreject_score", {})[condition] = indexed(
            paths["strongreject"], "strongreject", "score"
        )
        if condition not in SAFETY_ONLY:
            endpoints.setdefault("ifeval_prompt_strict", {})[condition] = ifeval_scores(
                paths["ifeval"]
            )
            endpoints.setdefault("gsm8k_strict", {})[condition] = gsm8k_scores(
                paths["gsm8k"], "strict-match"
            )
    base_rates = {
        name: sum(v["base"].values()) / len(v["base"]) for name, v in endpoints.items()
    }
    comparisons = {}
    for condition in EDITED:
        comparisons[condition] = {
            name: comparison(
                values["base"],
                values[condition],
                seed=stable_seed(tag, condition, name),
            )
            for name, values in endpoints.items()
            if condition in values
        }
    return {"base_rates": base_rates, "comparisons": comparisons}, inputs


def import_qwen(x87_results: Path, x87_gsm8k: Path) -> dict:
    x87 = read_json(x87_results)
    gsm = read_json(x87_gsm8k)
    key_map = {
        "orbench_hard_refusal": "orbench_hard_refusal",
        "orbench_toxic_refusal": "orbench_toxic_refusal",
        "strongreject_score": "strongreject_score",
        "ifeval_prompt_strict": "ifeval_prompt_strict",
    }
    comparisons = {}
    for condition in EDITED:
        row = x87["comparisons"][condition]
        comparisons[condition] = {
            ours: dict(row[theirs]) for ours, theirs in key_map.items()
        }
        comparisons[condition]["gsm8k_strict"] = dict(
            gsm["comparisons"][condition]["strict-match"]
        )
    base_rates = {
        name: comparisons[EDITED[0]][name]["base_mean"]
        for name in list(key_map) + ["gsm8k_strict"]
    }
    return {
        "base_rates": base_rates,
        "comparisons": comparisons,
        "imported_from_x87": True,
        "inputs": [file_record(x87_results), file_record(x87_gsm8k)],
    }


def main() -> None:
    args = parse_args()
    run = Path(args.run_dir)
    models = {}
    inputs = {}
    for tag in MODELS:
        models[tag], inputs[tag] = analyze_tag(run, tag)
    models["qwen"] = import_qwen(Path(args.x87_results), Path(args.x87_gsm8k))
    result = {
        "schema_version": 1,
        "experiment": "x88",
        "role": "exploratory_cross_model_extension",
        "claimable": False,
        "note": (
            "No pre-registration: descriptive question/prompt-paired deltas with "
            "uncorrected 95% intervals. The qwen row is imported from x87 "
            "(x86_m150/x75_m6 there equal x88's conditions on Qwen3.5-4B)."
        ),
        "estimand": "edited minus base; prompt-paired per model",
        "alpha": ALPHA,
        "conditions": list(EDITED),
        "models": models,
        "inputs": inputs,
    }
    atomic_json(args.out, result)
    summary = {
        tag: {
            condition: {
                name: round(block["mean_delta"], 4)
                for name, block in row["comparisons"][condition].items()
            }
            for condition in EDITED
        }
        for tag, row in models.items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
