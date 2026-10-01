"""Descriptive question-paired GSM8K deltas for x87 conditions (post-hoc).

Writes data/results/x87_gsm8k_posthoc.json. Never touches the x87 gate file
(x87_external_safety_utility.json); nothing here can pass or fail a claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from x87_common import (
    ALL_CONDITIONS,
    ALPHA,
    RESULTS,
    RUN,
    atomic_json,
    file_record,
    paired_arrays,
    paired_bootstrap,
    read_json,
    stable_seed,
)

FILTERS = ("strict-match", "flexible-extract")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--out", default=str(RESULTS / "x87_gsm8k_posthoc.json"))
    return parser.parse_args()


def sample_metric(sample: dict) -> float:
    for container in (sample, sample.get("metrics") or {}):
        if "exact_match" in container:
            return float(container["exact_match"])
    raise KeyError(f"GSM8K sample lacks exact_match: {sorted(sample)}")


def condition_scores(path: Path) -> tuple[dict[str, dict[str, float]], dict]:
    payload = read_json(path)
    scores: dict[str, dict[str, float]] = {name: {} for name in FILTERS}
    for sample in payload["samples"]:
        filter_name = sample.get("filter")
        if filter_name not in scores:
            raise ValueError(f"unexpected GSM8K filter {filter_name!r} in {path}")
        doc_id = str(sample["doc_id"])
        if doc_id in scores[filter_name]:
            raise ValueError(f"duplicate doc {doc_id} for {filter_name} in {path}")
        scores[filter_name][doc_id] = sample_metric(sample)
    aggregates = payload["results"]["gsm8k"]
    for filter_name, per_doc in scores.items():
        if not per_doc:
            raise ValueError(f"no samples for {filter_name} in {path}")
        reported = float(aggregates[f"exact_match,{filter_name}"])
        recomputed = sum(per_doc.values()) / len(per_doc)
        if abs(reported - recomputed) > 1e-9:
            raise ValueError(
                f"per-doc mean {recomputed} != reported {reported} for {filter_name} in {path}"
            )
    return scores, payload


def main() -> None:
    args = parse_args()
    run = Path(args.run_dir)
    scores, inputs = {}, {}
    for condition in ALL_CONDITIONS:
        path = run / f"x87_gsm8k_{condition}.json"
        scores[condition], payload = condition_scores(path)
        inputs[condition] = {
            "file": file_record(path),
            "post_hoc": bool(payload.get("post_hoc")),
            "lm_eval_version": payload.get("lm_eval_version"),
        }
    comparisons = {}
    for condition in ALL_CONDITIONS[1:]:
        comparisons[condition] = {
            filter_name: paired_bootstrap(
                *paired_arrays(scores["base"][filter_name], scores[condition][filter_name])[1:],
                alpha=ALPHA,
                seed=stable_seed("x87_gsm8k", condition, filter_name),
            )
            for filter_name in FILTERS
        }
    result = {
        "schema_version": 1,
        "experiment": "x87",
        "task": "gsm8k",
        "role": "post_hoc_capability_extension",
        "post_hoc": True,
        "claimable": False,
        "note": (
            "GSM8K was not pre-registered in the x87 plan. These are descriptive "
            "question-paired deltas with uncorrected 95% intervals; no margin, no gate."
        ),
        "estimand": "edited minus base exact_match; question-paired",
        "alpha": ALPHA,
        "base_accuracy": {
            filter_name: float(
                sum(scores["base"][filter_name].values()) / len(scores["base"][filter_name])
            )
            for filter_name in FILTERS
        },
        "comparisons": comparisons,
        "inputs": inputs,
    }
    atomic_json(args.out, result)
    summary = {
        condition: {
            filter_name: {
                "mean_delta": row[filter_name]["mean_delta"],
                "two_sided_ci": row[filter_name]["two_sided_ci"],
            }
            for filter_name in FILTERS
        }
        for condition, row in comparisons.items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
