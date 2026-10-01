"""x81 post-hoc (declared before any O contrast, see x81_posthoc_declaration.json):

Recompute the five contrasts after excluding admitted prompts whose frozen C
replacement is a sub-word token fragment. Secondary and exploratory; the
frozen cohort, the locked analyzer, and the primary endpoint are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from x81_analyze import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    load_generations,
    load_judgments,
    point_statistics,
    summarize_estimate,
    two_stage_bootstrap,
)
from x81_common import N_RESPONSES, RESULTS, RUN, atomic_json, file_record, load_historical_refusals, read_json


def main() -> None:
    declaration_path = RUN / "x81_posthoc_declaration.json"
    declaration = read_json(declaration_path)
    arms_manifest = read_json(RUN / "x81_arms.json")
    arms = sorted(arms_manifest["arms"], key=lambda row: int(row["id"]))
    arms_by_id = {int(row["id"]): row for row in arms}
    arms_sha = arms_manifest["arms_sha256"]
    fragments = set(declaration["fragment_replacements"])
    excluded = sorted(int(row["id"]) for row in arms if row["c_replacement"] in fragments)
    if excluded != sorted(declaration["excluded_prompt_ids"]):
        raise SystemExit("FATAL: excluded set differs from the declaration")

    generations = {
        c: load_generations(RUN / f"x81_generations_{c}.json", c, arms_by_id, arms_sha) for c in "ABC"
    }
    labels_new = {
        c: load_judgments(RUN / f"x81_refusal_{c}.jsonl", c, arms_by_id, generations[c], arms_sha)
        for c in "ABC"
    }
    historical = load_historical_refusals([int(row["id"]) for row in arms])
    keep = [row for row in arms if int(row["id"]) not in set(excluded)]
    labels = np.empty((len(keep), 4, N_RESPONSES), dtype=np.int8)
    for index, row in enumerate(keep):
        prompt_id = int(row["id"])
        labels[index, 0] = historical[prompt_id]
        for ci, c in enumerate("ABC", start=1):
            labels[index, ci] = labels_new[c][prompt_id]
    points = point_statistics(labels)
    boots = two_stage_bootstrap(labels, N_BOOTSTRAP, BOOTSTRAP_SEED + 4000)
    endpoints = {name: summarize_estimate(value, boots[name]) for name, value in points.items()}
    out = {
        "experiment": "x81",
        "analysis": "posthoc_fragment_sensitivity",
        "status": "secondary_exploratory_not_preregistered",
        "declaration": file_record(declaration_path),
        "excluded_prompt_ids": excluded,
        "n_excluded": len(excluded),
        "n_kept": len(keep),
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED + 4000},
        "endpoints": endpoints,
        "excluded_rows_only": None,
    }
    if excluded:
        sub = np.empty((len(excluded), 4, N_RESPONSES), dtype=np.int8)
        for index, prompt_id in enumerate(excluded):
            sub[index, 0] = historical[prompt_id]
            for ci, c in enumerate("ABC", start=1):
                sub[index, ci] = labels_new[c][prompt_id]
        out["excluded_rows_only"] = {
            "n": len(excluded),
            **{k: float(v) for k, v in point_statistics(sub).items()},
        }
    path = RESULTS / "x81_posthoc_fragment_sensitivity.json"
    atomic_json(path, out)
    print(json.dumps({k: endpoints[k]["estimate"] for k in endpoints}, indent=None))
    print(f"delta_C ci95 {endpoints['delta_C']['ci95']}  -> {path}")


if __name__ == "__main__":
    main()
