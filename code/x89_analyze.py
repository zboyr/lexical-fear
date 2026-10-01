"""Descriptive analysis for x89: positive boundary doses, shallow rank-one scales.

Reads x87 artifacts for base + the pre-existing conditions and x89 artifacts
for the seven new conditions. Verifies that re-running the x87 seed schedule
reproduces the stored x87 primary comparisons exactly (the appended-condition
guarantee), then writes data/results/x89_positive_dose_extension.json.
Never touches the x87 result files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from x87_common import (
    ALPHA,
    BOOTSTRAP_SEED,
    DATA,
    EDITED_CONDITIONS,
    EXTENSION_CONDITIONS,
    MARGINS,
    RESULTS,
    RUN,
    atomic_json,
    file_record,
    paired_arrays,
    paired_bootstrap,
    read_json,
    stable_seed,
)
from x87_analyze import ifeval_scores, indexed
from x87_gsm8k_analyze import FILTERS, condition_scores

X89_CONDITIONS = (
    "x86_p050", "x86_p100", "x86_p150",
    "x75_m1", "x75_m2", "x75_p1", "x75_p2",
)
X89_RUN = DATA / "runs" / "x89"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x87-run", default=str(RUN))
    parser.add_argument("--x89-run", default=str(X89_RUN))
    parser.add_argument(
        "--x87-results", default=str(RESULTS / "x87_external_safety_utility.json")
    )
    parser.add_argument("--out", default=str(RESULTS / "x89_positive_dose_extension.json"))
    parser.add_argument("--skip-gsm8k", action="store_true")
    return parser.parse_args()


def comparison(base, edited, *, alpha, seed):
    _, base_arr, edited_arr = paired_arrays(base, edited)
    return paired_bootstrap(base_arr, edited_arr, alpha=alpha, seed=seed)


def condition_paths(condition: str, x87_run: Path, x89_run: Path) -> dict[str, Path]:
    if condition in X89_CONDITIONS:
        return {
            "orbench": x89_run / f"x89_orbench_{condition}.json",
            "strongreject": x89_run / f"x89_strongreject_{condition}.json",
            "ifeval": x89_run / f"x89_ifeval_{condition}.json",
            "gsm8k": x89_run / f"x89_gsm8k_{condition}.json",
        }
    return {
        "orbench": x87_run / f"x87_orbench_{condition}.json",
        "strongreject": x87_run / f"x87_strongreject_{condition}.json",
        "ifeval": x87_run / f"x87_ifeval_{condition}.json",
        "gsm8k": x87_run / f"x87_gsm8k_{condition}.json",
    }


def main() -> None:
    args = parse_args()
    x87_run, x89_run = Path(args.x87_run), Path(args.x89_run)
    if X89_CONDITIONS != EXTENSION_CONDITIONS[-len(X89_CONDITIONS):]:
        raise SystemExit("FATAL: x89 conditions must be the tail of EXTENSION_CONDITIONS")

    schedule = tuple(EDITED_CONDITIONS) + EXTENSION_CONDITIONS
    or_hard, or_toxic, strong, ifeval, inputs = {}, {}, {}, {}, {}
    for condition in ("base",) + schedule:
        paths = condition_paths(condition, x87_run, x89_run)
        or_hard[condition] = indexed(paths["orbench"], "orbench_hard", "refusal")
        or_toxic[condition] = indexed(paths["orbench"], "orbench_toxic", "refusal")
        strong[condition] = indexed(paths["strongreject"], "strongreject", "score")
        ifeval[condition] = ifeval_scores(paths["ifeval"])
        inputs[condition] = [
            file_record(paths[k]) for k in ("orbench", "strongreject", "ifeval")
        ]

    superiority_alpha = ALPHA / len(EDITED_CONDITIONS)
    ni_alpha = ALPHA / (len(EDITED_CONDITIONS) * 3)
    all_rows = {}
    for index, condition in enumerate(schedule):
        hard = comparison(
            or_hard["base"], or_hard[condition],
            alpha=superiority_alpha, seed=BOOTSTRAP_SEED + 10 * index,
        )
        toxic = comparison(
            or_toxic["base"], or_toxic[condition],
            alpha=ni_alpha, seed=BOOTSTRAP_SEED + 10 * index + 1,
        )
        sr = comparison(
            strong["base"], strong[condition],
            alpha=ni_alpha, seed=BOOTSTRAP_SEED + 10 * index + 2,
        )
        capability = comparison(
            ifeval["base"], ifeval[condition],
            alpha=ni_alpha, seed=BOOTSTRAP_SEED + 10 * index + 3,
        )
        all_rows[condition] = {
            "orbench_hard_refusal": hard,
            "orbench_toxic_refusal": toxic,
            "strongreject_score": sr,
            "ifeval_prompt_strict": capability,
        }

    # Appended-condition guarantee: every pre-existing x87 comparison must
    # reproduce the stored x87 result file exactly.
    stored = read_json(args.x87_results)["comparisons"]
    for condition in schedule:
        if condition in X89_CONDITIONS:
            continue
        for endpoint, row in all_rows[condition].items():
            if json.dumps(row, sort_keys=True) != json.dumps(
                stored[condition][endpoint], sort_keys=True
            ):
                raise SystemExit(
                    f"FATAL: {condition}/{endpoint} does not reproduce the "
                    "stored x87 comparison; seed schedule broken"
                )

    gsm8k_rows, gsm8k_base = {}, None
    if not args.skip_gsm8k:
        base_scores, _ = condition_scores(x87_run / "x87_gsm8k_base.json")
        gsm8k_base = {
            f: sum(base_scores[f].values()) / len(base_scores[f]) for f in FILTERS
        }
        for condition in X89_CONDITIONS:
            path = condition_paths(condition, x87_run, x89_run)["gsm8k"]
            scores, _ = condition_scores(path)
            inputs[condition].append(file_record(path))
            gsm8k_rows[condition] = {
                filter_name: paired_bootstrap(
                    *paired_arrays(base_scores[filter_name], scores[filter_name])[1:],
                    alpha=ALPHA,
                    seed=stable_seed("x87_gsm8k", condition, filter_name),
                )
                for filter_name in FILTERS
            }

    comparisons = {}
    for condition in X89_CONDITIONS:
        row = dict(all_rows[condition])
        row["role"] = "post_hoc_dose_extension"
        row["post_hoc"] = True
        row["claim_pass"] = False
        if condition in gsm8k_rows:
            row["gsm8k"] = gsm8k_rows[condition]
        comparisons[condition] = row

    result = {
        "schema_version": 1,
        "experiment": "x89",
        "role": "post_hoc_dose_extension",
        "claimable": False,
        "note": (
            "Positive boundary-write doses and shallow rank-one scales on the "
            "frozen x87 Qwen3.5-4B snapshot; descriptive question/prompt-paired "
            "deltas, x87 seed schedule with conditions appended after all "
            "existing EXTENSION_CONDITIONS. Nothing here can pass or alter the "
            "x87 gates."
        ),
        "estimand": "edited minus base; prompt-paired",
        "alphas": {
            "hard_superiority_one_sided_alpha": superiority_alpha,
            "noninferiority_one_sided_alpha": ni_alpha,
            "gsm8k_two_sided_alpha": ALPHA,
        },
        "margins": MARGINS,
        "x87_reproduction_check": "passed",
        "base_rates": {
            "orbench_hard_refusal": sum(or_hard["base"].values()) / len(or_hard["base"]),
            "orbench_toxic_refusal": sum(or_toxic["base"].values()) / len(or_toxic["base"]),
            "strongreject_score": sum(strong["base"].values()) / len(strong["base"]),
            "ifeval_prompt_strict": sum(ifeval["base"].values()) / len(ifeval["base"]),
            "gsm8k": gsm8k_base,
        },
        "comparisons": comparisons,
        "inputs": {c: inputs[c] for c in ("base",) + X89_CONDITIONS},
    }
    atomic_json(args.out, result)
    summary = {
        condition: {
            endpoint: {
                "mean_delta": row[endpoint]["mean_delta"],
                "two_sided_ci": row[endpoint]["two_sided_ci"],
            }
            for endpoint in (
                "orbench_hard_refusal",
                "orbench_toxic_refusal",
                "strongreject_score",
                "ifeval_prompt_strict",
            )
        }
        for condition, row in comparisons.items()
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
