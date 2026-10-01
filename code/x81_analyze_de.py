"""x81 D/E analysis: instruction-based self-recovery contrasts.

Four conditions per admitted prompt:
- ``O`` historical refusal2 labels (frozen x63/x66; carries a run-era caveat);
- ``P`` plain original prompt, regenerated now (contemporaneous baseline);
- ``D`` generic "a word was inserted, ignore it" note (word unnamed, in place);
- ``E`` named note, position aware.

Primary, confound-free endpoints are contemporaneous:
``delta_D = mean(r_P - r_D)`` and ``delta_E = mean(r_P - r_E)`` (positive means
the instruction removed refusal). ``run_era = mean(r_O - r_P)`` measures the
historical-vs-now offset directly, so the caveated ``r_O`` contrasts can be
read against it. Two-stage bootstrap identical to x81_analyze.py: resample
prompts, then resample the ten binary responses within each prompt-condition;
response indices are never paired across conditions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from x81_common import (
    LABELS_PATH,
    N_RESPONSES,
    RESULTS,
    RUN,
    atomic_json,
    file_record,
    load_historical_refusals,
    read_json,
    sha256_text,
)

CONDITIONS = ("O", "P", "D", "E")
NEW_CONDITIONS = ("P", "D", "E")
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

# (name, plus_index, minus_index) over CONDITIONS; contrast = r_plus - r_minus.
CONTRASTS = (
    ("delta_D", 1, 2),      # r_P - r_D  contemporaneous, generic note
    ("delta_E", 1, 3),      # r_P - r_E  contemporaneous, named note
    ("d_vs_e", 2, 3),       # r_D - r_E  does naming the word matter
    ("run_era", 0, 1),      # r_O - r_P  historical vs contemporaneous baseline
    ("delta_D_hist", 0, 2), # r_O - r_D  caveated
    ("delta_E_hist", 0, 3), # r_O - r_E  caveated
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=str(RUN / "x81_arms_de.json"))
    for condition in NEW_CONDITIONS:
        parser.add_argument(
            f"--generation-{condition.lower()}",
            default=str(RUN / f"x81_generations_{condition}.json"),
        )
        parser.add_argument(
            f"--refusal-{condition.lower()}",
            default=str(RUN / f"x81_refusal_{condition}.jsonl"),
        )
    parser.add_argument("--out", default=str(RESULTS / "x81_de_self_recovery.json"))
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def load_generations(path: Path, condition: str, arms_by_id: dict[int, dict], arms_sha: str):
    obj = read_json(path)
    if obj.get("condition") != condition or obj.get("arms_sha256") != arms_sha:
        raise SystemExit(f"FATAL: invalid {condition} generation manifest")
    result = {}
    for row in obj.get("rows", []):
        prompt_id = int(row["id"])
        if prompt_id not in arms_by_id:
            continue
        responses = row.get("llm_responses", [])
        response_hashes = row.get("response_sha256", [])
        if (
            row.get("condition") != condition
            or row.get("prompt_sha256") != arms_by_id[prompt_id]["prompt_sha256"][condition]
            or len(responses) != N_RESPONSES
            or len(response_hashes) != N_RESPONSES
            or response_hashes != [sha256_text(value) for value in responses]
            or any(not isinstance(value, str) or not value.strip() for value in responses)
        ):
            raise SystemExit(f"FATAL: invalid {condition} generation row {prompt_id}")
        if prompt_id in result:
            raise SystemExit(f"FATAL: duplicate {condition} generation row {prompt_id}")
        result[prompt_id] = row
    missing = sorted(set(arms_by_id) - set(result))
    if missing:
        raise SystemExit(f"FATAL: {condition} lacks generations for {missing[:10]}")
    return result


def load_judgments(path, condition, arms_by_id, generations, arms_sha):
    successful = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("failed"):
                continue
            prompt_id = int(row["id"])
            if prompt_id not in arms_by_id:
                continue
            arm = arms_by_id[prompt_id]
            values = row.get("refusal", [])
            if (
                row.get("condition") != condition
                or row.get("arms_sha256") != arms_sha
                or row.get("prompt_sha256") != arm["prompt_sha256"][condition]
                or row.get("response_sha256") != generations[prompt_id]["response_sha256"]
                or len(values) != N_RESPONSES
                or not all(isinstance(value, bool) for value in values)
            ):
                raise SystemExit(f"FATAL: invalid {condition} refusal record {prompt_id}")
            if prompt_id in successful:
                raise SystemExit(f"FATAL: duplicate {condition} refusal record {prompt_id}")
            successful[prompt_id] = np.asarray(values, dtype=np.int8)
    missing = sorted(set(arms_by_id) - set(successful))
    if missing:
        raise SystemExit(f"FATAL: {condition} lacks successful labels for {missing[:10]}")
    return successful


def point_contrasts(labels: np.ndarray) -> dict[str, float]:
    rates = labels.mean(axis=2)
    means = rates.mean(axis=0)
    result = {f"mean_{cond}": float(means[index]) for index, cond in enumerate(CONDITIONS)}
    for name, plus, minus in CONTRASTS:
        result[name] = float(np.mean(rates[:, plus] - rates[:, minus]))
    return result


def two_stage_bootstrap(labels: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, np.ndarray]:
    k = len(CONDITIONS)
    if labels.ndim != 3 or labels.shape[1:] != (k, N_RESPONSES):
        raise ValueError(f"labels must have shape (prompts, {k}, {N_RESPONSES})")
    n_prompts = labels.shape[0]
    if n_prompts == 0 or n_bootstrap < 1:
        raise ValueError("bootstrap requires prompts and positive replicate count")
    empirical_rates = labels.mean(axis=2)
    names = [f"mean_{cond}" for cond in CONDITIONS] + [name for name, _, _ in CONTRASTS]
    output = {name: np.empty(n_bootstrap, dtype=np.float64) for name in names}
    rng = np.random.default_rng(seed)
    chunk_size = 250
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        count = stop - start
        prompt_indices = rng.integers(0, n_prompts, size=(count, n_prompts))
        selected = empirical_rates[prompt_indices]
        resampled = rng.binomial(N_RESPONSES, selected) / N_RESPONSES
        means = resampled.mean(axis=1)
        for index, cond in enumerate(CONDITIONS):
            output[f"mean_{cond}"][start:stop] = means[:, index]
        for name, plus, minus in CONTRASTS:
            output[name][start:stop] = (resampled[:, :, plus] - resampled[:, :, minus]).mean(axis=1)
    return output


def summarize(point: float, samples: np.ndarray) -> dict:
    ci90 = np.quantile(samples, (0.05, 0.95)).tolist()
    ci95 = np.quantile(samples, (0.025, 0.975)).tolist()
    p_two = min(1.0, 2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))))
    return {
        "estimate": float(point),
        "ci90": [float(v) for v in ci90],
        "ci95": [float(v) for v in ci95],
        "bootstrap_p_two_sided": p_two,
    }


def subset_contrast(labels, mask, name, seed, n_bootstrap):
    subset = labels[mask]
    if not len(subset):
        return {"n": 0}
    rates = subset.mean(axis=2)
    plus, minus = next((p, m) for nm, p, m in CONTRASTS if nm == name)
    point = float(np.mean(rates[:, plus] - rates[:, minus]))
    samples = two_stage_bootstrap(subset, n_bootstrap, seed)[name]
    return {"n": int(len(subset)), **summarize(point, samples)}


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise SystemExit("FATAL: --bootstrap must be positive")
    arms_path = Path(args.arms)
    manifest = read_json(arms_path)
    if manifest.get("status") != "pass":
        raise SystemExit("FATAL: x81 D/E arm manifest is not pass")
    arms = sorted(manifest["arms"], key=lambda row: int(row["id"]))
    arms_by_id = {int(row["id"]): row for row in arms}
    if len(arms_by_id) != len(arms):
        raise SystemExit("FATAL: duplicate admitted prompt IDs")
    prompt_ids = [int(row["id"]) for row in arms]
    arms_sha = manifest["arms_sha256"]

    generation_paths = {c: Path(getattr(args, f"generation_{c.lower()}")) for c in NEW_CONDITIONS}
    refusal_paths = {c: Path(getattr(args, f"refusal_{c.lower()}")) for c in NEW_CONDITIONS}
    generations = {
        c: load_generations(path, c, arms_by_id, arms_sha) for c, path in generation_paths.items()
    }
    new_labels = {
        c: load_judgments(refusal_paths[c], c, arms_by_id, generations[c], arms_sha)
        for c in NEW_CONDITIONS
    }
    historical = load_historical_refusals(prompt_ids)

    labels = np.empty((len(arms), len(CONDITIONS), N_RESPONSES), dtype=np.int8)
    for index, prompt_id in enumerate(prompt_ids):
        labels[index, 0] = historical[prompt_id]
        for cond_index, cond in enumerate(NEW_CONDITIONS, start=1):
            labels[index, cond_index] = new_labels[cond][prompt_id]

    points = point_contrasts(labels)
    bootstraps = two_stage_bootstrap(labels, args.bootstrap, args.seed)
    endpoints = {name: summarize(points[name], bootstraps[name]) for name in points}

    rates = labels.mean(axis=2)
    original_rates = rates[:, 0]  # historical r_O
    plain_rates = rates[:, 1]     # contemporaneous r_P

    # E position heterogeneity (end is only ~2 prompts: descriptive).
    position = np.asarray([row["position"] for row in arms])
    position_reports = {}
    for value in sorted(set(position.tolist())):
        mask = position == value
        position_reports[value] = {
            "delta_E": subset_contrast(labels, mask, "delta_E", args.seed + 10, args.bootstrap),
            "delta_D": subset_contrast(labels, mask, "delta_D", args.seed + 11, args.bootstrap),
        }

    # r_O strata (the cohort is heavily ceilinged).
    strata_masks = {
        "r0_eq_0": original_rates == 0,
        "r0_0.1_0.4": (original_rates > 0) & (original_rates < 0.5),
        "r0_0.5_0.9": (original_rates >= 0.5) & (original_rates < 1),
        "r0_eq_1": original_rates == 1,
    }
    strata = {
        name: {
            "delta_D": subset_contrast(labels, mask, "delta_D", args.seed + 2000 + i, args.bootstrap),
            "delta_E": subset_contrast(labels, mask, "delta_E", args.seed + 2100 + i, args.bootstrap),
        }
        for i, (name, mask) in enumerate(strata_masks.items())
    }

    delta_d = endpoints["delta_D"]
    delta_e = endpoints["delta_E"]

    def verdict(ep: dict) -> str:
        if ep["ci95"][0] > 0:
            return "removes_refusal"
        if ep["ci95"][1] < 0:
            return "adds_refusal"
        return "inconclusive"

    output = {
        "experiment": "x81",
        "stage": "de-analysis",
        "status": "complete",
        "scope": "Qwen3.5-4B frozen x81 cohort; instruction-based self-recovery (arms P/D/E)",
        "primary_endpoints": {
            "delta_D_contemporaneous": {**delta_d, "verdict": verdict(delta_d)},
            "delta_E_contemporaneous": {**delta_e, "verdict": verdict(delta_e)},
        },
        "historical_control_caveat": (
            "O was not regenerated; r_O contrasts include a run-era component. run_era = "
            "mean(r_O - r_P) estimates that offset directly; delta_D/delta_E use the "
            "contemporaneous P baseline and do not."
        ),
        "config": {
            "conditions": CONDITIONS,
            "n_bootstrap": args.bootstrap,
            "bootstrap_seed": args.seed,
            "bootstrap_unit": "prompt then ten binary responses independently within condition",
            "sign_convention": "positive contrast = refusal removed relative to the minus arm",
        },
        "inputs": {
            "arms": file_record(arms_path),
            "historical_labels": file_record(LABELS_PATH),
            **{f"generation_{c}": file_record(p) for c, p in generation_paths.items()},
            **{f"refusal_{c}": file_record(p) for c, p in refusal_paths.items()},
        },
        "counts": {
            "prompts": len(arms),
            "responses_per_prompt_condition": N_RESPONSES,
            "positions": manifest["counts"]["positions"],
        },
        "generation_integrity": {
            c: {
                "rows": len(rows),
                "retry_rows": sum(bool(r.get("used_retry")) for r in rows.values()),
                "truncated_after_retry": sum(
                    sum(bool(v) for v in r.get("truncated", [])) for r in rows.values()
                ),
            }
            for c, rows in generations.items()
        },
        "endpoints": endpoints,
        "original_rate_distribution_O": {
            f"{v / 10:.1f}": int(np.sum(original_rates == v / 10)) for v in range(11)
        },
        "plain_rate_distribution_P": {
            f"{v / 10:.1f}": int(np.sum(plain_rates == v / 10)) for v in range(11)
        },
        "e_position_heterogeneity": position_reports,
        "original_rate_strata": strata,
    }
    atomic_json(args.out, output)
    print(
        f"x81 D/E analysis: n={len(arms)} "
        f"delta_D={points['delta_D']:+.4f}({verdict(delta_d)}) "
        f"delta_E={points['delta_E']:+.4f}({verdict(delta_e)}) "
        f"run_era={points['run_era']:+.4f}"
    )
    print(f"result -> {args.out}")


if __name__ == "__main__":
    main()
