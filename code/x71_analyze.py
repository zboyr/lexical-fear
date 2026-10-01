"""Paired prompt-level analysis of refusal probability across signed doses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

from edit_common import atomic_json_dump, file_record  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        help="label=path; repeat for base, true doses, and control doses",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def load_condition(spec: str) -> dict:
    label, path = spec.split("=", 1)
    obj = json.loads(Path(path).read_text())
    return {
        "label": label,
        "path": path,
        "kind": obj["kind"],
        "dose": float(obj["dose"]),
        "rates": {int(row["prompt_id"]): float(row["rate"]) for row in obj["rows"]},
        "mean": float(obj["mean_refusal_rate"]),
    }


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def fixed_effect_slope(matrix: np.ndarray, doses: np.ndarray) -> float:
    x = doses - doses.mean()
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return float((centered * x).sum() / (matrix.shape[0] * np.square(x).sum()))


def main() -> None:
    args = parse_args()
    conditions = [load_condition(spec) for spec in args.condition]
    base = [condition for condition in conditions if condition["kind"] == "base"]
    if len(base) != 1 or base[0]["dose"] != 0:
        raise ValueError("exactly one dose-0 base score file is required")
    base = base[0]
    ids = sorted(base["rates"])
    for condition in conditions:
        if sorted(condition["rates"]) != ids:
            raise ValueError(f"prompt ids differ for condition {condition['label']}")

    rng = np.random.default_rng(args.seed)
    n = len(ids)
    bootstrap_indices = rng.integers(0, n, size=(args.bootstrap, n))
    base_rates = np.asarray([base["rates"][prompt_id] for prompt_id in ids])
    paired = []
    for condition in conditions:
        rates = np.asarray([condition["rates"][prompt_id] for prompt_id in ids])
        delta = rates - base_rates
        boot = delta[bootstrap_indices].mean(axis=1)
        paired.append(
            {
                "label": condition["label"],
                "kind": condition["kind"],
                "dose": condition["dose"],
                "mean_refusal_rate": float(rates.mean()),
                "mean_paired_delta_vs_historical_base": float(delta.mean()),
                "paired_delta_ci95": percentile_interval(boot),
            }
        )

    slopes = {}
    for kind in ("true", "control"):
        subset = [base] + sorted(
            [condition for condition in conditions if condition["kind"] == kind],
            key=lambda item: item["dose"],
        )
        if len(subset) < 3:
            continue
        doses = np.asarray([condition["dose"] for condition in subset], dtype=np.float64)
        matrix = np.asarray(
            [[condition["rates"][prompt_id] for condition in subset] for prompt_id in ids],
            dtype=np.float64,
        )
        slope = fixed_effect_slope(matrix, doses)
        boot = np.asarray(
            [fixed_effect_slope(matrix[index], doses) for index in bootstrap_indices]
        )
        slopes[kind] = {
            "doses": doses.tolist(),
            "prompt_fixed_effect_slope": slope,
            "ci95": percentile_interval(boot),
        }

    result = {
        "schema_version": 1,
        "n_prompts": n,
        "rubric": "x58 rx_600 applied identically to historical and edited outputs",
        "base_is_historical_x63": True,
        "new_alpha_zero_generation_performed": False,
        "paired_conditions": paired,
        "dose_slopes": slopes,
        "inputs": [file_record(condition["path"]) for condition in conditions],
    }
    atomic_json_dump(result, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

