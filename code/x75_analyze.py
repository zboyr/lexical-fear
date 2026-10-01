"""Prompt-paired analysis of x75 KL-matched refusal edits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

from edit_common import atomic_json_dump, file_record  # noqa: E402
from x71_analyze import fixed_effect_slope  # noqa: E402
from x75_weighting import ARMS, subgroup_masks  # noqa: E402


TRUE_DOSES = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)
CONTROL_DOSES = (-3.0, -1.0, 1.0, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Frozen x71 historical-base score")
    parser.add_argument("--x71-score", action="append", required=True)
    parser.add_argument("--x75-score", action="append", required=True)
    parser.add_argument("--prompt-splits", required=True)
    parser.add_argument(
        "--x71-probe", default="data/runs/x71/probe/x71_probe_true.pt"
    )
    parser.add_argument("--x75-probe", action="append", required=True)
    parser.add_argument(
        "--dose-kl",
        action="append",
        required=True,
        help="Every x75 calibration artifact; binds the analyzed condition set",
    )
    parser.add_argument("--deviations", default="data/runs/x75/x75_deviations.json")
    parser.add_argument("--out", default="data/results/x75_refusal_analysis.json")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def load_rates(path: str) -> tuple[dict, dict[int, float]]:
    obj = json.loads(Path(path).read_text())
    rates = {int(row["prompt_id"]): float(row["rate"]) for row in obj["rows"]}
    if len(rates) != int(obj["n_prompts"]):
        raise ValueError(f"duplicate prompt ids in {path}")
    return obj, rates


def interval(values: np.ndarray, alpha: float = 0.05) -> list[float]:
    return [float(x) for x in np.quantile(values, [alpha / 2, 1 - alpha / 2])]


def one_sided_p_greater(bootstrap_values: np.ndarray) -> float:
    return float((1 + np.count_nonzero(bootstrap_values <= 0)) / (len(bootstrap_values) + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def symmetric_effect(negative: np.ndarray, positive: np.ndarray) -> np.ndarray:
    if negative.shape != positive.shape:
        raise ValueError("signed condition arrays differ in shape")
    return 0.5 * (positive - negative)


def probe_cosines(x71_path: str, x75_paths: list[str]) -> dict:
    x71 = torch.load(x71_path, map_location="cpu", weights_only=False)["q_unit"].float()
    probes = {}
    inputs = [file_record(x71_path)]
    for path in x75_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        key = (obj["arm"], obj["kind"])
        if key in probes:
            raise ValueError(f"duplicate x75 probe artifact for {key}")
        probes[key] = obj["q_unit"].float()
        inputs.append(file_record(path))
    out = {}
    for arm in ARMS:
        true = probes[(arm, "true")]
        control = probes[(arm, "control")]
        out[arm] = {
            "true_vs_x71": float(torch.dot(true, x71) / (true.norm() * x71.norm())),
            "true_vs_control": float(
                torch.dot(true, control) / (true.norm() * control.norm())
            ),
        }
    return {"cosines": out, "inputs": inputs}


def main() -> None:
    args = parse_args()
    base_obj, base_map = load_rates(args.base)
    if base_obj.get("kind") != "base" or float(base_obj.get("dose")) != 0:
        raise ValueError("--base must be the historical x71 dose-zero score")
    ids = sorted(base_map)
    base = np.asarray([base_map[prompt_id] for prompt_id in ids], dtype=np.float64)

    x71 = {}
    input_records = [file_record(args.base)]
    for path in args.x71_score:
        obj, mapping = load_rates(path)
        if obj.get("kind") != "true":
            continue
        dose = float(obj["dose"])
        if dose in x71:
            raise ValueError(f"duplicate x71 true dose {dose}")
        if sorted(mapping) != ids:
            raise ValueError(f"x71 prompt ids differ in {path}")
        x71[dose] = np.asarray([mapping[prompt_id] for prompt_id in ids], dtype=np.float64)
        input_records.append(file_record(path))
    if set(x71) != set(TRUE_DOSES):
        raise ValueError(f"x71 true scores must cover {TRUE_DOSES}; got {sorted(x71)}")

    x75 = {}
    score_metadata = {}
    for path in args.x75_score:
        obj, mapping = load_rates(path)
        key = (obj["arm"], obj["kind"], float(obj["reference_dose"]))
        if key in x75:
            raise ValueError(f"duplicate x75 score {key}")
        if sorted(mapping) != ids:
            raise ValueError(f"x75 prompt ids differ in {path}")
        if abs(float(obj["achieved_kl"]) - float(obj["target_kl"])) > float(obj["kl_tolerance"]):
            raise ValueError(f"x75 score {key} does not meet its KL tolerance")
        x75[key] = np.asarray([mapping[prompt_id] for prompt_id in ids], dtype=np.float64)
        score_metadata[key] = {
            "path": path,
            "actual_dose": float(obj["actual_dose"]),
            "target_kl": float(obj["target_kl"]),
            "achieved_kl": float(obj["achieved_kl"]),
            "kl_tolerance": float(obj["kl_tolerance"]),
            "rank_one_delta_frobenius_norm": float(obj["rank_one_delta_frobenius_norm"]),
            "mean_refusal_rate": float(obj["mean_refusal_rate"]),
            "generation_integrity": obj["generation_integrity"],
        }
        input_records.append(file_record(path))
    # Deviation D3: calibration decides the condition set.  Every matched
    # anchor must have exactly one score; a failed anchor must have none.
    calibration_status = {}
    for path in args.dose_kl:
        obj = json.loads(Path(path).read_text())
        if obj.get("experiment") != "x75":
            raise ValueError(f"not an x75 calibration artifact: {path}")
        arm, kind = obj["arm"], obj["kind"]
        if (arm, kind) in calibration_status:
            raise ValueError(f"duplicate calibration artifact for {arm}/{kind}")
        calibration_status[(arm, kind)] = {
            float(row["reference_dose"]): row["status"] for row in obj["calibrations"]
        }
        input_records.append(file_record(path))
    for arm in ARMS:
        for kind, doses in (("true", TRUE_DOSES), ("control", CONTROL_DOSES)):
            if (arm, kind) not in calibration_status:
                raise ValueError(f"missing calibration artifact for {arm}/{kind}")
            for dose in doses:
                status = calibration_status[(arm, kind)].get(dose)
                if status is None:
                    raise ValueError(f"calibration for {arm}/{kind} lacks anchor {dose}")
                if status == "matched" and (arm, kind, dose) not in x75:
                    raise ValueError(f"missing score for matched {arm}/{kind} anchor {dose}")
                if status != "matched" and (arm, kind, dose) in x75:
                    raise ValueError(f"score exists for unmatched {arm}/{kind} anchor {dose}")

    prompt_splits = json.loads(Path(args.prompt_splits).read_text())
    test_rows = {int(row["prompt_id"]): row for row in prompt_splits["test"]}
    if sorted(test_rows) != ids:
        raise ValueError("prompt_splits test ids differ from score ids")
    historical_r = np.asarray(
        [
            float(test_rows[prompt_id]["refusal_count"])
            / float(test_rows[prompt_id]["num_responses"])
            for prompt_id in ids
        ],
        dtype=np.float64,
    )
    masks = subgroup_masks(historical_r)
    input_records.append(file_record(args.prompt_splits))

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(0, len(ids), size=(args.bootstrap, len(ids)))
    x71_low = symmetric_effect(x71[-1.0], x71[1.0])
    has_primary = {
        arm: (arm, "true", -1.0) in x75 and (arm, "true", 1.0) in x75 for arm in ARMS
    }
    has_control_pair = {
        arm: (arm, "control", -1.0) in x75 and (arm, "control", 1.0) in x75
        for arm in ARMS
    }
    primary = []
    primary_bootstraps = []
    for arm in ARMS:
        # Deviation D3: an arm whose signed +/-1 anchors failed KL calibration
        # cannot show the primary effect; it stays in the Holm family at p=1.
        if not has_primary[arm]:
            primary_bootstraps.append(None)
            primary.append(
                {
                    "arm": arm,
                    "kl_match_passed": False,
                    "calibration_failed": True,
                    "difference_vs_x71_p_one_sided": 1.0,
                }
            )
            continue
        arm_effect = symmetric_effect(x75[(arm, "true", -1.0)], x75[(arm, "true", 1.0)])
        vs_x71 = arm_effect - x71_low
        boot_x71 = vs_x71[bootstrap_indices].mean(axis=1)
        primary_bootstraps.append(boot_x71)
        row = {
            "arm": arm,
            "mean_effect": float(arm_effect.mean()),
            "x71_mean_effect": float(x71_low.mean()),
            "difference_vs_x71": float(vs_x71.mean()),
            "difference_vs_x71_ci95": interval(boot_x71),
            "difference_vs_x71_p_one_sided": one_sided_p_greater(boot_x71),
            "kl_match_passed": True,
            "calibration_failed": False,
        }
        if has_control_pair[arm]:
            control_effect = symmetric_effect(
                x75[(arm, "control", -1.0)], x75[(arm, "control", 1.0)]
            )
            vs_control = arm_effect - control_effect
            boot_control = vs_control[bootstrap_indices].mean(axis=1)
            row["control_mean_effect"] = float(control_effect.mean())
            row["difference_vs_control"] = float(vs_control.mean())
            row["difference_vs_control_ci95"] = interval(boot_control)
        else:
            row["control_pair_calibration_failed"] = True
        primary.append(row)

    adjusted = holm_adjust([row["difference_vs_x71_p_one_sided"] for row in primary])
    family_alpha = 0.05 / len(primary)
    for row, adjusted_p, boot in zip(primary, adjusted, primary_bootstraps):
        row["holm_adjusted_p"] = adjusted_p
        if boot is None:
            row["edit_efficiency_gate"] = False
            continue
        row["bonferroni_familywise_ci95"] = interval(boot, alpha=family_alpha)
        row["edit_efficiency_gate"] = bool(
            adjusted_p < 0.05
            and row["bonferroni_familywise_ci95"][0] > 0
            and "difference_vs_control_ci95" in row
            and row["difference_vs_control_ci95"][0] > 0
        )

    subgroup_results = {}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"predeclared subgroup {name} is empty")
        block = {"n_prompts": int(mask.sum()), "x71_mean_effect": float(x71_low[mask].mean()), "arms": {}}
        subgroup_boot = rng.integers(0, int(mask.sum()), size=(args.bootstrap, int(mask.sum())))
        for arm in ARMS:
            if not has_primary[arm]:
                block["arms"][arm] = {"calibration_failed": True}
                continue
            arm_effect = symmetric_effect(
                x75[(arm, "true", -1.0)][mask], x75[(arm, "true", 1.0)][mask]
            )
            difference = arm_effect - x71_low[mask]
            entry = {
                "mean_effect": float(arm_effect.mean()),
                "difference_vs_x71": float(difference.mean()),
                "difference_vs_x71_ci95": interval(difference[subgroup_boot].mean(axis=1)),
            }
            if has_control_pair[arm]:
                control_effect = symmetric_effect(
                    x75[(arm, "control", -1.0)][mask],
                    x75[(arm, "control", 1.0)][mask],
                )
                entry["control_mean_effect"] = float(control_effect.mean())
            block["arms"][arm] = entry
        subgroup_results[name] = block

    paired_conditions = []
    for key, rates in sorted(x75.items()):
        delta = rates - base
        boot = delta[bootstrap_indices].mean(axis=1)
        paired_conditions.append(
            {
                "arm": key[0],
                "kind": key[1],
                "reference_dose": key[2],
                **score_metadata[key],
                "mean_paired_delta_vs_historical_base": float(delta.mean()),
                "paired_delta_ci95": interval(boot),
                "effect_per_delta_frobenius_norm": float(
                    delta.mean() / score_metadata[key]["rank_one_delta_frobenius_norm"]
                ),
            }
        )

    secondary = {}
    reference_doses = np.asarray([*TRUE_DOSES[:4], 0.0, *TRUE_DOSES[4:]], dtype=np.float64)
    high_x71 = symmetric_effect(x71[-3.0], x71[3.0])
    for arm in ARMS:
        entry = {}
        # Full dose-ladder statistics need every matched true anchor (D3).
        if all((arm, "true", dose) in x75 for dose in TRUE_DOSES):
            x75_matrix = np.stack(
                [x75[(arm, "true", dose)] for dose in TRUE_DOSES[:4]]
                + [base]
                + [x75[(arm, "true", dose)] for dose in TRUE_DOSES[4:]],
                axis=1,
            )
            x71_matrix = np.stack(
                [x71[dose] for dose in TRUE_DOSES[:4]]
                + [base]
                + [x71[dose] for dose in TRUE_DOSES[4:]],
                axis=1,
            )
            entry.update(
                {
                    "monotone_mean_refusal_over_reference_doses": bool(
                        np.all(np.diff(x75_matrix.mean(axis=0)) >= 0)
                    ),
                    "reference_coordinate_slope": fixed_effect_slope(
                        x75_matrix, reference_doses
                    ),
                    "x71_reference_coordinate_slope": fixed_effect_slope(
                        x71_matrix, reference_doses
                    ),
                }
            )
        else:
            entry["dose_ladder_incomplete"] = True
        if (arm, "true", -3.0) in x75 and (arm, "true", 3.0) in x75:
            high_effect = symmetric_effect(
                x75[(arm, "true", -3.0)], x75[(arm, "true", 3.0)]
            )
            entry.update(
                {
                    "high_kl_symmetric_effect": float(high_effect.mean()),
                    "high_kl_difference_vs_x71": float((high_effect - high_x71).mean()),
                    "high_kl_difference_vs_x71_ci95": interval(
                        (high_effect - high_x71)[bootstrap_indices].mean(axis=1)
                    ),
                }
            )
        else:
            entry["high_kl_pair_calibration_failed"] = True
        secondary[arm] = entry

    cosine_block = probe_cosines(args.x71_probe, args.x75_probe)
    input_records.extend(cosine_block["inputs"])
    if Path(args.deviations).is_file():
        input_records.append(file_record(args.deviations))
    result = {
        "calibration_status": {
            f"{arm}_{kind}": {
                str(dose): status for dose, status in sorted(statuses.items())
            }
            for (arm, kind), statuses in sorted(calibration_status.items())
        },
        "arms_with_matched_primary_anchors": [
            arm for arm in ARMS if has_primary[arm]
        ],
        "schema_version": 1,
        "experiment": "x75",
        "n_prompts": len(ids),
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.seed,
        "base_is_historical_x63": True,
        "new_alpha_zero_generation_performed": False,
        "comparison_axis": "x71 sign-specific realized benign-KL anchors",
        "primary_low_kl": primary,
        "any_arm_passes_edit_efficiency_gate": any(
            row["edit_efficiency_gate"] for row in primary
        ),
        "subgroups": subgroup_results,
        "paired_conditions": paired_conditions,
        "secondary": secondary,
        "probe_cosines": cosine_block["cosines"],
        "inputs": input_records,
    }
    atomic_json_dump(result, args.out)
    print(json.dumps({"primary_low_kl": primary, "secondary": secondary}, indent=2))


if __name__ == "__main__":
    main()
