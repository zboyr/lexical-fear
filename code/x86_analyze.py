"""x86 Stage 6: locked causal, specificity, integrity, compliance, and benign analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from x86_common import RESULTS, RUN, atomic_json, code_hashes, file_record, read_json

N_BOOTSTRAP = 10_000
SEED = 8610


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(RUN / "x86_manifest.json"))
    parser.add_argument("--calibration", default=str(RUN / "x86_kl_calibration.json"))
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(RESULTS / "x86_boundary_intervention.json"))
    parser.add_argument("--allow-code-drift", action="store_true", help="permit recorded post-freeze changes to judge/analyze (not the frozen pipeline)")
    return parser.parse_args()


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def interval(samples: np.ndarray) -> list[float]:
    finite = samples[np.isfinite(samples)]
    if not len(finite):
        return [None, None]
    return [float(value) for value in np.quantile(finite, (0.025, 0.975))]


def one_sided_p(samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    return float((1 + np.sum(finite <= 0)) / (len(finite) + 1))


def condition_paths(run_dir: Path, condition: str, method: str) -> tuple[Path, Path]:
    return (
        run_dir / f"x86_generations_{condition}.json",
        run_dir / f"x86_{method}_{condition}.jsonl",
    )


def load_condition(
    run_dir: Path,
    condition: str,
    method: str,
    expected_ids: list[int],
    manifest_sha: str,
    calibration_sha: str,
) -> tuple[np.ndarray, dict]:
    generation_path, judge_path = condition_paths(run_dir, condition, method)
    if not generation_path.exists() or not judge_path.exists():
        raise SystemExit(f"FATAL: missing {method} artifacts for {condition}")
    generation = read_json(generation_path)
    if (
        generation.get("experiment") != "x86"
        or generation.get("condition") != condition
        or generation.get("manifest", {}).get("sha256") != manifest_sha
        or generation.get("calibration", {}).get("sha256") != calibration_sha
    ):
        raise SystemExit(f"FATAL: {condition} generation lineage mismatch")
    rows = {int(row["id"]): row for row in generation["rows"]}
    if sorted(rows) != sorted(expected_ids):
        raise SystemExit(f"FATAL: {condition} generation ID coverage mismatch")
    field = "refusal" if method == "refusal2" else "complied"
    labels = {}
    with judge_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("failed"):
                continue
            row_id = int(record["id"])
            source = rows.get(row_id)
            if source is None:
                continue
            if (
                record.get("condition") != condition
                or record.get("method") != method
                or record.get("response_sha256") != source["response_sha256"]
                or record.get("run_hash") != source["run_hash"]
                or len(record.get(field, [])) != len(source["llm_responses"])
                or not all(isinstance(value, bool) for value in record[field])
            ):
                raise SystemExit(f"FATAL: invalid {method} row {condition}/{row_id}")
            if row_id in labels:
                raise SystemExit(f"FATAL: duplicate {method} row {condition}/{row_id}")
            labels[row_id] = np.asarray(record[field], dtype=np.int8)
    if sorted(labels) != sorted(expected_ids):
        missing = sorted(set(expected_ids) - set(labels))[:10]
        raise SystemExit(f"FATAL: incomplete {method} judgments for {condition}: {missing}")
    widths = {len(labels[row_id]) for row_id in expected_ids}
    if len(widths) != 1:
        raise SystemExit(f"FATAL: response count varies within {condition}")
    array = np.stack([labels[row_id] for row_id in expected_ids])
    return array, {
        "generation": file_record(generation_path),
        "judgment": file_record(judge_path),
        "rows": rows,
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return np.nan
    return float(np.sum(values * weights) / total)


def causal_metrics(rates: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, float]:
    wm = lambda value: weighted_mean(value, weights)
    suppress_half = wm(rates["H_0"] - rates["H_both_half"])
    suppress_full = wm(rates["H_0"] - rates["H_both_full"])
    induce_half = wm(rates["N_both_half"] - rates["N_0"])
    induce_full = wm(rates["N_both_full"] - rates["N_0"])
    bidir_half = (suppress_half + induce_half) / 2.0
    bidir_full = (suppress_full + induce_full) / 2.0
    gap0 = wm(rates["H_0"] - rates["N_0"])
    gapfull = wm(rates["H_both_full"] - rates["N_both_full"])
    closure = (gap0 - gapfull) / gap0 if gap0 > 0 else np.nan
    control = {}
    for family in ("random", "permutation"):
        h = wm(rates["H_0"] - rates[f"H_{family}_full"])
        n = wm(rates[f"N_{family}_full"] - rates["N_0"])
        control[family] = (h + n) / 2.0
    minus1 = (
        wm(rates["H_0"] - rates["H_minus1_full"])
        + wm(rates["N_minus1_full"] - rates["N_0"])
    ) / 2.0
    tbg = (
        wm(rates["H_0"] - rates["H_tbg_full"])
        + wm(rates["N_tbg_full"] - rates["N_0"])
    ) / 2.0
    # Cell-fixed-effect slope: for each cell regress oriented outcome
    # z=(r_N-r_H)/2 on relative dose x={0,.5,1}, then weight cell slopes.
    z = np.stack(
        [
            (rates["N_0"] - rates["H_0"]) / 2.0,
            (rates["N_both_half"] - rates["H_both_half"]) / 2.0,
            (rates["N_both_full"] - rates["H_both_full"]) / 2.0,
        ],
        axis=1,
    )
    x = np.asarray([0.0, 0.5, 1.0])
    centered = x - x.mean()
    cell_slope = (z @ centered) / float(np.square(centered).sum())
    slope = wm(cell_slope)
    return {
        "suppress_H_half": suppress_half,
        "suppress_H_full": suppress_full,
        "induce_N_half": induce_half,
        "induce_N_full": induce_full,
        "bidirectional_half": bidir_half,
        "bidirectional_full": bidir_full,
        "gap_0": gap0,
        "gap_full": gapfull,
        "closure": closure,
        "random_bidirectional_full": control["random"],
        "permutation_bidirectional_full": control["permutation"],
        "true_minus_random": bidir_full - control["random"],
        "true_minus_permutation": bidir_full - control["permutation"],
        "minus1_bidirectional_full": minus1,
        "tbg_bidirectional_full": tbg,
        "position_synergy": bidir_full - minus1 - tbg,
        "oriented_fixed_effect_slope": slope,
    }


def crossed_bootstrap(
    labels: dict[str, np.ndarray],
    prompt_index: np.ndarray,
    word_index: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    prompts, words = np.unique(prompt_index), np.unique(word_index)
    rates = {name: values.mean(axis=1) for name, values in labels.items()}
    names = list(causal_metrics(rates, np.ones(len(prompt_index))))
    output = {name: np.empty(n_bootstrap, dtype=float) for name in names}
    widths = {name: values.shape[1] for name, values in labels.items()}
    for bootstrap_index in range(n_bootstrap):
        prompt_counts = Counter(rng.choice(prompts, len(prompts), replace=True).tolist())
        word_counts = Counter(rng.choice(words, len(words), replace=True).tolist())
        weights = np.asarray(
            [prompt_counts[int(p)] * word_counts[int(w)] for p, w in zip(prompt_index, word_index)],
            dtype=float,
        )
        if weights.sum() == 0:
            raise AssertionError("empty crossed-bootstrap replicate")
        sampled = {
            name: rng.binomial(widths[name], rates[name]) / widths[name]
            for name in labels
        }
        metrics = causal_metrics(sampled, weights)
        for name, value in metrics.items():
            output[name][bootstrap_index] = value
    return output


def printable_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(character.isprintable() or character in "\n\t" for character in text) / len(text)


def repeated_fourgram_dominated(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 4:
        return False
    grams = [tuple(tokens[index : index + 4]) for index in range(len(tokens) - 3)]
    return max(Counter(grams).values()) / len(grams) > 0.5


def integrity_report(generation_meta: dict[str, dict]) -> tuple[dict, bool]:
    report = {}
    base_truncation = {}
    for arm in ("H", "N"):
        rows = generation_meta[f"{arm}_0"]["rows"].values()
        values = [bool(value) for row in rows for value in row["truncated"]]
        base_truncation[arm] = float(np.mean(values))
    all_pass = True
    for condition, meta in generation_meta.items():
        if not condition.startswith(("H_", "N_")):
            continue
        rows = list(meta["rows"].values())
        texts = [text for row in rows for text in row["llm_responses"]]
        truncation = [bool(value) for row in rows for value in row["truncated"]]
        printable = [printable_fraction(text) for text in texts]
        lengths = [len(text) for text in texts]
        dominated = [repeated_fourgram_dominated(text) for text in texts]
        arm = condition[0]
        checks = {
            "nonempty": all(bool(text.strip()) for text in texts),
            "truncation_within_margin": float(np.mean(truncation)) <= base_truncation[arm] + 0.05,
            "printable_median_ge_098": float(np.median(printable)) >= 0.98,
            "length_median_ge_20": float(np.median(lengths)) >= 20,
            "repetition_fraction_le_005": float(np.mean(dominated)) <= 0.05,
        }
        passed = all(checks.values())
        all_pass = all_pass and passed
        report[condition] = {
            "n_responses": len(texts),
            "truncation_rate": float(np.mean(truncation)),
            "median_printable_fraction": float(np.median(printable)),
            "median_char_length": float(np.median(lengths)),
            "repeated_fourgram_dominated_fraction": float(np.mean(dominated)),
            "checks": checks,
            "pass": passed,
        }
    return report, all_pass


def benign_report(labels: dict[str, np.ndarray], n_bootstrap: int, seed: int) -> dict:
    rates = {name: values.mean(axis=1) for name, values in labels.items()}
    point = {name: float(value.mean()) for name, value in rates.items()}
    rng = np.random.default_rng(seed)
    deltas = {name: np.empty(n_bootstrap) for name in labels if name != "B_0"}
    n = next(iter(labels.values())).shape[1]
    for index in range(n_bootstrap):
        rows = rng.integers(0, len(rates["B_0"]), len(rates["B_0"]))
        base = rng.binomial(n, rates["B_0"][rows]) / n
        for name in deltas:
            treated = rng.binomial(n, rates[name][rows]) / n
            deltas[name][index] = float((treated - base).mean())
    return {
        "condition_means": point,
        "delta_vs_B0": {
            name: {
                "estimate": point[name] - point["B_0"],
                "ci95": interval(samples),
            }
            for name, samples in deltas.items()
        },
    }


def summarize_metrics(point: dict, samples: dict[str, np.ndarray]) -> dict:
    return {
        name: {
            "estimate": float(value) if np.isfinite(value) else None,
            "ci95": interval(samples[name]),
        }
        for name, value in point.items()
    }


def main() -> None:
    args = parse_args()
    manifest_path, calibration_path = Path(args.manifest), Path(args.calibration)
    run_dir = Path(args.run_dir)
    manifest, calibration = read_json(manifest_path), read_json(calibration_path)
    manifest_record, calibration_record = file_record(manifest_path), file_record(calibration_path)
    if manifest.get("status") != "frozen" or calibration.get("status") != "pass":
        raise SystemExit("FATAL: x86 G0 manifest/calibration gate not passed")
    # Frozen-pipeline files must match the manifest; post-freeze changes to the
    # downstream judge/analyze scripts are permitted only under --allow-code-drift
    # and are recorded in the result. The frozen intervention pipeline
    # (common/build/calibrate/generate) is never allowed to drift.
    frozen_pipeline = {"x86_common.py", "x86_build_manifest.py", "x86_calibrate_kl.py", "x86_generate.py"}
    current_hashes = code_hashes()
    code_drift = {
        name: {"expected": expected, "current": current_hashes.get(name)}
        for name, expected in manifest["code_sha256"].items()
        if current_hashes.get(name) != expected
    }
    for name in code_drift:
        if name in frozen_pipeline or not args.allow_code_drift:
            raise SystemExit(f"FATAL: x86 code changed after manifest freeze: {name}")

    target_ids = [int(row["target_id"]) for row in manifest["target_cells"]]
    prompt_index = np.asarray([row["prompt_index"] for row in manifest["target_cells"]], dtype=int)
    word_index = np.asarray([row["word_index"] for row in manifest["target_cells"]], dtype=int)
    target_conditions = [row["condition"] for row in manifest["target_conditions"]]
    # comply2 is a secondary/diagnostic outcome. It is loaded only if every target
    # condition has complete, valid comply2 judgments; otherwise the compliance
    # section is reported as pending and the primary refusal2 gates still run.
    # (Recorded deviation, 2026-08-24: comply2 judging was blocked by OpenRouter
    # credit exhaustion; the refusal2 primary endpoint is unaffected.)
    def comply2_ready(condition: str) -> bool:
        gpath, jpath = condition_paths(run_dir, condition, "comply2")
        if not gpath.exists() or not jpath.exists():
            return False
        try:
            load_condition(
                run_dir, condition, "comply2", target_ids,
                manifest_record["sha256"], calibration_record["sha256"],
            )
        except SystemExit:
            return False
        return True

    comply2_available = all(comply2_ready(condition) for condition in target_conditions)

    target_refusal, target_comply = {}, {}
    target_meta = {}
    inputs = {"manifest": file_record(manifest_path), "calibration": file_record(calibration_path)}
    for condition in target_conditions:
        target_refusal[condition], meta = load_condition(
            run_dir,
            condition,
            "refusal2",
            target_ids,
            manifest_record["sha256"],
            calibration_record["sha256"],
        )
        target_meta[condition] = meta
        inputs[f"generation_{condition}"] = meta["generation"]
        inputs[f"refusal2_{condition}"] = meta["judgment"]
        if comply2_available:
            target_comply[condition], comply_meta = load_condition(
                run_dir,
                condition,
                "comply2",
                target_ids,
                manifest_record["sha256"],
                calibration_record["sha256"],
            )
            inputs[f"comply2_{condition}"] = comply_meta["judgment"]

    benign_ids = [int(row["full_row"]) for row in manifest["benign_evaluation"]]
    benign_labels = {}
    for condition in [row["condition"] for row in manifest["benign_conditions"]]:
        benign_labels[condition], meta = load_condition(
            run_dir,
            condition,
            "refusal2",
            benign_ids,
            manifest_record["sha256"],
            calibration_record["sha256"],
        )
        inputs[f"generation_{condition}"] = meta["generation"]
        inputs[f"refusal2_{condition}"] = meta["judgment"]

    refusal_rates = {name: values.mean(axis=1) for name, values in target_refusal.items()}
    point = causal_metrics(refusal_rates, np.ones(len(target_ids)))
    boot = crossed_bootstrap(
        target_refusal, prompt_index, word_index, args.bootstrap, args.seed
    )
    primary = summarize_metrics(point, boot)
    endpoint_names = ("suppress_H_full", "induce_N_full")
    raw_p = [one_sided_p(boot[name]) for name in endpoint_names]
    adjusted_p = holm(raw_p)
    for name, p_value, adj in zip(endpoint_names, raw_p, adjusted_p):
        primary[name]["one_sided_p"] = p_value
        primary[name]["holm_adjusted_p"] = adj
    specificity_names = ("true_minus_random", "true_minus_permutation")
    spec_p = [one_sided_p(boot[name]) for name in specificity_names]
    spec_adj = holm(spec_p)
    for name, p_value, adj in zip(specificity_names, spec_p, spec_adj):
        primary[name]["one_sided_p"] = p_value
        primary[name]["holm_adjusted_p"] = adj

    integrity, g3 = integrity_report(target_meta)
    g0 = bool(
        manifest["audits"].get("model", {}).get("dose_zero_equivalence", {}).get("all_pass")
        and calibration["true_dose_gate_pass"]
        and calibration["null_calibration_gate_pass"]
    )
    g1 = bool(
        g0
        and point["suppress_H_full"] >= 0.03
        and primary["suppress_H_full"]["ci95"][0] > 0
        and point["induce_N_full"] >= 0.03
        and primary["induce_N_full"]["ci95"][0] > 0
        and all(primary[name]["holm_adjusted_p"] < 0.05 for name in endpoint_names)
        and point["bidirectional_half"] > 0
        and point["bidirectional_full"] > point["bidirectional_half"]
        and primary["oriented_fixed_effect_slope"]["ci95"][0] > 0
    )
    g2 = bool(
        g1
        and all(primary[name]["ci95"][0] > 0 for name in specificity_names)
        and all(primary[name]["holm_adjusted_p"] < 0.05 for name in specificity_names)
    )

    if comply2_available:
        comply_rates = {name: values.mean(axis=1) for name, values in target_comply.items()}
        comply_point = causal_metrics(comply_rates, np.ones(len(target_ids)))
        comply_boot = crossed_bootstrap(
            target_comply, prompt_index, word_index, args.bootstrap, args.seed + 1
        )
        compliance = summarize_metrics(comply_point, comply_boot)
        compliance["interpretation"] = {
            "H_subtraction_compliance_increase": {
                "estimate": float(-comply_point["suppress_H_full"]),
                "ci95": interval(-comply_boot["suppress_H_full"]),
            },
            "N_addition_compliance_decrease": {
                "estimate": float(-comply_point["induce_N_full"]),
                "ci95": interval(-comply_boot["induce_N_full"]),
            },
        }
    else:
        compliance = {
            "status": "pending",
            "reason": "comply2 judgments incomplete (OpenRouter credit exhaustion 2026-08-24); "
            "secondary substantive-compliance analysis not computed. refusal2 primary is complete.",
        }

    benign = benign_report(benign_labels, args.bootstrap, args.seed + 2)
    status = "complete"
    if g0 and g1 and g2 and g3:
        decision = "causally_usable_and_specific"
    elif g0 and g1 and g3:
        decision = "causally_usable_specificity_failed"
    elif g0 and g3 and point["suppress_H_full"] > 0 and point["induce_N_full"] <= 0:
        decision = "mediation_only"
    elif g0 and g3 and point["induce_N_full"] > 0 and point["suppress_H_full"] <= 0:
        decision = "sufficiency_only"
    else:
        decision = "causal_gate_failed"
    output = {
        "experiment": "x86",
        "status": "complete" if comply2_available else "primary_complete_comply2_pending",
        "comply2_available": comply2_available,
        "code_drift": code_drift,
        "decision": decision,
        "scope": "Qwen3.5-4B x85 fold-specific state-30 TBG-1/TBG activation intervention",
        "config": {
            "n_target_cells": len(target_ids),
            "n_benign_prompts": len(benign_ids),
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "a_star": calibration["admitted_a_star"],
        },
        "inputs": inputs,
        "gates": {"G0_integrity_calibration": g0, "G1_bidirectional": g1, "G2_specificity": g2, "G3_output_integrity": g3},
        "refusal": primary,
        "output_integrity": integrity,
        "compliance_secondary": compliance,
        "benign_over_refusal": benign,
        "wording_rule": (
            "Only G0+G1+G3 licenses causally usable; G2 additionally licenses specificity "
            "against the two tested KL-matched controls. Compliance remains a separate construct."
        ),
    }
    atomic_json(Path(args.out), output)
    print(f"x86 analysis: decision={decision}, gates={output['gates']} -> {args.out}")


if __name__ == "__main__":
    main()
