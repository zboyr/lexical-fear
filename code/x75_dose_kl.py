"""Calibrate x75 signed doses to immutable x71 benign-KL anchors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

from edit_common import atomic_json_dump, file_record, load_causal_lm, load_config  # noqa: E402
from edit_rank1 import install_rank_one_edit  # noqa: E402
from x71_train_edit import select_last_positions, tokenize_prompts  # noqa: E402
from x75_weighting import ARMS  # noqa: E402


TRUE_REFERENCE_DOSES = (-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0)
CONTROL_REFERENCE_DOSES = (-3.0, -1.0, 1.0, 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--kind", required=True, choices=("true", "control"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--prompt-splits", required=True)
    parser.add_argument("--reference-kl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-doses", nargs="*", type=float)
    parser.add_argument("--absolute-tolerance", type=float, default=5e-5)
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--max-iterations", type=int, default=14)
    parser.add_argument("--max-abs-dose", type=float, default=16.0)
    parser.add_argument("--bracket-growth", type=float, default=2.0)
    parser.add_argument("--monotonic-slack", type=float, default=1e-6)
    return parser.parse_args()


def target_tolerance(target: float, absolute: float, relative: float) -> float:
    if target <= 0 or absolute <= 0 or relative < 0:
        raise ValueError("KL target/tolerances must be positive")
    return max(float(absolute), float(relative) * float(target))


def reference_anchors(obj: dict, requested: tuple[float, ...]) -> list[dict]:
    available = {float(row["dose"]): float(row["kl_full_mean"]) for row in obj["measurements"]}
    anchors = []
    for dose in requested:
        if dose == 0 or dose not in available:
            raise ValueError(f"reference KL artifact lacks requested nonzero dose {dose}")
        anchors.append({"reference_dose": float(dose), "target_kl": available[dose]})
    return anchors


def calibrate_target(
    measure: Callable[[float], float],
    *,
    reference_dose: float,
    target_kl: float,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_iterations: int,
    max_abs_dose: float,
    bracket_growth: float,
    monotonic_slack: float,
) -> dict:
    """Bracket and bisect one signed target using validation-only KL calls."""
    if reference_dose == 0:
        raise ValueError("reference dose must be nonzero")
    sign = -1.0 if reference_dose < 0 else 1.0
    tolerance = target_tolerance(target_kl, absolute_tolerance, relative_tolerance)
    low_abs, low_kl = 0.0, 0.0
    high_abs = min(abs(reference_dose), max_abs_dose)
    high_kl = float(measure(sign * high_abs))
    evaluations = [(low_abs, low_kl), (high_abs, high_kl)]

    while high_kl < target_kl - tolerance and high_abs < max_abs_dose:
        next_abs = min(max_abs_dose, high_abs * bracket_growth)
        if next_abs <= high_abs:
            break
        next_kl = float(measure(sign * next_abs))
        evaluations.append((next_abs, next_kl))
        if next_kl + monotonic_slack < high_kl:
            return {
                "status": "non_monotone_bracket",
                "reference_dose": reference_dose,
                "target_kl": target_kl,
                "tolerance": tolerance,
                "evaluations": evaluations,
            }
        low_abs, low_kl = high_abs, high_kl
        high_abs, high_kl = next_abs, next_kl

    if high_kl < target_kl - tolerance:
        return {
            "status": "unbracketed",
            "reference_dose": reference_dose,
            "target_kl": target_kl,
            "tolerance": tolerance,
            "evaluations": evaluations,
        }

    best_abs, best_kl = min(evaluations, key=lambda row: abs(row[1] - target_kl))
    for _ in range(max_iterations):
        if abs(best_kl - target_kl) <= tolerance:
            break
        middle_abs = 0.5 * (low_abs + high_abs)
        middle_kl = float(measure(sign * middle_abs))
        evaluations.append((middle_abs, middle_kl))
        if middle_kl + monotonic_slack < low_kl or middle_kl > high_kl + monotonic_slack:
            return {
                "status": "non_monotone_bisection",
                "reference_dose": reference_dose,
                "target_kl": target_kl,
                "tolerance": tolerance,
                "evaluations": evaluations,
            }
        if abs(middle_kl - target_kl) < abs(best_kl - target_kl):
            best_abs, best_kl = middle_abs, middle_kl
        if middle_kl < target_kl:
            low_abs, low_kl = middle_abs, middle_kl
        else:
            high_abs, high_kl = middle_abs, middle_kl

    passed = abs(best_kl - target_kl) <= tolerance
    return {
        "status": "matched" if passed else "tolerance_failed",
        "reference_dose": float(reference_dose),
        "target_kl": float(target_kl),
        "tolerance": float(tolerance),
        "actual_dose": float(sign * best_abs),
        "achieved_kl": float(best_kl),
        "absolute_error": float(abs(best_kl - target_kl)),
        "within_tolerance": bool(passed),
        "evaluations": evaluations,
    }


class KLEvaluator:
    def __init__(self, args: argparse.Namespace, config: dict, adapter: dict):
        self.args = args
        self.config = config
        self.device = torch.device(args.device)
        prompt_splits = json.loads(Path(args.prompt_splits).read_text())
        self.rows = prompt_splits["benign_validation"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = load_causal_lm(config["model_id"])
        self.model.config.use_cache = False
        self.model.eval()
        self.wrapper = install_rank_one_edit(
            self.model,
            block_index=int(adapter["block_index"]),
            q_unit=adapter["q_unit"],
            scale=0.0,
        )
        self.wrapper.right.data.copy_(adapter["right"].float().to(self.wrapper.right.device))
        self.wrapper.right.requires_grad_(False)
        self.last_n = int(config["edit"]["retain_last_n"])
        self.date_string = config["probe_chat_date"]
        self.cache: dict[float, float] = {0.0: 0.0}

    def measure(self, dose: float) -> float:
        key = round(float(dose), 12)
        if key in self.cache:
            return self.cache[key]
        means = []
        batch_size = int(self.args.batch_size)
        with torch.inference_mode():
            for start in range(0, len(self.rows), batch_size):
                prompts = [
                    str(row["prompt"]) for row in self.rows[start : start + batch_size]
                ]
                tokens = tokenize_prompts(
                    self.tokenizer, prompts, self.date_string, self.device
                )
                self.wrapper.scale = 0.0
                base = self.model(**tokens, return_dict=True, use_cache=False).logits
                base = select_last_positions(
                    base, tokens["attention_mask"], self.last_n
                ).float()
                base_logp = F.log_softmax(base, dim=-1)
                self.wrapper.scale = float(dose)
                edited = self.model(**tokens, return_dict=True, use_cache=False).logits
                edited = select_last_positions(
                    edited, tokens["attention_mask"], self.last_n
                ).float()
                edited_logp = F.log_softmax(edited, dim=-1)
                kl = float((base_logp.exp() * (base_logp - edited_logp)).sum(-1).mean())
                if not math.isfinite(kl):
                    raise RuntimeError(f"non-finite KL at dose {dose}, batch {start}")
                means.append(kl)
        self.wrapper.scale = 0.0
        result = float(sum(means) / len(means))
        self.cache[key] = result
        print(json.dumps({"dose": dose, "kl": result}), flush=True)
        return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    if adapter.get("metadata", {}).get("probe_kind") != args.kind:
        raise ValueError("adapter probe kind disagrees with --kind")
    probe = torch.load(adapter["probe_path"], map_location="cpu", weights_only=False)
    if probe.get("arm") != args.arm or probe.get("kind") != args.kind:
        raise ValueError("adapter probe artifact disagrees with requested arm/kind")

    reference = json.loads(Path(args.reference_kl).read_text())
    defaults = TRUE_REFERENCE_DOSES if args.kind == "true" else CONTROL_REFERENCE_DOSES
    requested = tuple(args.reference_doses) if args.reference_doses else defaults
    anchors = reference_anchors(reference, requested)
    budget = float(config["edit"]["kl_budget"])
    if any(anchor["target_kl"] > budget for anchor in anchors):
        raise ValueError("a requested x71 KL anchor exceeds the inherited budget")

    evaluator = KLEvaluator(args, config, adapter)
    calibrations = []
    for anchor in anchors:
        calibrations.append(
            calibrate_target(
                evaluator.measure,
                reference_dose=anchor["reference_dose"],
                target_kl=anchor["target_kl"],
                absolute_tolerance=args.absolute_tolerance,
                relative_tolerance=args.relative_tolerance,
                max_iterations=args.max_iterations,
                max_abs_dose=args.max_abs_dose,
                bracket_growth=args.bracket_growth,
                monotonic_slack=args.monotonic_slack,
            )
        )

    right_norm = float(adapter["right"].float().norm())
    for row in calibrations:
        if "actual_dose" in row:
            row["right_norm"] = right_norm
            row["rank_one_delta_frobenius_norm"] = abs(row["actual_dose"]) * right_norm
    passed = all(row.get("status") == "matched" for row in calibrations)
    result = {
        "schema_version": 1,
        "experiment": "x75",
        "arm": args.arm,
        "kind": args.kind,
        "status": "complete" if passed else "calibration_failed",
        "kl_definition": "x71 full-512 benign validation, last prompt positions, KL(base||edited)",
        "kl_budget": budget,
        "batch_size": int(args.batch_size),
        "n_benign_validation_rows": len(evaluator.rows),
        "calibration_settings": {
            "absolute_tolerance": args.absolute_tolerance,
            "relative_tolerance": args.relative_tolerance,
            "max_iterations": args.max_iterations,
            "max_abs_dose": args.max_abs_dose,
            "bracket_growth": args.bracket_growth,
            "monotonic_slack": args.monotonic_slack,
        },
        "calibrations": calibrations,
        "all_measurements": [
            {"dose": dose, "kl_full_mean": kl}
            for dose, kl in sorted(evaluator.cache.items())
        ],
        "adapter": file_record(args.adapter),
        "probe": file_record(adapter["probe_path"]),
        "reference_kl": file_record(args.reference_kl),
        "prompt_splits": file_record(args.prompt_splits),
    }
    atomic_json_dump(result, args.out)
    print(json.dumps({"status": result["status"], "calibrations": calibrations}, indent=2))
    if not passed:
        raise SystemExit("one or more KL anchors failed calibration")


if __name__ == "__main__":
    main()
