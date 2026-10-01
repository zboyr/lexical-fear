"""x86 Stage 1: benign-KL dose admission and matched-null calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from x86_common import (
    BOUNDARY_POSITIONS,
    DOSE_GRID,
    KL_CAP,
    KL_SLACK,
    MODEL_ID,
    RUN,
    BoundaryHook,
    atomic_json,
    file_record,
    format_user_chat,
    install_hook,
    load_causal_lm,
    load_direction_vectors,
    read_json,
)

ABS_MATCH_TOL = 5e-5
REL_MATCH_TOL = 0.05
NULL_MAX_MULTIPLIER = 8.0
MAX_BISECTIONS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(RUN / "x86_manifest.json"))
    parser.add_argument("--out", default=str(RUN / "x86_kl_calibration.json"))
    parser.add_argument("--limit", type=int, default=0, help="smoke only; cannot produce a passing artifact")
    return parser.parse_args()


def monotone(records: list[dict], slack: float = KL_SLACK) -> bool:
    ordered = sorted(records, key=lambda row: float(row["magnitude"]))
    values = [float(row["mean_kl"]) for row in ordered]
    return all(right + slack >= left for left, right in zip(values, values[1:]))


class KLRunner:
    def __init__(self, manifest: dict, rows: list[dict], vectors: dict[str, np.ndarray]):
        self.manifest = manifest
        self.rows = rows
        self.vectors = vectors
        model_source = manifest["model_snapshot"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = load_causal_lm(model_source)
        self.model.eval()
        self.hook = BoundaryHook()
        self.handle = install_hook(self.model, self.hook)
        self.examples = []
        for row in rows:
            chat = format_user_chat(self.tokenizer, row["prompt"])
            tokens = self.tokenizer(
                chat, return_tensors="pt", add_special_tokens=False
            ).to(self.model.device)
            length = int(tokens["input_ids"].shape[1])
            self.hook.enabled = True
            self.hook.configure(0.0, length - 2, length - 1, None, None)
            with torch.inference_mode():
                logits = self.model(**tokens, return_dict=True, use_cache=False).logits[0, -1].float()
            self.examples.append(
                {
                    "fold": int(row["fold"]),
                    "tokens": tokens,
                    "base_logp": F.log_softmax(logits, dim=-1).cpu(),
                    "length": length,
                }
            )

    def close(self) -> None:
        self.handle.remove()
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def measure(
        self,
        family: str,
        signed_dose: float,
        positions: tuple[str, ...] = BOUNDARY_POSITIONS,
    ) -> dict:
        values = []
        by_fold = {fold: [] for fold in range(4)}
        for example in self.examples:
            fold, length = example["fold"], example["length"]
            minus1 = (
                torch.from_numpy(self.vectors[f"{family}_f{fold}_tbg_minus1"])
                if "tbg_minus1" in positions
                else None
            )
            tbg = (
                torch.from_numpy(self.vectors[f"{family}_f{fold}_tbg"])
                if "tbg" in positions
                else None
            )
            self.hook.enabled = True
            self.hook.reset_counters()
            self.hook.configure(signed_dose, length - 2, length - 1, minus1, tbg)
            with torch.inference_mode():
                logits = self.model(
                    **example["tokens"], return_dict=True, use_cache=False
                ).logits[0, -1].float()
            expected_minus1 = int(signed_dose != 0 and minus1 is not None)
            expected_tbg = int(signed_dose != 0 and tbg is not None)
            if (
                self.hook.prefill_calls != 1
                or self.hook.decode_calls != 0
                or self.hook.writes_minus1 != expected_minus1
                or self.hook.writes_tbg != expected_tbg
            ):
                raise SystemExit("FATAL: KL hook call/write audit failed")
            edited = F.log_softmax(logits, dim=-1).cpu()
            base = example["base_logp"]
            kl = float((base.exp() * (base - edited)).sum())
            if not np.isfinite(kl):
                raise SystemExit("FATAL: non-finite benign KL")
            values.append(kl)
            by_fold[fold].append(kl)
        return {
            "family": family,
            "signed_dose": float(signed_dose),
            "magnitude": abs(float(signed_dose)),
            "positions": list(positions),
            "n_prompts": len(values),
            "mean_kl": float(np.mean(values)),
            "max_kl": float(np.max(values)),
            "fold_mean_kl": {
                str(fold): float(np.mean(fold_values)) for fold, fold_values in by_fold.items()
            },
        }


def admit_true_dose(records: dict[str, list[dict]]) -> float | None:
    admitted = []
    for magnitude in DOSE_GRID:
        eligible = True
        for sign_name in ("minus", "plus"):
            prefix = [
                row for row in records[sign_name] if float(row["magnitude"]) <= magnitude
            ]
            current = next(row for row in prefix if float(row["magnitude"]) == magnitude)
            if (
                current["mean_kl"] > KL_CAP
                or max(current["fold_mean_kl"].values()) > KL_CAP
                or not monotone(prefix)
            ):
                eligible = False
        if eligible:
            admitted.append(float(magnitude))
    return max(admitted) if admitted else None


def calibrate_null(
    runner: KLRunner,
    family: str,
    sign: float,
    target_kl: float,
) -> dict:
    tolerance = max(ABS_MATCH_TOL, REL_MATCH_TOL * target_kl)
    zero = runner.measure(family, 0.0)
    high = runner.measure(family, sign * NULL_MAX_MULTIPLIER)
    evaluations = [zero, high]
    if high["mean_kl"] + tolerance < target_kl:
        return {
            "status": "unbracketed",
            "target_kl": target_kl,
            "tolerance": tolerance,
            "evaluations": evaluations,
        }
    low_value, high_value = 0.0, NULL_MAX_MULTIPLIER
    selected = None
    for _ in range(MAX_BISECTIONS):
        midpoint = (low_value + high_value) / 2.0
        result = runner.measure(family, sign * midpoint)
        evaluations.append(result)
        if abs(result["mean_kl"] - target_kl) <= tolerance:
            selected = result
            break
        if result["mean_kl"] < target_kl:
            low_value = midpoint
        else:
            high_value = midpoint
    if selected is None:
        selected = min(evaluations, key=lambda row: abs(row["mean_kl"] - target_kl))
    nonzero = [row for row in evaluations if row["magnitude"] > 0]
    if not monotone(nonzero):
        status = "nonmonotone"
    elif abs(selected["mean_kl"] - target_kl) > tolerance:
        status = "tolerance_miss"
    else:
        status = "pass"
    return {
        "status": status,
        "target_kl": target_kl,
        "tolerance": tolerance,
        "matched_signed_dose": selected["signed_dose"],
        "matched_mean_kl": selected["mean_kl"],
        "evaluations": sorted(evaluations, key=lambda row: row["signed_dose"]),
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    if manifest.get("experiment") != "x86" or manifest.get("status") != "frozen":
        raise SystemExit("FATAL: x86 manifest lacks completed model audits")
    if manifest.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: x86 model mismatch")
    rows = list(manifest["benign_calibration"])
    if args.limit:
        rows = rows[: args.limit]
    vectors = load_direction_vectors(manifest)
    runner = KLRunner(manifest, rows, vectors)
    try:
        true = {"minus": [], "plus": []}
        for magnitude in DOSE_GRID:
            true["minus"].append(runner.measure("true", -float(magnitude)))
            true["plus"].append(runner.measure("true", +float(magnitude)))
        admitted = admit_true_dose(true) if not args.limit else None
        nulls = {}
        single_position = {}
        if admitted is not None:
            target_minus = next(
                row["mean_kl"] for row in true["minus"] if row["magnitude"] == admitted
            )
            target_plus = next(
                row["mean_kl"] for row in true["plus"] if row["magnitude"] == admitted
            )
            for family in ("random", "permutation"):
                nulls[family] = {
                    "minus": calibrate_null(runner, family, -1.0, target_minus),
                    "plus": calibrate_null(runner, family, +1.0, target_plus),
                }
            for position in BOUNDARY_POSITIONS:
                single_position[position] = {
                    "minus": runner.measure("true", -admitted, (position,)),
                    "plus": runner.measure("true", +admitted, (position,)),
                }
    finally:
        runner.close()

    controls_pass = bool(
        admitted is not None
        and all(
            nulls[family][sign]["status"] == "pass"
            for family in ("random", "permutation")
            for sign in ("minus", "plus")
        )
    )
    status = "pass" if admitted is not None and controls_pass and not args.limit else "failed"
    output = {
        "experiment": "x86",
        "stage": 1,
        "status": status,
        "model_id": MODEL_ID,
        "manifest": file_record(manifest_path),
        "directions": manifest["directions_npz"],
        "n_benign_calibration": len(rows),
        "smoke_limit": args.limit,
        "config": {
            "dose_grid": list(DOSE_GRID),
            "kl_cap": KL_CAP,
            "kl_slack": KL_SLACK,
            "null_max_multiplier": NULL_MAX_MULTIPLIER,
            "max_bisections": MAX_BISECTIONS,
            "abs_match_tolerance": ABS_MATCH_TOL,
            "relative_match_tolerance": REL_MATCH_TOL,
        },
        "true_both": true,
        "admitted_a_star": admitted,
        "single_position_at_a_star": single_position,
        "null_calibration": nulls,
        "true_dose_gate_pass": admitted is not None,
        "null_calibration_gate_pass": controls_pass,
    }
    atomic_json(Path(args.out), output)
    print(
        f"x86 KL: status={status}, a*={admitted}, controls={controls_pass} -> {args.out}"
    )
    if status != "pass" and not args.limit:
        raise SystemExit("x86 KL calibration gate failed")


if __name__ == "__main__":
    main()
