"""x86 Stage 0: rebuild fold directions, target graph, benign splits, nulls, and hook audits.

Reads only the frozen x85 cohort and the 24 x85 feature shards. It never opens
any x85 generation, judgment, per-cell analysis, or level-aggregation artifact.
The x85 all-data directions npz is used only as a lineage cosine check.
"""

from __future__ import annotations

import argparse
import glob
import platform
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from x86_common import (
    BENIGN_CALIB_SEED,
    BENIGN_EVAL_SEED,
    BOUNDARY_POSITIONS,
    DOSE_ZERO_AUDIT_SEED,
    HOOK_BLOCK,
    INTERVENTION_STATE,
    MODEL_ID,
    PERMUTATION_NULL_SEED,
    RESULTS,
    RUN,
    TARGET_GRAPH_SEED,
    BoundaryHook,
    atomic_json,
    benign_conditions,
    code_hashes,
    cohort_path,
    file_record,
    fold_direction,
    format_user_chat,
    install_hook,
    load_causal_lm,
    orthogonal_random_null,
    permutation_sign_null,
    read_json,
    sha256_file,
    sha256_text,
    target_conditions,
    target_graph,
)

DATA = RUN.parents[1]
BENIGN_POOL = DATA / "pools" / "cp_15k_benign.json"
BENIGN_ROW_OFFSET = 15000
N_BENIGN_PER_SPLIT = 256
N_BENIGN_PER_FOLD = 64
N_AUDIT_PROMPTS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x85-cohort", default=str(DATA / "runs" / "x85" / "x85_cohort.json"))
    parser.add_argument("--features-glob", default=str(DATA / "runs" / "x85" / "features" / "x85_hidden_shard_*.pt"))
    parser.add_argument("--x85-directions", default=str(RESULTS / "x85_crossed_keyword_directions.npz"))
    parser.add_argument("--out", default=str(cohort_path()))
    parser.add_argument("--directions-out", default=str(RUN / "x86_directions.npz"))
    parser.add_argument("--skip-model-audit", action="store_true", help="smoke only; cannot freeze")
    return parser.parse_args()


def load_boundary_states(feature_paths: list[Path], cohort: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Return cell_id -> [position, hidden] fp32 at state 30, plus a shard-audit record."""
    position_index = None
    state_index = INTERVENTION_STATE
    vectors: dict[str, np.ndarray] = {}
    shard_records = []
    seen_cell_ids: set[str] = set()
    cells_by_index = {int(cell["cell_index"]): cell for cell in cohort["cells"]}
    n_states = hidden_dim = None
    for path in feature_paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("experiment") != "x86" and artifact.get("experiment") != "x85":
            raise SystemExit(f"FATAL: {path} is not an x85 feature shard")
        if artifact.get("cohort_sha256") != cohort["cohort_sha256"]:
            raise SystemExit(f"FATAL: {path} cohort hash mismatch")
        names = list(artifact["position_names"])
        if tuple(names[2:4]) != BOUNDARY_POSITIONS:
            raise SystemExit(f"FATAL: {path} boundary position order unexpected: {names}")
        if position_index is None:
            position_index = [names.index(name) for name in BOUNDARY_POSITIONS]
            n_states = int(artifact["num_states"])
            hidden_dim = int(artifact["hidden_dim"])
        if int(artifact["num_states"]) != n_states or int(artifact["hidden_dim"]) != hidden_dim:
            raise SystemExit(f"FATAL: {path} state/hidden dimensions changed")
        states = artifact["states"]  # [cell, position, state, hidden] fp16
        for row, cell_record in enumerate(artifact["cells"]):
            cell_id = cell_record["cell_id"]
            reference = cells_by_index[int(cell_record["cell_index"])]
            if (
                cell_id != reference["cell_id"]
                or cell_record["prompt_sha256"] != reference["prompt_sha256"]
                or cell_record["position_indices"] != reference["position_indices"]
                or int(cell_record["n_input_tokens"]) != int(reference["n_input_tokens"])
            ):
                raise SystemExit(f"FATAL: {path} cell audit mismatch for {cell_id}")
            if cell_id in seen_cell_ids:
                raise SystemExit(f"FATAL: duplicate cell {cell_id} across shards")
            seen_cell_ids.add(cell_id)
            slab = states[row, position_index, state_index, :].to(torch.float32).numpy()
            vectors[cell_id] = slab  # [2, hidden]
        shard_records.append(file_record(path))
    if len(seen_cell_ids) != len(cohort["cells"]):
        raise SystemExit(
            f"FATAL: shard coverage {len(seen_cell_ids)} != {len(cohort['cells'])} cohort cells"
        )
    audit = {
        "n_shards": len(feature_paths),
        "n_cells": len(seen_cell_ids),
        "num_states": n_states,
        "hidden_dim": hidden_dim,
        "state_index": state_index,
        "boundary_position_indices": position_index,
        "shards": shard_records,
    }
    return vectors, audit


def build_delta_grid(cohort: dict, vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    prompts = sorted(cohort["prompts"], key=lambda row: int(row["prompt_index"]))
    words = sorted(cohort["words"], key=lambda row: int(row["word_index"]))
    n_prompts, n_words = len(prompts), len(words)
    hidden_dim = next(iter(vectors.values())).shape[1]
    deltas = {
        position: np.zeros((n_prompts, n_words, hidden_dim), dtype=np.float64)
        for position in BOUNDARY_POSITIONS
    }
    for prompt in prompts:
        prompt_index, prompt_id = int(prompt["prompt_index"]), int(prompt["id"])
        for word in words:
            word_index = int(word["word_index"])
            harm = vectors[f"H:p{prompt_id}:w{word_index:02d}"]
            neutral = vectors[f"N:p{prompt_id}:w{word_index:02d}"]
            for offset, position in enumerate(BOUNDARY_POSITIONS):
                deltas[position][prompt_index, word_index] = harm[offset] - neutral[offset]
    prompt_folds = np.asarray([int(row["prompt_fold"]) for row in prompts], dtype=int)
    word_folds = np.asarray([int(row["word_fold"]) for row in words], dtype=int)
    return deltas, prompt_folds, word_folds, prompts, words


def build_directions(
    deltas: dict[str, np.ndarray],
    prompt_folds: np.ndarray,
    word_folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(PERMUTATION_NULL_SEED)
    prompt_signs = rng.choice([-1.0, 1.0], size=len(prompt_folds))
    word_signs = rng.choice([-1.0, 1.0], size=len(word_folds))
    vectors: dict[str, np.ndarray] = {}
    audit: dict = {"fold_norms": {}, "cross_fold_cosine": {}, "permutation_cosine": {}}
    for position in BOUNDARY_POSITIONS:
        grid = deltas[position]
        fold_true = {}
        for fold in range(4):
            true = fold_direction(grid, prompt_folds, word_folds, fold).astype(np.float32)
            norm = float(np.linalg.norm(true))
            if not np.isfinite(norm) or norm == 0.0:
                raise SystemExit(f"FATAL: degenerate true direction fold {fold} {position}")
            fold_true[fold] = true
            vectors[f"true_f{fold}_{position}"] = true
            vectors[f"random_f{fold}_{position}"] = orthogonal_random_null(true, fold, position).astype(np.float32)
            permutation = permutation_sign_null(
                grid, prompt_folds, word_folds, fold, prompt_signs, word_signs
            ).astype(np.float32)
            vectors[f"permutation_f{fold}_{position}"] = permutation
            audit["fold_norms"][f"f{fold}_{position}"] = norm
            cos_perm = float(
                permutation @ true / (np.linalg.norm(permutation) * norm)
            )
            audit["permutation_cosine"][f"f{fold}_{position}"] = cos_perm
        cosine_matrix = np.zeros((4, 4))
        for a in range(4):
            for b in range(4):
                cosine_matrix[a, b] = float(
                    fold_true[a] @ fold_true[b]
                    / (np.linalg.norm(fold_true[a]) * np.linalg.norm(fold_true[b]))
                )
        audit["cross_fold_cosine"][position] = cosine_matrix.tolist()
    audit["permutation_sign_seed"] = PERMUTATION_NULL_SEED
    return vectors, audit


def lineage_cosine(x86_vectors: dict[str, np.ndarray], x85_directions_path: Path) -> dict:
    if not x85_directions_path.exists():
        return {"status": "x85_directions_absent"}
    with np.load(x85_directions_path) as data:
        available = set(data.files)
        report = {}
        for position in BOUNDARY_POSITIONS:
            key = f"global_unit_{position}_s{INTERVENTION_STATE}"
            if key not in available:
                report[position] = None
                continue
            reference = data[key].astype(np.float64)
            reference = reference / np.linalg.norm(reference)
            cosines = []
            for fold in range(4):
                vector = x86_vectors[f"true_f{fold}_{position}"].astype(np.float64)
                cosines.append(float(vector @ reference / np.linalg.norm(vector)))
            report[position] = cosines
    return {"status": "checked", "true_fold_cosine_to_x85_global_unit": report}


def build_target_cells(cohort: dict) -> list[dict]:
    cells_by_key = {}
    for cell in cohort["cells"]:
        if cell["condition"] in ("H", "N"):
            cells_by_key[(cell["condition"], int(cell["prompt_index"]), int(cell["word_index"]))] = cell
    fold_manifest = {"prompts": cohort["prompts"], "words": cohort["words"]}
    edges = target_graph(fold_manifest, TARGET_GRAPH_SEED)
    target_cells = []
    for target_id, edge in enumerate(edges):
        prompt_index, word_index = int(edge["prompt_index"]), int(edge["word_index"])

        def info(condition: str) -> dict:
            cell = cells_by_key[(condition, prompt_index, word_index)]
            return {
                "prompt": cell["prompt"],
                "prompt_id": int(cell["prompt_id"]),
                "prompt_sha256": cell["prompt_sha256"],
                "rendered_chat_sha256": cell["rendered_chat_sha256"],
                "tbg_minus1_idx": int(cell["position_indices"]["tbg_minus1"]),
                "tbg_idx": int(cell["position_indices"]["tbg"]),
                "category": cell["category"],
                "harm_word": cell["harm_word"],
                "neutral_word": cell["neutral_word"],
            }

        target_cells.append(
            {
                "target_id": target_id,
                "prompt_index": prompt_index,
                "word_index": word_index,
                "fold": int(edge["fold"]),
                "stratum": edge["stratum"],
                "H": info("H"),
                "N": info("N"),
            }
        )
    return target_cells


def build_benign_split(indices_path: Path, seed: int, pool: list[dict]) -> list[dict]:
    indices = [int(value) for value in read_json(indices_path)]
    benign_rows = sorted(index for index in indices if index >= BENIGN_ROW_OFFSET)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(benign_rows), size=N_BENIGN_PER_SPLIT, replace=False)
    selected = sorted(benign_rows[int(index)] for index in chosen)
    rows = []
    for order, full_row in enumerate(selected):
        pool_index = full_row - BENIGN_ROW_OFFSET
        rows.append(
            {
                "full_row": full_row,
                "order": order,
                "fold": order // N_BENIGN_PER_FOLD,
                "prompt": pool[pool_index]["prompt"],
                "prompt_sha256": sha256_text(pool[pool_index]["prompt"]),
            }
        )
    return rows


def dose_zero_audit(model, tokenizer, hook: BoundaryHook, prompts: list[str]) -> dict:
    records = []
    all_pass = True
    for text in prompts:
        chat = format_user_chat(tokenizer, text)
        tokens = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
        length = int(tokens["input_ids"].shape[1])
        hook.enabled = False
        with torch.inference_mode():
            base_logits = model(**tokens, return_dict=True, use_cache=False).logits[0, -1].float().cpu()
        hook.enabled = True
        hook.reset_counters()
        hook.configure(0.0, length - 2, length - 1, None, None)
        with torch.inference_mode():
            hooked_logits = model(**tokens, return_dict=True, use_cache=False).logits[0, -1].float().cpu()
        logits_equal = bool(torch.equal(base_logits, hooked_logits))
        forward_counters = {
            "prefill_calls": hook.prefill_calls,
            "decode_calls": hook.decode_calls,
            "writes_minus1": hook.writes_minus1,
            "writes_tbg": hook.writes_tbg,
        }

        def greedy(hooked: bool) -> list[int]:
            hook.enabled = hooked
            hook.reset_counters()
            if hooked:
                hook.configure(0.0, length - 2, length - 1, None, None)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id
                )
            return generated[0, length:].tolist()

        greedy_base = greedy(False)
        greedy_hooked = greedy(True)  # counters below come from the hooked run
        greedy_prefill = hook.prefill_calls
        greedy_decode = hook.decode_calls
        greedy_writes = hook.writes_minus1 + hook.writes_tbg

        def sampled(hooked: bool, seed: int = 12345) -> list[int]:
            hook.enabled = hooked
            hook.reset_counters()
            if hooked:
                hook.configure(0.0, length - 2, length - 1, None, None)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                generated = model.generate(
                    **tokens,
                    max_new_tokens=8,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                )
            return generated[0, length:].tolist()

        sampled_base = sampled(False)
        sampled_hooked = sampled(True)
        checks = {
            "logits_bitwise_equal": logits_equal,
            "greedy_token_identical": greedy_base == greedy_hooked,
            "sampled_token_identical": sampled_base == sampled_hooked,
            "one_prefill_zero_writes": forward_counters["prefill_calls"] == 1
            and forward_counters["writes_minus1"] == 0
            and forward_counters["writes_tbg"] == 0
            and forward_counters["decode_calls"] == 0,
            "generate_prefill_one_no_writes": greedy_prefill == 1 and greedy_writes == 0 and greedy_decode >= 1,
        }
        passed = all(checks.values())
        all_pass = all_pass and passed
        records.append({"input_length": length, "checks": checks, "counters": forward_counters, "pass": passed})
    return {"all_pass": all_pass, "n_prompts": len(prompts), "prompts": records}


def main() -> None:
    args = parse_args()
    cohort_file = Path(args.x85_cohort)
    cohort = read_json(cohort_file)
    if cohort.get("experiment") != "x85" or cohort.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: invalid x85 cohort")
    feature_paths = [Path(value) for value in sorted(glob.glob(args.features_glob))]
    if len(feature_paths) != 24:
        raise SystemExit(f"FATAL: expected 24 x85 feature shards, found {len(feature_paths)}")

    vectors, shard_audit = load_boundary_states(feature_paths, cohort)
    deltas, prompt_folds, word_folds, prompts, words = build_delta_grid(cohort, vectors)
    direction_vectors, direction_audit = build_directions(deltas, prompt_folds, word_folds)
    lineage = lineage_cosine(direction_vectors, Path(args.x85_directions))

    directions_out = Path(args.directions_out)
    directions_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(directions_out, **direction_vectors)

    target_cells = build_target_cells(cohort)
    pool = read_json(BENIGN_POOL)
    benign_calibration = build_benign_split(
        DATA / "runs" / "x66" / "qwen_seed42_train_indices.json", BENIGN_CALIB_SEED, pool
    )
    benign_evaluation = build_benign_split(
        DATA / "runs" / "x66" / "qwen_seed42_test_indices.json", BENIGN_EVAL_SEED, pool
    )
    # Dose-zero audit prompts: 16 benign train rows disjoint from the calibration set.
    calibration_rows = {row["full_row"] for row in benign_calibration}
    train_benign = sorted(
        index
        for index in (int(value) for value in read_json(DATA / "runs" / "x66" / "qwen_seed42_train_indices.json"))
        if index >= BENIGN_ROW_OFFSET and index not in calibration_rows
    )
    audit_rng = np.random.default_rng(DOSE_ZERO_AUDIT_SEED)
    audit_choice = audit_rng.choice(len(train_benign), size=N_AUDIT_PROMPTS, replace=False)
    audit_prompts = [pool[train_benign[int(index)] - BENIGN_ROW_OFFSET]["prompt"] for index in sorted(audit_choice)]

    model_audit = {"skipped": True}
    status = "unfrozen"
    if not args.skip_model_audit:
        model_source = cohort["model_snapshot"]
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_causal_lm(model_source)
        model.eval()
        hook = BoundaryHook()
        handle = install_hook(model, hook)
        try:
            model_audit = {
                "hook_block": HOOK_BLOCK,
                "intervention_state": INTERVENTION_STATE,
                "dose_zero_equivalence": dose_zero_audit(model, tokenizer, hook, audit_prompts),
            }
        finally:
            handle.remove()
        if not model_audit["dose_zero_equivalence"]["all_pass"]:
            raise SystemExit("FATAL: x86 dose-zero equivalence audit failed")
        status = "frozen"

    manifest = {
        "experiment": "x86",
        "stage": 0,
        "status": status,
        "model_id": MODEL_ID,
        "model_snapshot": cohort["model_snapshot"],
        "x85_cohort": file_record(cohort_file),
        "x85_cohort_sha256": cohort["cohort_sha256"],
        "template_audit": cohort["template_audit"],
        "directions_npz": file_record(directions_out),
        "seeds": {
            "target_graph": TARGET_GRAPH_SEED,
            "benign_calibration": BENIGN_CALIB_SEED,
            "benign_evaluation": BENIGN_EVAL_SEED,
            "random_null": 8602,
            "permutation_null": PERMUTATION_NULL_SEED,
            "dose_zero_audit": DOSE_ZERO_AUDIT_SEED,
        },
        "counts": {
            "target_cells": len(target_cells),
            "target_conditions": len(target_conditions()),
            "benign_calibration": len(benign_calibration),
            "benign_evaluation": len(benign_evaluation),
            "benign_conditions": len(benign_conditions()),
            "target_responses": len(target_cells) * len(target_conditions()) * 10,
            "benign_responses": len(benign_evaluation) * len(benign_conditions()) * 5,
        },
        "audits": {
            "shards": shard_audit,
            "directions": direction_audit,
            "x85_lineage": lineage,
            "model": model_audit,
        },
        "target_conditions": target_conditions(),
        "benign_conditions": benign_conditions(),
        "target_cells": target_cells,
        "benign_calibration": benign_calibration,
        "benign_evaluation": benign_evaluation,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "code_sha256": code_hashes(),
    }
    atomic_json(Path(args.out), manifest)
    print(
        f"x86 manifest ({status}): {len(target_cells)} target cells, "
        f"{len(benign_calibration)}/{len(benign_evaluation)} benign calib/eval, "
        f"directions -> {directions_out}"
    )


if __name__ == "__main__":
    main()
