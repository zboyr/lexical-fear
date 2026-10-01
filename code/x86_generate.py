"""x86 Stages 2--4: resumable target or benign generation with boundary hooks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, decode_sequences, eos_id_set
from x86_common import (
    BENIGN_RESPONSES,
    BOUNDARY_POSITIONS,
    MODEL_ID,
    N_RESPONSES,
    RUN,
    BoundaryHook,
    atomic_json,
    file_record,
    format_user_chat,
    install_hook,
    load_causal_lm,
    load_direction_vectors,
    read_json,
    sha256_text,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--manifest", default=str(RUN / "x86_manifest.json"))
    parser.add_argument("--calibration", default=str(RUN / "x86_kl_calibration.json"))
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def resolve_condition(manifest: dict, calibration: dict, name: str) -> tuple[str, dict, float]:
    all_conditions = {
        row["condition"]: ("target", row) for row in manifest["target_conditions"]
    }
    all_conditions.update(
        {row["condition"]: ("benign", row) for row in manifest["benign_conditions"]}
    )
    if name not in all_conditions:
        raise SystemExit(f"FATAL: unknown x86 condition {name}")
    scope, row = all_conditions[name]
    a_star = float(calibration["admitted_a_star"])
    sign = float(row.get("sign", +1.0))
    if row["dose_kind"] == "zero":
        dose = 0.0
    elif row["dose_kind"] == "half":
        dose = sign * a_star / 2.0
    elif row["dose_kind"] == "full":
        dose = sign * a_star
    elif row["dose_kind"] == "null_full":
        sign_name = "plus" if sign > 0 else "minus"
        record = calibration["null_calibration"][row["family"]][sign_name]
        if record["status"] != "pass":
            raise SystemExit(f"FATAL: null calibration failed for {name}")
        dose = float(record["matched_signed_dose"])
    else:
        raise SystemExit(f"FATAL: unknown dose kind {row['dose_kind']}")
    return scope, row, dose


def generate_batch(model, tokenizer, eos_ids, chat: str, n: int, max_new: int, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = generated[:, inputs["input_ids"].shape[1] :]
    return decode_sequences(continuation, tokenizer, eos_ids, max_new)


def valid_completed(row: dict, expected_id: int, condition: str, run_hash: str, n: int) -> bool:
    return (
        int(row.get("id", -1)) == expected_id
        and row.get("condition") == condition
        and row.get("run_hash") == run_hash
        and len(row.get("llm_responses", [])) == n
        and len(row.get("response_sha256", [])) == n
        and len(row.get("gen_tokens", [])) == n
        and len(row.get("truncated", [])) == n
        and all(isinstance(value, str) and value.strip() for value in row["llm_responses"])
    )


def main() -> None:
    args = parse_args()
    manifest_path, calibration_path = Path(args.manifest), Path(args.calibration)
    manifest, calibration = read_json(manifest_path), read_json(calibration_path)
    if manifest.get("status") != "frozen" or calibration.get("status") != "pass":
        raise SystemExit("FATAL: x86 manifest/calibration gates have not passed")
    if manifest.get("model_id") != MODEL_ID or calibration.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: x86 model mismatch")
    scope, condition, signed_dose = resolve_condition(manifest, calibration, args.condition)
    vectors = load_direction_vectors(manifest)
    n_responses = N_RESPONSES if scope == "target" else BENIGN_RESPONSES
    if scope == "target":
        source_rows = list(manifest["target_cells"])
    else:
        source_rows = list(manifest["benign_evaluation"])
    if args.limit:
        source_rows = source_rows[: args.limit]
    expected_ids = {
        int(row["target_id"] if scope == "target" else row["full_row"])
        for row in source_rows
    }
    out_path = (
        Path(args.out)
        if args.out
        else RUN / f"x86_generations_{args.condition}.json"
    )
    run_hash = sha256_text(
        f"{manifest_path.resolve()}|{file_record(manifest_path)['sha256']}|"
        f"{file_record(calibration_path)['sha256']}|{args.condition}|{signed_dose}"
    )

    done: dict[int, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("run_hash") != run_hash or prior.get("condition") != args.condition:
            raise SystemExit("FATAL: generation resume belongs to another x86 run")
        for row in prior.get("rows", []):
            row_id = int(row["id"])
            if row_id not in expected_ids:
                raise SystemExit(f"FATAL: resume row {row_id} is outside selected scope")
            if not valid_completed(row, row_id, args.condition, run_hash, n_responses):
                raise SystemExit(f"FATAL: invalid generation resume row {row_id}")
            if row_id in done:
                raise SystemExit(f"FATAL: duplicate generation resume row {row_id}")
            done[row_id] = row

    model_source = manifest["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_lm(model_source)
    model.eval()
    eos_ids = eos_id_set(tokenizer, model)
    hook = BoundaryHook()
    hook.enabled = True
    handle = install_hook(model, hook)

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "experiment": "x86",
                "stage": 3 if scope == "target" else 4,
                "scope": scope,
                "condition": args.condition,
                "condition_config": condition,
                "signed_dose": signed_dose,
                "n_responses": n_responses,
                "model_id": MODEL_ID,
                "model_source": model_source,
                "manifest": file_record(manifest_path),
                "calibration": file_record(calibration_path),
                "run_hash": run_hash,
                "sampling": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "max_new_tokens": MAX_NEW,
                    "retry_max_new_tokens": RETRY_NEW,
                    "common_random_numbers": "same per-(scope,row,arm) seed across conditions",
                },
                "rows": [done[key] for key in sorted(done)],
            },
        )

    try:
        for offset, source in enumerate(source_rows, start=1):
            if scope == "target":
                row_id = int(source["target_id"])
                arm = condition["arm"]
                info = source[arm]
                text, fold = info["prompt"], int(source["fold"])
                prompt_hash = info["prompt_sha256"]
                chat_hash = info["rendered_chat_sha256"]
                minus1_idx, tbg_idx = int(info["tbg_minus1_idx"]), int(info["tbg_idx"])
                seed = stable_seed("target", row_id, arm)
                metadata = {
                    "target_id": row_id,
                    "prompt_id": info["prompt_id"],
                    "prompt_index": source["prompt_index"],
                    "word_index": source["word_index"],
                    "stratum": source["stratum"],
                    "category": info["category"],
                    "harm_word": info["harm_word"],
                    "neutral_word": info["neutral_word"],
                    "arm": arm,
                }
            else:
                row_id = int(source["full_row"])
                text, fold = source["prompt"], int(source["fold"])
                prompt_hash = sha256_text(text)
                seed = stable_seed("benign", row_id)
                metadata = {"full_row": row_id, "benign_order": source["order"]}
                chat_hash = None
                minus1_idx = tbg_idx = -1
            if row_id in done:
                continue
            chat = format_user_chat(tokenizer, text)
            if chat_hash is not None and sha256_text(chat) != chat_hash:
                raise SystemExit(f"FATAL: rendered chat mismatch for row {row_id}")
            input_length = len(tokenizer(chat, add_special_tokens=False)["input_ids"])
            if scope == "benign":
                minus1_idx, tbg_idx = input_length - 2, input_length - 1
            elif (minus1_idx, tbg_idx) != (input_length - 2, input_length - 1):
                raise SystemExit(f"FATAL: target boundary indices changed for row {row_id}")
            family = condition["family"]
            positions = tuple(condition["positions"])
            vec_minus1 = (
                torch.from_numpy(vectors[f"{family}_f{fold}_tbg_minus1"])
                if "tbg_minus1" in positions
                else None
            )
            vec_tbg = (
                torch.from_numpy(vectors[f"{family}_f{fold}_tbg"])
                if "tbg" in positions
                else None
            )
            hook.reset_counters()
            hook.configure(signed_dose, minus1_idx, tbg_idx, vec_minus1, vec_tbg)
            texts, lengths, truncated = generate_batch(
                model, tokenizer, eos_ids, chat, n_responses, MAX_NEW, seed
            )
            first_counters = {
                "prefill_calls": hook.prefill_calls,
                "decode_calls": hook.decode_calls,
                "writes_minus1": hook.writes_minus1,
                "writes_tbg": hook.writes_tbg,
            }
            used_retry = any(truncated)
            retry_seed = None
            retry_counters = None
            if used_retry:
                retry_seed = stable_seed("retry4096", scope, row_id, condition.get("arm", "B"))
                hook.reset_counters()
                texts, lengths, truncated = generate_batch(
                    model, tokenizer, eos_ids, chat, n_responses, RETRY_NEW, retry_seed
                )
                retry_counters = {
                    "prefill_calls": hook.prefill_calls,
                    "decode_calls": hook.decode_calls,
                    "writes_minus1": hook.writes_minus1,
                    "writes_tbg": hook.writes_tbg,
                }
            if len(texts) != n_responses or any(not value.strip() for value in texts):
                raise SystemExit(f"FATAL: incomplete/empty generation row {row_id}")
            expected_writes_minus1 = int(signed_dose != 0 and vec_minus1 is not None)
            expected_writes_tbg = int(signed_dose != 0 and vec_tbg is not None)
            for attempt_name, counters in (
                ("first", first_counters),
                ("retry", retry_counters),
            ):
                if counters is None:
                    continue
                if (
                    counters["prefill_calls"] != 1
                    or counters["writes_minus1"] != expected_writes_minus1
                    or counters["writes_tbg"] != expected_writes_tbg
                ):
                    raise SystemExit(
                        f"FATAL: {attempt_name} hook counter mismatch for row {row_id}: {counters}"
                    )
            done[row_id] = {
                "id": row_id,
                "condition": args.condition,
                "scope": scope,
                "fold": fold,
                "prompt": text,
                "prompt_sha256": prompt_hash,
                "rendered_chat_sha256": sha256_text(chat),
                "input_length": input_length,
                "family": family,
                "positions": list(positions),
                "signed_dose": signed_dose,
                "seed": seed,
                "used_retry": used_retry,
                "retry_seed": retry_seed,
                "hook_counters": first_counters,
                "retry_hook_counters": retry_counters,
                "llm_responses": texts,
                "response_sha256": [sha256_text(value) for value in texts],
                "gen_tokens": lengths,
                "truncated": truncated,
                "run_hash": run_hash,
                **metadata,
            }
            checkpoint()
            print(f"x86 generate {args.condition}: {offset}/{len(source_rows)}", flush=True)
    finally:
        handle.remove()
    checkpoint()
    print(f"x86 generation {args.condition}: {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()
