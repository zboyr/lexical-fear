"""x81 Stage 0: audit x66 lineage, split mapping, and local model identity."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

from transformers import AutoTokenizer

from x81_common import (
    BLANK,
    FILL_INSTRUCTION,
    LABELS_PATH,
    MASK,
    MODEL_ID,
    PROMPTS_PATH,
    RUN,
    SEMANTIC_INSTRUCTION,
    TEST_INDICES_PATH,
    TRAIN_INDICES_PATH,
    X64_PATH,
    atomic_json,
    file_record,
    format_user_chat,
    load_historical_refusals,
    load_prompt_rows,
    load_split_rows,
    load_x64_refusals,
    model_snapshot_record,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--model-snapshot", required=True, help="resolved local HF snapshot to fingerprint and use"
    )
    parser.add_argument("--out", default=str(RUN / "x81_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != MODEL_ID:
        raise SystemExit(f"FATAL: x81 is frozen to {MODEL_ID}")
    prompts = load_prompt_rows()
    train = load_split_rows(TRAIN_INDICES_PATH)
    test = load_split_rows(TEST_INDICES_PATH)
    train_ids = {int(row["id"]) for row in train}
    test_ids = {int(row["id"]) for row in test}
    if train_ids & test_ids:
        raise SystemExit("FATAL: x66 train/test harmful prompt IDs overlap")
    if train_ids | test_ids != set(prompts):
        raise SystemExit("FATAL: x66 seed-42 harmful split does not cover x63 retained prompts")

    historical = load_historical_refusals()
    x64 = load_x64_refusals()
    if set(historical) != set(prompts) or set(x64) != set(prompts):
        raise SystemExit("FATAL: prompt/label/x64 ID sets differ")
    mismatches = [prompt_id for prompt_id in prompts if historical[prompt_id] != x64[prompt_id]]
    if mismatches:
        raise SystemExit(f"FATAL: x65/x64 refusal arrays differ for {mismatches[:10]}")

    snapshot = str(Path(args.model_snapshot).resolve())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    blank_ids = tokenizer.encode(BLANK, add_special_tokens=False)
    fill_example = FILL_INSTRUCTION.format(masked_prompt=f"alpha {MASK} beta")
    fill_chat = format_user_chat(tokenizer, fill_example)
    fill_chat_ids = tokenizer.encode(fill_chat, add_special_tokens=False)
    fill_with_blank_ids = tokenizer.encode(fill_chat + BLANK, add_special_tokens=False)
    blank_is_one_answer_token = (
        fill_with_blank_ids[: len(fill_chat_ids)] == fill_chat_ids
        and len(fill_with_blank_ids) == len(fill_chat_ids) + 1
    )
    if len(blank_ids) != 1 or not blank_is_one_answer_token:
        raise SystemExit(
            "FATAL: blank sentinel is not one stable token at the frozen answer position"
        )
    special_values = set(tokenizer.all_special_tokens)
    if MASK in special_values:
        raise SystemExit("FATAL: x81 mask unexpectedly collides with tokenizer special token")

    versions = {}
    for package in ("torch", "transformers", "tokenizers", "scikit-learn", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    result = {
        "experiment": "x81",
        "stage": 0,
        "status": "pass",
        "model_id": args.model,
        "model_snapshot": model_snapshot_record(snapshot),
        "counts": {
            "all_harmful": len(prompts),
            "discovery": len(train),
            "evaluation": len(test),
        },
        "split_disjoint": True,
        "split_complete": True,
        "historical_equals_x64": True,
        "blank_token_id_isolated": int(blank_ids[0]),
        "blank_token_id_at_answer": int(fill_with_blank_ids[-1]),
        "mask": MASK,
        "templates": {
            "fill_sha256": sha256_text(FILL_INSTRUCTION),
            "semantic_sha256": sha256_text(SEMANTIC_INSTRUCTION),
            "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
            "rendered_fill_example_sha256": sha256_text(fill_chat),
        },
        "inputs": {
            "prompts": file_record(PROMPTS_PATH),
            "labels": file_record(LABELS_PATH),
            "x64": file_record(X64_PATH),
            "train_indices": file_record(TRAIN_INDICES_PATH),
            "test_indices": file_record(TEST_INDICES_PATH),
        },
        "versions": versions,
        "note": "Evaluation refusal values were integrity-checked here but are not emitted.",
    }
    atomic_json(args.out, result)
    print(f"x81 audit PASS -> {args.out}")


if __name__ == "__main__":
    main()
