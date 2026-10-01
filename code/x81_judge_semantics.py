"""x81 Stage 3a: same-checkpoint semantic judgments for every top fill."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x81_common import (
    BLANK,
    MODEL_ID,
    RUN,
    SEMANTIC_INSTRUCTION,
    TOP_K_FILLS,
    atomic_json,
    file_record,
    format_user_chat,
    load_causal_lm,
    match_case,
    read_json,
    replace_span,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--fills", default=str(RUN / "x81_fills.json"))
    parser.add_argument("--out", default=str(RUN / "x81_semantics.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="maximum number of prompts")
    return parser.parse_args()


def exact_decision(raw: str) -> str:
    value = raw.strip().upper()
    return value if value in {"SAME", "CHANGED"} else "CHANGED"


def substituted_prompt(row: dict, candidate: dict) -> str:
    surface = candidate["surface"]
    replacement = "" if surface == BLANK else match_case(surface, row["target_word"])
    return replace_span(row["prompt"], int(row["start"]), int(row["end"]), replacement)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("FATAL: --batch-size must be positive")
    fills_path = Path(args.fills)
    if not fills_path.exists():
        raise SystemExit("FATAL: run x81_score_fills.py first")
    fills = read_json(fills_path)
    if fills.get("model_id") != args.model or fills.get("uses_evaluation_refusal_labels") is not False:
        raise SystemExit("FATAL: invalid fill manifest")
    if args.model != MODEL_ID:
        raise SystemExit(f"FATAL: x81 is frozen to {MODEL_ID}")
    model_source = fills["model_source"]
    source_rows = list(fills["rows"])
    if args.limit:
        source_rows = source_rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_lm(model_source)

    work = []
    for row in source_rows:
        if len(row["candidates"]) != TOP_K_FILLS:
            raise SystemExit(f"FATAL: prompt {row['id']} does not have {TOP_K_FILLS} fills")
        for candidate in row["candidates"]:
            modified = substituted_prompt(row, candidate)
            instruction = SEMANTIC_INSTRUCTION.format(original=row["prompt"], modified=modified)
            work.append((row, candidate, modified, format_user_chat(tokenizer, instruction)))

    by_id: dict[int, list[dict]] = {int(row["id"]): [] for row in source_rows}
    for offset in range(0, len(work), args.batch_size):
        batch = work[offset : offset + args.batch_size]
        rendered = [item[3] for item in batch]
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer_ids = generated[:, inputs["input_ids"].shape[1] :]
        raw_answers = tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
        for (row, candidate, modified, chat), raw in zip(batch, raw_answers):
            parsed = exact_decision(raw)
            eligible = bool(candidate["lexical_candidate"])
            by_id[int(row["id"])].append(
                {
                    **candidate,
                    "modified_prompt": modified,
                    "modified_prompt_sha256": sha256_text(modified),
                    "semantic_raw": raw,
                    "semantic_parsed": parsed,
                    "eligible_for_same": eligible,
                    "decision": parsed if eligible else "CHANGED",
                    "rendered_chat_sha256": sha256_text(chat),
                }
            )
        print(f"x81 semantics: {min(offset + args.batch_size, len(work))}/{len(work)}", flush=True)

    rows = []
    fills_by_id = {int(row["id"]): row for row in source_rows}
    for prompt_id, candidates in by_id.items():
        candidates.sort(key=lambda row: int(row["rank"]))
        if len(candidates) != TOP_K_FILLS:
            raise SystemExit(f"FATAL: incomplete semantic judgments for {prompt_id}")
        source = fills_by_id[prompt_id]
        rows.append(
            {
                "id": prompt_id,
                "prompt": source["prompt"],
                "prompt_sha256": source["prompt_sha256"],
                "target_word": source["target_word"],
                "target_word_lower": source["target_word_lower"],
                "start": source["start"],
                "end": source["end"],
                "topk_mass": source["topk_mass"],
                "entropy_nats": source["entropy_nats"],
                "candidates": candidates,
            }
        )

    output = {
        "experiment": "x81",
        "stage": "3a",
        "model_id": args.model,
        "model_source": model_source,
        "model_manifest_sha256": fills["model_manifest_sha256"],
        "uses_evaluation_refusal_labels": False,
        "config": {
            "sampling": False,
            "thinking": False,
            "max_new_tokens": 8,
            "parser": "exact stripped SAME or CHANGED; otherwise CHANGED",
            "nonlexical_candidate_policy": "record model answer but force CHANGED",
        },
        "input": file_record(fills_path),
        "template_sha256": sha256_text(SEMANTIC_INSTRUCTION),
        "rows": rows,
    }
    atomic_json(args.out, output)
    print(f"x81 semantic judgments: {len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
