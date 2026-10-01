"""x83 Stage 1: same-model detection gate for the inserted word.

For every (prompt, position) the frozen checkpoint is shown the inserted text in
a fresh chat and asked which single word was accidentally inserted. The gate
statistic is the teacher-forced probability mass, at the first answer position,
of the inserted word's surface variants (case / leading-space forms), summed.
Also records the greedy answer and the top-10 first-token distribution.
"""

from __future__ import annotations

import argparse
import math

import torch
from transformers import AutoTokenizer

from x83_common import (
    DETECT_INSTRUCTION,
    DETECT_THRESHOLD,
    MODEL_ID,
    POSITIONS,
    RUN,
    atomic_json,
    cohort_path,
    file_record,
    format_user_chat,
    load_causal_lm,
    read_json,
    sha256_text,
    surface_variants,
)

GREEDY_TOKENS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(RUN / "x83_detect.json"))
    return parser.parse_args()


def variant_log_probs(model, tokenizer, prefix_ids: list[int], variants: list[str]) -> list[dict]:
    """Teacher-forced log P(variant tokens | prefix) for each surface variant, one batch."""
    variant_ids = [tokenizer(v, add_special_tokens=False)["input_ids"] for v in variants]
    sequences = [prefix_ids + ids for ids in variant_ids]
    longest = max(len(seq) for seq in sequences)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(sequences), longest), pad, dtype=torch.long)
    attention = torch.zeros((len(sequences), longest), dtype=torch.long)
    for row, seq in enumerate(sequences):
        input_ids[row, longest - len(seq) :] = torch.tensor(seq, dtype=torch.long)
        attention[row, longest - len(seq) :] = 1
    input_ids = input_ids.to(model.device)
    attention = attention.to(model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    results = []
    for row, (variant, ids) in enumerate(zip(variants, variant_ids)):
        k = len(ids)
        total = 0.0
        per_token = []
        for j, token_id in enumerate(ids):
            position = longest - k + j  # index of this token in the padded row
            lp = float(log_probs[row, position - 1, token_id])
            per_token.append(lp)
            total += lp
        results.append(
            {
                "surface": variant,
                "token_ids": [int(t) for t in ids],
                "log_prob": total,
                "prob": math.exp(total),
                "per_token_log_prob": per_token,
            }
        )
    return results


def first_token_top10(model, tokenizer, prefix_ids: list[int]) -> tuple[list[dict], float]:
    input_ids = torch.tensor([prefix_ids], dtype=torch.long).to(model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum())
    top = torch.topk(probs, 10)
    rows = [
        {"token_id": int(i), "decoded": tokenizer.decode([int(i)]), "prob": float(p)}
        for p, i in zip(top.values.tolist(), top.indices.tolist())
    ]
    return rows, entropy


def greedy_answer(model, tokenizer, prefix_ids: list[int]) -> str:
    input_ids = torch.tensor([prefix_ids], dtype=torch.long).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=GREEDY_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    cohort = read_json(cohort_path())
    if cohort.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: cohort model mismatch")
    model_source = cohort["model_snapshot"]
    units = list(cohort["units"])
    if args.limit:
        units = units[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_lm(model_source)

    rows = []
    for index, unit in enumerate(units, start=1):
        prompt_id = int(unit["id"])
        word = unit["word"]
        variants = surface_variants(word)
        record = {"id": prompt_id, "word": word, "category": unit["category"], "positions": {}}
        for position in POSITIONS:
            text = unit["prompts"][position]
            if sha256_text(text) != unit["prompt_sha256"][position]:
                raise SystemExit(f"FATAL: prompt hash mismatch {prompt_id}/{position}")
            chat = format_user_chat(tokenizer, DETECT_INSTRUCTION.format(text=text))
            prefix_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
            variant_rows = variant_log_probs(model, tokenizer, prefix_ids, variants)
            p_detect = float(sum(v["prob"] for v in variant_rows))
            top10, entropy = first_token_top10(model, tokenizer, prefix_ids)
            greedy = greedy_answer(model, tokenizer, prefix_ids)
            record["positions"][position] = {
                "prompt_sha256": unit["prompt_sha256"][position],
                "rendered_chat_sha256": sha256_text(chat),
                "p_detect": p_detect,
                "admitted": p_detect > DETECT_THRESHOLD,
                "variants": variant_rows,
                "first_token_top10": top10,
                "first_token_entropy_nats": entropy,
                "greedy_answer": greedy,
                "greedy_matches": greedy.strip().strip(".\"'").casefold() == word,
            }
        rows.append(record)
        if index % 10 == 0 or index == len(units):
            print(f"x83 detect: {index}/{len(units)}", flush=True)

    admitted = {p: sum(r["positions"][p]["admitted"] for r in rows) for p in POSITIONS}
    common = sum(all(r["positions"][p]["admitted"] for p in POSITIONS) for r in rows)
    output = {
        "experiment": "x83",
        "stage": 1,
        "model_id": MODEL_ID,
        "model_source": model_source,
        "cohort_sha256": cohort["cohort_sha256"],
        "input": file_record(cohort_path()),
        "config": {
            "threshold": DETECT_THRESHOLD,
            "statistic": "sum of teacher-forced sequence probabilities over surface variants",
            "variants": "word, Word, WORD, and the same three with a leading space",
            "greedy_tokens": GREEDY_TOKENS,
            "thinking": False,
        },
        "counts": {"prompts": len(rows), "admitted": admitted, "admitted_all_three": common},
        "rows": rows,
    }
    atomic_json(args.out, output)
    print(f"x83 detect: admitted {admitted}, common {common} -> {args.out}")


if __name__ == "__main__":
    main()
