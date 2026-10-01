"""x83 Stage 2: generate O / S / M / E responses with the x63 Qwen sampling contract.

Identical engine, settings, and resume semantics to x81_generate (whose
functions are reused); O is regenerated here, unlike x81.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from x81_generate import MAX_NEW, RETRY_NEW, TEMPERATURE, TOP_K, TOP_P, eos_id_set, generate_row, valid_completed
from x83_common import (
    CONDITIONS,
    MODEL_ID,
    N_RESPONSES,
    RUN,
    atomic_json,
    cohort_path,
    file_record,
    format_user_chat,
    load_causal_lm,
    read_json,
    sha256_text,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--out", help="default: data/runs/x83/x83_generations_COND.json")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out) if args.out else RUN / f"x83_generations_{args.condition}.json"
    cohort = read_json(cohort_path())
    if cohort.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: cohort model mismatch")
    model_source = cohort["model_snapshot"]
    units = list(cohort["units"])
    if args.limit:
        units = units[: args.limit]
    expected = {int(unit["id"]): unit for unit in units}
    cohort_sha = cohort["cohort_sha256"]

    done: dict[int, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("condition") != args.condition or prior.get("cohort_sha256") != cohort_sha:
            raise SystemExit("FATAL: resume output belongs to another condition/cohort")
        for row in prior.get("rows", []):
            prompt_id = int(row["id"])
            if prompt_id not in expected:
                continue
            expected_sha = expected[prompt_id]["prompt_sha256"][args.condition]
            if not valid_completed(row, expected_sha, cohort_sha):
                raise SystemExit(f"FATAL: invalid completed resume row {prompt_id}")
            if prompt_id in done:
                raise SystemExit(f"FATAL: duplicate completed resume row {prompt_id}")
            done[prompt_id] = row

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_lm(model_source)
    eos_ids = eos_id_set(tokenizer, model)

    def checkpoint() -> None:
        atomic_json(
            out_path,
            {
                "experiment": "x83",
                "stage": 2,
                "model_id": MODEL_ID,
                "model_source": model_source,
                "condition": args.condition,
                "cohort_sha256": cohort_sha,
                "arms_sha256": cohort_sha,  # field name kept for the shared judge/validators
                "input": file_record(cohort_path()),
                "config": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_return_sequences": N_RESPONSES,
                    "max_new_tokens": MAX_NEW,
                    "retry_max_new_tokens": RETRY_NEW,
                    "thinking": False,
                    "seed_policy": "sha256-derived per-(prompt,condition), x83 namespace, base 42",
                },
                "rows": [done[prompt_id] for prompt_id in sorted(done)],
            },
        )

    todo = [unit for unit in units if int(unit["id"]) not in done]
    for index, unit in enumerate(todo, start=1):
        prompt_id = int(unit["id"])
        prompt = unit["prompts"][args.condition]
        if sha256_text(prompt) != unit["prompt_sha256"][args.condition]:
            raise SystemExit(f"FATAL: prompt hash mismatch for {prompt_id}")
        chat = format_user_chat(tokenizer, prompt)
        seed = stable_seed(prompt_id, args.condition)
        texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(prompt_id, f"{args.condition}:retry4096")
            texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed)
        if len(texts) != N_RESPONSES or any(not value.strip() for value in texts):
            raise SystemExit(f"FATAL: incomplete/empty generation row {prompt_id}")
        done[prompt_id] = {
            "id": prompt_id,
            "condition": args.condition,
            "word": unit["word"],
            "prompt": prompt,
            "prompt_sha256": unit["prompt_sha256"][args.condition],
            "rendered_chat_sha256": sha256_text(chat),
            "arms_sha256": cohort_sha,
            "seed": seed,
            "used_retry": used_retry,
            "retry_seed": retry_seed,
            "llm_responses": texts,
            "response_sha256": [sha256_text(value) for value in texts],
            "gen_tokens": lengths,
            "truncated": truncated,
        }
        checkpoint()
        print(f"x83 generate {args.condition}: {index}/{len(todo)} new", flush=True)
    checkpoint()
    print(f"x83 generation {args.condition}: {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()
