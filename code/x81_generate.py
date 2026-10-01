"""x81 Stage 4a: generate the new A/B/C arms with the x63 distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x81_common import (
    MODEL_ID,
    N_RESPONSES,
    RUN,
    atomic_json,
    file_record,
    format_user_chat,
    load_causal_lm,
    read_json,
    sha256_text,
    stable_seed,
)

MAX_NEW = 2048
RETRY_NEW = 4096
TEMPERATURE = 1.0
TOP_P = 0.9
TOP_K = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=("A", "B", "C", "P", "D", "E"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--arms", default=str(RUN / "x81_arms.json"))
    parser.add_argument("--out", help="default: data/runs/x81/x81_generations_CONDITION.json")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def eos_id_set(tokenizer, model) -> set[int]:
    result = set()
    for value in (tokenizer.eos_token_id, model.generation_config.eos_token_id):
        if value is None:
            continue
        result.update(value if isinstance(value, (list, tuple)) else [value])
    return {int(value) for value in result}


def decode_sequences(sequences, tokenizer, eos_ids: set[int], max_new: int):
    texts, lengths, truncated = [], [], []
    for sequence in sequences:
        ids = sequence.tolist()
        hits = [index for index, token_id in enumerate(ids) if token_id in eos_ids]
        if hits:
            length, is_truncated = hits[0] + 1, False
        else:
            length, is_truncated = len(ids), len(ids) >= max_new
        texts.append(tokenizer.decode(sequence, skip_special_tokens=True).strip())
        lengths.append(length)
        truncated.append(is_truncated)
    return texts, lengths, truncated


def generate_row(model, tokenizer, eos_ids, chat: str, max_new: int, seed: int):
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
            num_return_sequences=N_RESPONSES,
            pad_token_id=tokenizer.pad_token_id,
        )
    continuation = generated[:, inputs["input_ids"].shape[1] :]
    return decode_sequences(continuation, tokenizer, eos_ids, max_new)


def valid_completed(row: dict, expected_sha: str, arms_sha: str) -> bool:
    return (
        row.get("prompt_sha256") == expected_sha
        and row.get("arms_sha256") == arms_sha
        and len(row.get("llm_responses", [])) == N_RESPONSES
        and len(row.get("gen_tokens", [])) == N_RESPONSES
        and len(row.get("truncated", [])) == N_RESPONSES
        and all(isinstance(value, str) and value.strip() for value in row["llm_responses"])
    )


def main() -> None:
    args = parse_args()
    arms_path = Path(args.arms)
    out_path = Path(args.out) if args.out else RUN / f"x81_generations_{args.condition}.json"
    if not arms_path.exists():
        raise SystemExit("FATAL: run x81_build_arms.py first")
    manifest = read_json(arms_path)
    if manifest.get("status") != "pass":
        raise SystemExit("FATAL: x81 support gate did not pass")
    if manifest.get("model_id") != args.model:
        raise SystemExit("FATAL: model ID differs from arm manifest")
    if args.model != MODEL_ID:
        raise SystemExit(f"FATAL: x81 is frozen to {MODEL_ID}")
    model_source = manifest["model_source"]
    arms = list(manifest["arms"])
    if args.limit:
        arms = arms[: args.limit]
    expected = {int(row["id"]): row for row in arms}

    done: dict[int, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("condition") != args.condition or prior.get("arms_sha256") != manifest["arms_sha256"]:
            raise SystemExit("FATAL: resume output belongs to another condition/arm manifest")
        for row in prior.get("rows", []):
            prompt_id = int(row["id"])
            if prompt_id not in expected:
                continue
            expected_sha = expected[prompt_id]["prompt_sha256"][args.condition]
            if not valid_completed(row, expected_sha, manifest["arms_sha256"]):
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
        rows = [done[prompt_id] for prompt_id in sorted(done)]
        atomic_json(
            out_path,
            {
                "experiment": "x81",
                "stage": "4a",
                "model_id": args.model,
                "model_source": model_source,
                "model_manifest_sha256": manifest["model_manifest_sha256"],
                "condition": args.condition,
                "arms_sha256": manifest["arms_sha256"],
                "input": file_record(arms_path),
                "config": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_return_sequences": N_RESPONSES,
                    "max_new_tokens": MAX_NEW,
                    "retry_max_new_tokens": RETRY_NEW,
                    "thinking": False,
                    "seed_policy": "sha256-derived per-(prompt,condition), base 42",
                },
                "rows": rows,
            },
        )

    todo = [row for row in arms if int(row["id"]) not in done]
    for index, arm in enumerate(todo, start=1):
        prompt_id = int(arm["id"])
        prompt = arm["prompts"][args.condition]
        if sha256_text(prompt) != arm["prompt_sha256"][args.condition]:
            raise SystemExit(f"FATAL: arm text hash mismatch for {prompt_id}")
        chat = format_user_chat(tokenizer, prompt)
        seed = stable_seed(prompt_id, args.condition)
        texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(prompt_id, f"{args.condition}:retry4096")
            texts, lengths, truncated = generate_row(
                model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed
            )
        if len(texts) != N_RESPONSES or any(not value.strip() for value in texts):
            raise SystemExit(f"FATAL: incomplete/nonempty generation row {prompt_id}")
        done[prompt_id] = {
            "id": prompt_id,
            "condition": args.condition,
            "prompt": prompt,
            "prompt_sha256": arm["prompt_sha256"][args.condition],
            "rendered_chat_sha256": sha256_text(chat),
            "arms_sha256": manifest["arms_sha256"],
            "seed": seed,
            "used_retry": used_retry,
            "retry_seed": retry_seed,
            "llm_responses": texts,
            "response_sha256": [sha256_text(value) for value in texts],
            "gen_tokens": lengths,
            "truncated": truncated,
        }
        checkpoint()
        print(f"x81 generate {args.condition}: {index}/{len(todo)} new", flush=True)
    checkpoint()
    print(f"x81 generation {args.condition}: {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()
