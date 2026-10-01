"""x85 Stage 2: generate the sparse O/H/N behavior cells under the x83 contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from x81_generate import (
    MAX_NEW,
    RETRY_NEW,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    eos_id_set,
    generate_row,
)
from x85_common import (
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
    parser.add_argument("--cohort", default=str(cohort_path()))
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def valid_completed(row: dict, cell: dict, cohort_sha: str) -> bool:
    return (
        int(row.get("id", -1)) == int(cell["behavior_id"])
        and row.get("cell_id") == cell["cell_id"]
        and row.get("prompt_sha256") == cell["prompt_sha256"]
        and row.get("arms_sha256") == cohort_sha
        and len(row.get("llm_responses", [])) == N_RESPONSES
        and len(row.get("response_sha256", [])) == N_RESPONSES
        and len(row.get("gen_tokens", [])) == N_RESPONSES
        and len(row.get("truncated", [])) == N_RESPONSES
        and all(isinstance(value, str) and value.strip() for value in row["llm_responses"])
    )


def main() -> None:
    args = parse_args()
    cohort_file = Path(args.cohort)
    manifest = read_json(cohort_file)
    if manifest.get("experiment") != "x85" or manifest.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: invalid x85 cohort")
    cells = [
        cell
        for cell in manifest["cells"]
        if cell["behavior"] and cell["condition"] == args.condition
    ]
    cells.sort(key=lambda row: int(row["behavior_id"]))
    if args.limit:
        cells = cells[: args.limit]
    expected = {int(cell["behavior_id"]): cell for cell in cells}
    out_path = Path(args.out) if args.out else RUN / f"x85_generations_{args.condition}.json"
    cohort_sha = manifest["cohort_sha256"]

    done: dict[int, dict] = {}
    if out_path.exists():
        prior = read_json(out_path)
        if prior.get("condition") != args.condition or prior.get("cohort_sha256") != cohort_sha:
            raise SystemExit("FATAL: resume output belongs to another condition/cohort")
        for row in prior.get("rows", []):
            behavior_id = int(row["id"])
            if behavior_id not in expected:
                continue
            if not valid_completed(row, expected[behavior_id], cohort_sha):
                raise SystemExit(f"FATAL: invalid completed generation row {behavior_id}")
            if behavior_id in done:
                raise SystemExit(f"FATAL: duplicate completed generation row {behavior_id}")
            done[behavior_id] = row

    model_source = manifest["model_snapshot"]
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
                "experiment": "x85",
                "stage": 2,
                "model_id": MODEL_ID,
                "model_source": model_source,
                "condition": args.condition,
                "cohort_sha256": cohort_sha,
                "arms_sha256": cohort_sha,
                "input": file_record(cohort_file),
                "config": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_return_sequences": N_RESPONSES,
                    "max_new_tokens": MAX_NEW,
                    "retry_max_new_tokens": RETRY_NEW,
                    "thinking": False,
                    "seed_policy": "sha256-derived per cell in the x85 namespace, base 85",
                },
                "rows": [done[key] for key in sorted(done)],
            },
        )

    todo = [cell for cell in cells if int(cell["behavior_id"]) not in done]
    for index, cell in enumerate(todo, start=1):
        behavior_id = int(cell["behavior_id"])
        text = cell["prompt"]
        if sha256_text(text) != cell["prompt_sha256"]:
            raise SystemExit(f"FATAL: prompt hash mismatch for {cell['cell_id']}")
        chat = format_user_chat(tokenizer, text)
        if sha256_text(chat) != cell["rendered_chat_sha256"]:
            raise SystemExit(f"FATAL: rendered chat mismatch for {cell['cell_id']}")
        seed = stable_seed(cell["cell_id"], args.condition)
        texts, lengths, truncated = generate_row(model, tokenizer, eos_ids, chat, MAX_NEW, seed)
        used_retry = any(truncated)
        retry_seed = None
        if used_retry:
            retry_seed = stable_seed(cell["cell_id"], f"{args.condition}:retry4096")
            texts, lengths, truncated = generate_row(
                model, tokenizer, eos_ids, chat, RETRY_NEW, retry_seed
            )
        if len(texts) != N_RESPONSES or any(not value.strip() for value in texts):
            raise SystemExit(f"FATAL: incomplete/empty generation row {cell['cell_id']}")
        done[behavior_id] = {
            "id": behavior_id,
            "cell_id": cell["cell_id"],
            "condition": args.condition,
            "prompt_id": int(cell["prompt_id"]),
            "word_index": cell["word_index"],
            "harm_word": cell["harm_word"],
            "neutral_word": cell["neutral_word"],
            "prompt": text,
            "prompt_sha256": cell["prompt_sha256"],
            "rendered_chat_sha256": cell["rendered_chat_sha256"],
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
        print(f"x85 generate {args.condition}: {index}/{len(todo)} new", flush=True)
    checkpoint()
    print(f"x85 generation {args.condition}: {len(done)} rows -> {out_path}")


if __name__ == "__main__":
    main()

