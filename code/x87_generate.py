"""Generate deterministic responses for frozen OR-Bench/StrongREJECT rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x87_common import (
    ALL_CONDITIONS,
    BENCHMARKS,
    MAX_NEW_TOKENS,
    MODEL_ID,
    RUN,
    assert_qwen_boundary,
    atomic_json,
    file_record,
    format_user_chat,
    load_causal_lm,
    public_condition_metadata,
    read_json,
    setup_condition,
    sha256_text,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument("--manifest", default=str(RUN / "x87_manifest.json"))
    parser.add_argument("--x86-directions", default="data/runs/x86/x86_directions.npz")
    parser.add_argument("--x86-calibration", default="data/runs/x86/x86_kl_calibration.json")
    parser.add_argument("--x75-adapter", default="data/runs/x87/x75_qwen_refusal_weighted_true.pt")
    parser.add_argument("--out")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def decode_new(tokenizer, sequences, prompt_len: int) -> list[str]:
    return [
        tokenizer.decode(row[prompt_len:], skip_special_tokens=True).strip()
        for row in sequences
    ]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "frozen" or manifest.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: invalid x87 manifest")
    model_source = manifest["model_snapshot"]["snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_lm(model_source)
    setup = setup_condition(
        model,
        args.condition,
        x86_directions=args.x86_directions,
        x86_calibration=args.x86_calibration,
        x75_adapter=args.x75_adapter,
    )
    model.eval()
    rows = list(manifest["rows"])
    if args.limit:
        rows = rows[: args.limit]
    output_path = Path(args.out) if args.out else RUN / f"x87_generations_{args.condition}.json"
    done = {}
    if output_path.exists():
        prior = read_json(output_path)
        if prior.get("condition") != args.condition or prior.get("manifest", {}).get("sha256") != file_record(manifest_path)["sha256"]:
            raise SystemExit("FATAL: generation resume belongs to another run")
        for row in prior.get("rows", []):
            done[row["id"]] = row

    def checkpoint() -> None:
        atomic_json(
            output_path,
            {
                "schema_version": 1,
                "experiment": "x87",
                "condition": args.condition,
                "model_id": MODEL_ID,
                "model_source": model_source,
                "manifest": file_record(manifest_path),
                "intervention": public_condition_metadata(setup),
                "generation": {
                    "do_sample": False,
                    "temperature": 0,
                    "max_new_tokens": args.max_new_tokens,
                    "enable_thinking": False,
                },
                "rows": [done[key] for key in sorted(done)],
            },
        )

    pending = [row for row in rows if row["id"] not in done]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        chats = [format_user_chat(tokenizer, row["prompt"]) for row in batch]
        for chat in chats:
            assert_qwen_boundary(tokenizer, chat)
        encoded = tokenizer(
            chats, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        torch.manual_seed(stable_seed(args.condition, start))
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        responses = decode_new(tokenizer, generated, encoded["input_ids"].shape[1])
        if any(not response for response in responses):
            raise RuntimeError(f"empty response in batch starting {start}")
        for source, chat, response in zip(batch, chats, responses):
            if sha256_text(source["prompt"]) != source["prompt_sha256"]:
                raise RuntimeError(f"manifest prompt hash changed for {source['id']}")
            done[source["id"]] = {
                **source,
                "condition": args.condition,
                "rendered_chat_sha256": sha256_text(chat),
                "response": response,
                "response_sha256": sha256_text(response),
            }
        checkpoint()
        print(f"{args.condition}: {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    checkpoint()
    observed = {row["benchmark"] for row in done.values()}
    if not observed.issubset(BENCHMARKS):
        raise RuntimeError(f"unexpected benchmark tags: {observed}")


if __name__ == "__main__":
    main()

