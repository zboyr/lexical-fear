"""Generate deterministic responses for the frozen x87 rows under one x88 model/condition."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from x88_common import (
    CONDITIONS,
    MAX_NEW_TOKENS,
    MODELS,
    RUN,
    assert_boundary,
    atomic_json,
    default_paths,
    file_record,
    format_user_chat,
    load_model,
    load_tokenizer,
    public_condition_metadata,
    read_json,
    setup_condition,
    sha256_text,
    stable_seed,
    tag_dir,
)

BENCHMARKS = ("orbench_hard", "orbench_toxic", "strongreject")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--manifest", default=str(RUN / "x88_manifest.json"))
    parser.add_argument("--directions")
    parser.add_argument("--adapter")
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
    if manifest.get("status") != "frozen" or manifest.get("experiment") != "x88":
        raise SystemExit("FATAL: invalid x88 manifest")
    entry = manifest["models"][args.model]
    paths = default_paths(args.model)
    tokenizer = load_tokenizer(entry["snapshot"], args.model)
    model = load_model(entry["snapshot"], args.model)
    setup = setup_condition(
        model,
        args.model,
        args.condition,
        directions=args.directions or paths["directions"],
        adapter_path=args.adapter or paths["adapter"],
    ) if args.condition != "base" else setup_condition(model, args.model, "base")
    model.eval()
    rows = list(manifest["rows"])
    if args.limit:
        rows = rows[: args.limit]
    output_path = (
        Path(args.out)
        if args.out
        else tag_dir(args.model) / f"x88_generations_{args.condition}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if output_path.exists():
        prior = read_json(output_path)
        if (
            prior.get("condition") != args.condition
            or prior.get("model_tag") != args.model
            or prior.get("manifest", {}).get("sha256") != file_record(manifest_path)["sha256"]
        ):
            raise SystemExit("FATAL: generation resume belongs to another run")
        for row in prior.get("rows", []):
            done[row["id"]] = row

    def checkpoint() -> None:
        atomic_json(
            output_path,
            {
                "schema_version": 1,
                "experiment": "x88",
                "model_tag": args.model,
                "condition": args.condition,
                "model_id": MODELS[args.model]["model_id"],
                "model_source": entry["snapshot"],
                "manifest": file_record(manifest_path),
                "intervention": public_condition_metadata(setup),
                "generation": {
                    "do_sample": False,
                    "temperature": 0,
                    "max_new_tokens": args.max_new_tokens,
                },
                "rows": [done[key] for key in sorted(done)],
            },
        )

    pending = [row for row in rows if row["id"] not in done]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        chats = [format_user_chat(tokenizer, row["prompt"]) for row in batch]
        for chat in chats:
            assert_boundary(tokenizer, chat, entry["boundary_ids"], args.model)
        encoded = tokenizer(
            chats, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        torch.manual_seed(stable_seed(args.model, args.condition, start))
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
        print(
            f"{args.model}/{args.condition}: {min(start + len(batch), len(pending))}/{len(pending)}",
            flush=True,
        )
    checkpoint()
    observed = {row["benchmark"] for row in done.values()}
    if not observed.issubset(set(BENCHMARKS) | {"ifeval"}):
        raise RuntimeError(f"unexpected benchmark tags: {observed}")


if __name__ == "__main__":
    main()
