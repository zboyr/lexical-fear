"""Freeze the x88 manifest: per-model snapshots, boundary audits, x87 rows."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from x88_common import (
    MODELS,
    RUN,
    TAGS,
    atomic_json,
    file_record,
    format_user_chat,
    hook_block_for,
    load_tokenizer,
    read_json,
    sha256_text,
)

AUDIT_PROMPTS = (
    "x88 boundary audit",
    "A second, longer boundary-audit prompt to prove the trailing ids are template constants.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x87-manifest", default=str(RUN.parent / "x87" / "x87_manifest.json"))
    parser.add_argument("--out", default=str(RUN / "x88_manifest.json"))
    return parser.parse_args()


def resolve_snapshot(model_id: str) -> dict:
    from huggingface_hub import snapshot_download

    path = snapshot_download(model_id)
    return {"model_id": model_id, "snapshot": path, "revision": Path(path).name}


def boundary_audit(tag: str, snapshot: str) -> dict:
    from transformers import AutoConfig

    tokenizer = load_tokenizer(snapshot, tag)
    trailing = []
    for prompt in AUDIT_PROMPTS:
        rendered = format_user_chat(tokenizer, prompt)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            raise SystemExit(f"FATAL: {tag} renders fewer than 2 prefill tokens")
        trailing.append(list(ids[-2:]))
    if trailing[0] != trailing[1]:
        raise SystemExit(
            f"FATAL: {tag} trailing ids are not template constants: {trailing}"
        )
    config = AutoConfig.from_pretrained(
        snapshot, trust_remote_code=MODELS[tag]["trust_remote_code"]
    )
    n_blocks = int(config.get_text_config().num_hidden_layers)
    return {
        "boundary_ids": trailing[0],
        "boundary_tokens": [tokenizer.decode([i]) for i in trailing[0]],
        "n_blocks": n_blocks,
        "hook_block": hook_block_for(n_blocks),
    }


def main() -> None:
    args = parse_args()
    x87_path = Path(args.x87_manifest)
    x87 = read_json(x87_path)
    if x87.get("status") != "frozen":
        raise SystemExit("FATAL: x87 manifest is not frozen")
    rows = x87["rows"]
    rows_sha = sha256_text(
        "\n".join(f"{row['id']}\t{row['prompt_sha256']}" for row in rows)
    )
    if rows_sha != x87.get("rows_sha256"):
        raise SystemExit("FATAL: x87 rows fail their recorded sha256")
    models = {}
    for tag in TAGS:
        snapshot = resolve_snapshot(MODELS[tag]["model_id"])
        audit = boundary_audit(tag, snapshot["snapshot"])
        models[tag] = {**snapshot, **audit}
        print(tag, audit["boundary_tokens"], f"blocks={audit['n_blocks']}",
              f"hook={audit['hook_block']}", flush=True)
    manifest = {
        "schema_version": 1,
        "experiment": "x88",
        "status": "frozen",
        "role": "exploratory_cross_model_extension",
        "conditions": ["base", "x86_m150", "x75_m6"],
        "models": models,
        "imported": {"qwen": "rows come from x87 result files, never rerun here"},
        "judge_snapshot": x87["model_snapshot"],
        "x87_manifest": file_record(x87_path),
        "benchmarks": {
            "ifeval": {"lm_eval_version": importlib.metadata.version("lm-eval")},
            "gsm8k": {"lm_eval_version": importlib.metadata.version("lm-eval")},
        },
        "rows_sha256": rows_sha,
        "rows": rows,
    }
    atomic_json(args.out, manifest)
    print(f"frozen: {args.out} ({len(rows)} rows, {len(models)} models)")


if __name__ == "__main__":
    main()
