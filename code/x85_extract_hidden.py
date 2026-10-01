"""x85 Stage 1: sharded four-position residual-stream extraction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x85_common import (
    FEATURES,
    MODEL_ID,
    N_FEATURE_SHARDS,
    POSITIONS,
    cohort_path,
    format_user_chat,
    load_causal_lm,
    locate_positions,
    read_json,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=str(cohort_path()))
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=N_FEATURE_SHARDS)
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def default_out(shard: int, num_shards: int) -> Path:
    return FEATURES / f"x85_hidden_shard_{shard:03d}-of-{num_shards:03d}.pt"


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("FATAL: shard must satisfy 0 <= shard < num_shards")
    manifest = read_json(Path(args.cohort))
    if manifest.get("experiment") != "x85" or manifest.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: invalid x85 cohort")
    cells = [
        cell for cell in manifest["cells"] if int(cell["cell_index"]) % args.num_shards == args.shard
    ]
    if args.limit:
        cells = cells[: args.limit]
    if not cells:
        raise SystemExit("FATAL: empty extraction shard")
    out_path = Path(args.out) if args.out else default_out(args.shard, args.num_shards)
    if out_path.exists():
        prior = torch.load(out_path, map_location="cpu", weights_only=False)
        expected = [cell["cell_id"] for cell in cells]
        if (
            prior.get("experiment") == "x85"
            and prior.get("cohort_sha256") == manifest["cohort_sha256"]
            and prior.get("cell_ids") == expected
        ):
            print(f"x85 hidden shard already complete: {out_path}")
            return
        raise SystemExit("FATAL: existing shard output does not match requested cells/cohort")

    model_source = manifest["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    model = load_causal_lm(model_source)
    position_names = list(POSITIONS)
    state_rows = []
    cell_records = []
    n_states = hidden_dim = None

    for index, cell in enumerate(cells, start=1):
        text = cell["prompt"]
        if sha256_text(text) != cell["prompt_sha256"]:
            raise SystemExit(f"FATAL: prompt hash mismatch for {cell['cell_id']}")
        rendered = format_user_chat(tokenizer, text)
        if sha256_text(rendered) != cell["rendered_chat_sha256"]:
            raise SystemExit(f"FATAL: rendered chat mismatch for {cell['cell_id']}")
        span = None if cell["inserted_span"] is None else tuple(cell["inserted_span"])
        input_ids, positions = locate_positions(tokenizer, rendered, text, span)
        if positions != cell["position_indices"] or len(input_ids) != cell["n_input_tokens"]:
            raise SystemExit(f"FATAL: token-position audit mismatch for {cell['cell_id']}")
        inputs = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        with torch.inference_mode():
            outputs = model(input_ids=inputs, output_hidden_states=True, return_dict=True)
        if n_states is None:
            n_states = len(outputs.hidden_states)
            hidden_dim = int(outputs.hidden_states[0].shape[-1])
        if len(outputs.hidden_states) != n_states:
            raise SystemExit("FATAL: hidden-state count changed within shard")
        row = torch.empty(
            (len(position_names), n_states, hidden_dim), dtype=torch.float16, device="cpu"
        )
        for state_index, hidden in enumerate(outputs.hidden_states):
            for position_index, name in enumerate(position_names):
                row[position_index, state_index] = hidden[0, positions[name]].float().cpu().to(torch.float16)
        state_rows.append(row)
        cell_records.append(
            {
                "cell_index": int(cell["cell_index"]),
                "cell_id": cell["cell_id"],
                "prompt_sha256": cell["prompt_sha256"],
                "n_input_tokens": len(input_ids),
                "position_indices": positions,
            }
        )
        print(f"x85 hidden shard {args.shard}: {index}/{len(cells)}", flush=True)

    artifact = {
        "experiment": "x85",
        "stage": 1,
        "model_id": MODEL_ID,
        "model_source": model_source,
        "cohort_sha256": manifest["cohort_sha256"],
        "shard": args.shard,
        "num_shards": args.num_shards,
        "position_names": position_names,
        "num_states": int(n_states),
        "hidden_dim": int(hidden_dim),
        "cell_ids": [cell["cell_id"] for cell in cells],
        "cells": cell_records,
        "states": torch.stack(state_rows),  # [cell, position, state, hidden]
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(artifact, temp_path)
    os.replace(temp_path, out_path)
    print(f"x85 hidden shard {args.shard}: {len(cells)} cells -> {out_path}")


if __name__ == "__main__":
    main()
