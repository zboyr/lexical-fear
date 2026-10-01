"""x85 Stage 0: freeze prompts, word controls, folds, cells, and behavior graph."""

from __future__ import annotations

import argparse
import platform
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from x85_common import (
    BASE_SEED,
    BEHAVIOR_PROMPT_DEGREE,
    CONDITIONS,
    MODEL_ID,
    N_FEATURE_SHARDS,
    N_PROMPTS,
    N_RESPONSES,
    N_WORDS,
    POSITIONS,
    PRIMARY_STATES,
    RUN,
    WORD_PAIRS,
    append_word,
    assign_prompt_folds,
    assign_word_folds,
    atomic_json,
    behavior_edges,
    cohort_path,
    file_record,
    format_user_chat,
    locate_positions,
    prompt_stratum,
    read_json,
    select_prompts,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x83-cohort", default="data/runs/x83/x83_cohort.json")
    parser.add_argument("--model-snapshot", help="default: snapshot recorded by x83")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--out", default=str(cohort_path()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = RUN.parents[1] / "runs" / "x83" / "x83_cohort.json"
    if args.x83_cohort != "data/runs/x83/x83_cohort.json":
        from pathlib import Path

        source_path = Path(args.x83_cohort)
    source = read_json(source_path)
    if source.get("experiment") != "x83" or source.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: invalid x83 source cohort")
    model_source = args.model_snapshot or source["model_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

    x83_words = {word for words in source["lexicon"].values() for word in words}
    harms = [harm for _, harm, _ in WORD_PAIRS]
    neutrals = [neutral for _, _, neutral in WORD_PAIRS]
    if len(WORD_PAIRS) != N_WORDS or len(set(harms)) != N_WORDS or len(set(neutrals)) != N_WORDS:
        raise SystemExit("FATAL: x85 word/control list is not 24 unique pairs")
    if not set(harms).issubset(x83_words):
        raise SystemExit("FATAL: x85 harm words are not all in the frozen x83 lexicon")

    word_rows = []
    word_folds = assign_word_folds(args.seed + 2)
    for word_index, (category, harm, neutral) in enumerate(WORD_PAIRS):
        harm_ids = tokenizer.encode(" " + harm, add_special_tokens=False)
        neutral_ids = tokenizer.encode(" " + neutral, add_special_tokens=False)
        if len(harm_ids) != len(neutral_ids):
            raise SystemExit(
                f"FATAL: tokenizer-length mismatch {harm} ({len(harm_ids)}) vs "
                f"{neutral} ({len(neutral_ids)})"
            )
        word_rows.append(
            {
                "word_index": word_index,
                "category": category,
                "harm": harm,
                "neutral": neutral,
                "leading_space_harm_ids": [int(value) for value in harm_ids],
                "leading_space_neutral_ids": [int(value) for value in neutral_ids],
                "n_subtokens": len(harm_ids),
                "word_fold": word_folds[harm],
            }
        )

    selected = select_prompts(source["units"], args.seed)
    prompt_folds = assign_prompt_folds(selected, args.seed + 1)
    edges = behavior_edges((int(unit["id"]) for unit in selected), args.seed + 3)
    prompts = []
    for prompt_index, unit in enumerate(selected):
        prompt_id = int(unit["id"])
        prompts.append(
            {
                "prompt_index": prompt_index,
                "id": prompt_id,
                "prompt": unit["prompts"]["O"],
                "prompt_sha256": unit["prompt_sha256"]["O"],
                "historical_refusal_rate": float(unit["historical_refusal_rate"]),
                "stratum": prompt_stratum(float(unit["historical_refusal_rate"])),
                "prompt_fold": prompt_folds[prompt_id],
            }
        )

    sentinel = format_user_chat(tokenizer, "alpha bomb")
    sentinel_ids, sentinel_positions = locate_positions(
        tokenizer, sentinel, "alpha bomb", (6, 10)
    )
    tbg_minus1_id = int(tokenizer.convert_tokens_to_ids("</think>"))
    if sentinel_ids[-2] != tbg_minus1_id or tokenizer.decode([sentinel_ids[-1]]) != "\n\n":
        raise SystemExit(
            "FATAL: x85 requires TBG-1=</think> and TBG=double-newline under the frozen template"
        )
    template_audit = {
        "sentinel_rendered_sha256": sha256_text(sentinel),
        "tail_ids": sentinel_ids[-8:],
        "tail_tokens": tokenizer.convert_ids_to_tokens(sentinel_ids[-8:]),
        "tail_decoded": [tokenizer.decode([value]) for value in sentinel_ids[-8:]],
        "positions": sentinel_positions,
    }

    cells = []
    behavior_id = 0
    for prompt_row in prompts:
        prompt_id = prompt_row["id"]
        base = prompt_row["prompt"]
        rows = [("O", None, None, base, None, True)]
        for word_row in word_rows:
            harm_text, harm_span = append_word(base, word_row["harm"])
            neutral_text, neutral_span = append_word(base, word_row["neutral"])
            in_behavior = (prompt_id, word_row["harm"]) in edges
            rows.extend(
                [
                    ("H", word_row, word_row["harm"], harm_text, harm_span, in_behavior),
                    ("N", word_row, word_row["neutral"], neutral_text, neutral_span, in_behavior),
                ]
            )
        for condition, word_row, inserted_word, text, span, in_behavior in rows:
            if condition == "O" and any(cell["condition"] == "O" and cell["prompt_id"] == prompt_id for cell in cells):
                continue
            rendered = format_user_chat(tokenizer, text)
            input_ids, position_indices = locate_positions(tokenizer, rendered, text, span)
            cell_id = f"O:p{prompt_id}" if condition == "O" else f"{condition}:p{prompt_id}:w{word_row['word_index']:02d}"
            row = {
                "cell_index": len(cells),
                "cell_id": cell_id,
                "condition": condition,
                "prompt_id": prompt_id,
                "prompt_index": prompt_row["prompt_index"],
                "prompt_fold": prompt_row["prompt_fold"],
                "word_index": None if word_row is None else word_row["word_index"],
                "word_fold": None if word_row is None else word_row["word_fold"],
                "category": None if word_row is None else word_row["category"],
                "harm_word": None if word_row is None else word_row["harm"],
                "neutral_word": None if word_row is None else word_row["neutral"],
                "inserted_word": inserted_word,
                "inserted_span": None if span is None else list(span),
                "prompt": text,
                "prompt_sha256": sha256_text(text),
                "rendered_chat_sha256": sha256_text(rendered),
                "n_input_tokens": len(input_ids),
                "position_indices": position_indices,
                "behavior": bool(in_behavior),
                "behavior_id": behavior_id if in_behavior else None,
            }
            cells.append(row)
            if in_behavior:
                behavior_id += 1

    expected_cells = N_PROMPTS * (1 + 2 * N_WORDS)
    if len(cells) != expected_cells:
        raise SystemExit(f"FATAL: built {len(cells)} cells, expected {expected_cells}")
    counts_by_condition = Counter(cell["condition"] for cell in cells)
    behavior_counts = Counter(cell["condition"] for cell in cells if cell["behavior"])
    if counts_by_condition != Counter({"O": 96, "H": 2304, "N": 2304}):
        raise SystemExit(f"FATAL: dense condition counts are wrong: {counts_by_condition}")
    if behavior_counts != Counter({"O": 96, "H": 576, "N": 576}):
        raise SystemExit(f"FATAL: behavior condition counts are wrong: {behavior_counts}")

    cohort_sha = sha256_text(
        "\n".join(
            f"{cell['cell_index']}\t{cell['cell_id']}\t{cell['prompt_sha256']}\t{int(cell['behavior'])}"
            for cell in cells
        )
    )
    output = {
        "experiment": "x85",
        "stage": 0,
        "status": "frozen",
        "model_id": MODEL_ID,
        "model_snapshot": model_source,
        "design": {
            "position": "append-only E position: prompt + space + one word",
            "latent_grid": "dense 96 prompts x 24 paired H/N words plus one O per prompt",
            "behavior_grid": (
                "regular sparse bipartite graph: prompt degree 6, harm-word degree 24; "
                "paired neutral conditions and all O prompts"
            ),
            "primary_states": list(PRIMARY_STATES),
            "positions": list(POSITIONS),
            "feature_shards": N_FEATURE_SHARDS,
            "generation": "x83/x63 contract: 10 samples, T=1.0, top-p=0.9, top-k=50, 2048 tokens plus one 4096 retry",
            "judge": "x83/x64 refusal2 contract",
            "selection_reads": "x83 cohort membership, prompt text, and historical O rate only; no x83 detection/generation/judgment outcome",
        },
        "template_audit": template_audit,
        "inputs": {"x83_cohort": file_record(source_path)},
        "seeds": {
            "base": args.seed,
            "prompt_folds": args.seed + 1,
            "word_folds": args.seed + 2,
            "behavior_graph": args.seed + 3,
        },
        "counts": {
            "prompts": len(prompts),
            "words": len(word_rows),
            "dense_cells": len(cells),
            "behavior_cells": behavior_id,
            "by_condition": dict(counts_by_condition),
            "behavior_by_condition": dict(behavior_counts),
            "behavior_prompt_degree": BEHAVIOR_PROMPT_DEGREE,
            "responses_per_behavior_cell": N_RESPONSES,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tokenizer_class": tokenizer.__class__.__name__,
        },
        "cohort_sha256": cohort_sha,
        "prompts": prompts,
        "words": word_rows,
        "cells": cells,
    }
    from pathlib import Path

    out_path = Path(args.out)
    atomic_json(out_path, output)
    print(
        f"x85 cohort: {len(prompts)} prompts x {len(word_rows)} word pairs, "
        f"{len(cells)} dense cells / {behavior_id} behavior cells -> {out_path}"
    )


if __name__ == "__main__":
    main()
