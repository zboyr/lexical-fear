"""Per-model boundary-shift directions from the frozen x85 crossed cohort.

Re-renders the model-independent x85 cell texts with the target model's chat
template, extracts hidden states at the two trailing prefill positions at the
model's frozen relative-depth state, and rebuilds the four fold directions
exactly as x86 stage 0 did (mean H-minus-N over fold-excluded cells).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from x86_common import fold_direction
from x88_common import (
    DATA,
    MODELS,
    RUN,
    assert_boundary,
    atomic_json,
    file_record,
    format_user_chat,
    load_model,
    load_tokenizer,
    read_json,
    tag_dir,
)

POSITIONS = ("tbg_minus1", "tbg")
N_FOLDS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--manifest", default=str(RUN / "x88_manifest.json"))
    parser.add_argument("--cohort", default=str(DATA / "runs" / "x85" / "x85_cohort.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out")
    parser.add_argument("--audit-out")
    return parser.parse_args()


def paired_cells(cohort: dict) -> tuple[list[dict], list[dict], list[int], list[int]]:
    by_key: dict[tuple[int, int], dict[str, dict]] = {}
    for cell in cohort["cells"]:
        if cell["condition"] not in ("H", "N"):
            continue
        key = (int(cell["prompt_id"]), int(cell["word_index"]))
        by_key.setdefault(key, {})[cell["condition"]] = cell
    prompt_ids = sorted({k[0] for k in by_key})
    word_indices = sorted({k[1] for k in by_key})
    prompt_folds: dict[int, int] = {}
    word_folds: dict[int, int] = {}
    h_cells, n_cells = [], []
    for key in sorted(by_key):
        pair = by_key[key]
        if set(pair) != {"H", "N"}:
            raise SystemExit(f"FATAL: unpaired cell at {key}: {sorted(pair)}")
        for cell in pair.values():
            pf, wf = int(cell["prompt_fold"]), int(cell["word_fold"])
            if prompt_folds.setdefault(key[0], pf) != pf:
                raise SystemExit(f"FATAL: inconsistent prompt fold at {key}")
            if word_folds.setdefault(key[1], wf) != wf:
                raise SystemExit(f"FATAL: inconsistent word fold at {key}")
        h_cells.append(pair["H"])
        n_cells.append(pair["N"])
    if len(by_key) != len(prompt_ids) * len(word_indices):
        raise SystemExit(
            f"FATAL: H/N grid is not complete: {len(by_key)} pairs vs "
            f"{len(prompt_ids)}x{len(word_indices)}"
        )
    return (
        h_cells,
        n_cells,
        [prompt_folds[p] for p in prompt_ids],
        [word_folds[w] for w in word_indices],
    )


def extract(model, tokenizer, cells, state_index, boundary_ids, tag, batch_size):
    order = sorted(range(len(cells)), key=lambda i: len(cells[i]["prompt"]))
    out = [None] * len(cells)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        chats = [format_user_chat(tokenizer, cells[i]["prompt"]) for i in chunk]
        for chat in chats:
            assert_boundary(tokenizer, chat, boundary_ids, tag)
        encoded = tokenizer(
            chats, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.inference_mode():
            result = model(**encoded, output_hidden_states=True, use_cache=False)
        states = result.hidden_states[state_index][:, -2:, :].float().cpu().numpy()
        for row, i in enumerate(chunk):
            out[i] = states[row]
        if (start // batch_size) % 20 == 0:
            print(f"{tag}: {start + len(chunk)}/{len(order)}", flush=True)
    return np.stack(out)  # [n_pairs, 2, hidden]


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    if manifest.get("status") != "frozen":
        raise SystemExit("FATAL: invalid x88 manifest")
    entry = manifest["models"][args.model]
    cohort = read_json(args.cohort)
    if cohort.get("experiment") != "x85":
        raise SystemExit("FATAL: invalid x85 cohort")
    h_cells, n_cells, prompt_folds, word_folds = paired_cells(cohort)
    n_p, n_w = len(prompt_folds), len(word_folds)
    tokenizer = load_tokenizer(entry["snapshot"], args.model)
    model = load_model(entry["snapshot"], args.model)
    model.eval()
    state_index = int(entry["hook_block"]) + 1
    h_states = extract(
        model, tokenizer, h_cells, state_index, entry["boundary_ids"], args.model, args.batch_size
    )
    n_states = extract(
        model, tokenizer, n_cells, state_index, entry["boundary_ids"], args.model, args.batch_size
    )
    deltas = (h_states - n_states).reshape(n_p, n_w, 2, -1)
    arrays, norms, cosines = {}, {}, {}
    for pos_i, position in enumerate(POSITIONS):
        grid = deltas[:, :, pos_i, :]
        folds = [
            fold_direction(grid, prompt_folds, word_folds, fold) for fold in range(N_FOLDS)
        ]
        for fold, vec in enumerate(folds):
            arrays[f"true_f{fold}_{position}"] = vec.astype(np.float32)
        norms[position] = [float(np.linalg.norm(v)) for v in folds]
        units = [v / np.linalg.norm(v) for v in folds]
        cosines[position] = float(
            min(
                float(np.dot(units[a], units[b]))
                for a in range(N_FOLDS)
                for b in range(a + 1, N_FOLDS)
            )
        )
    out = Path(args.out) if args.out else tag_dir(args.model) / "x88_directions.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    audit = {
        "schema_version": 1,
        "experiment": "x88",
        "model_tag": args.model,
        "model_id": MODELS[args.model]["model_id"],
        "snapshot": entry["snapshot"],
        "state_index": state_index,
        "hook_block": int(entry["hook_block"]),
        "boundary_ids": entry["boundary_ids"],
        "n_pairs": int(n_p * n_w),
        "grid": [int(n_p), int(n_w)],
        "fold_norms": norms,
        "min_cross_fold_cosine": cosines,
        "cohort": file_record(args.cohort),
        "cohort_sha256": cohort.get("cohort_sha256"),
        "directions": file_record(out),
    }
    audit_out = (
        Path(args.audit_out)
        if args.audit_out
        else tag_dir(args.model) / "x88_directions_audit.json"
    )
    atomic_json(audit_out, audit)
    print({"tag": args.model, "norms": norms, "min_cosine": cosines})


if __name__ == "__main__":
    main()
