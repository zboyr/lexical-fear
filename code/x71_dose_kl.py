"""Measure benign-prompt KL as a function of generation-time dose.

Forward passes only: no training and no generation. Implements the frozen
measurement in DOSE_EXTENSION.md with the same KL definition as
``train_edit.validation_metrics``: right padding, probe chat date, last-8
prompt-token positions, KL(base || edited) with base at scale 0, per-batch
means averaged over batches, stored-order ``benign_validation`` rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

from edit_common import atomic_json_dump, file_record, load_causal_lm, load_config  # noqa: E402
from edit_rank1 import install_rank_one_edit  # noqa: E402
from x71_train_edit import select_last_positions, tokenize_prompts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt-splits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--doses", nargs="+", type=float, required=True)
    parser.add_argument("--match-v-norm-from")
    parser.add_argument("--config")
    parser.add_argument(
        "--model-source",
        help="exact local snapshot to load; artifact model_id remains the canonical config model_id",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--subset-batches", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(dose == 0 for dose in args.doses):
        raise SystemExit("dose 0 is the identity edit; measure non-zero doses only")
    config = load_config(args.config)
    device = torch.device("cuda")

    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    right = adapter["right"].float()
    norm_reference_record = None
    if args.match_v_norm_from:
        reference = torch.load(args.match_v_norm_from, map_location="cpu", weights_only=False)
        target_norm = reference["right"].float().norm()
        right = right * (target_norm / right.norm().clamp_min(1e-12))
        norm_reference_record = file_record(args.match_v_norm_from)

    prompt_splits = json.loads(Path(args.prompt_splits).read_text())
    rows = prompt_splits["benign_validation"]

    model_source = args.model_source or config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = load_causal_lm(model_source)
    model.config.use_cache = False
    model.eval()
    wrapper = install_rank_one_edit(
        model,
        block_index=int(adapter["block_index"]),
        q_unit=adapter["q_unit"],
        scale=0.0,
    )
    wrapper.right.data.copy_(right.to(wrapper.right.device))
    wrapper.right.requires_grad_(False)

    last_n = int(config["edit"]["retain_last_n"])
    date_string = config["probe_chat_date"]
    batch_size = int(args.batch_size)
    batch_means: dict[float, list[float]] = {dose: [] for dose in args.doses}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            prompts = [str(row["prompt"]) for row in rows[start : start + batch_size]]
            tokens = tokenize_prompts(tokenizer, prompts, date_string, device)
            wrapper.scale = 0.0
            base = model(**tokens, return_dict=True, use_cache=False).logits
            base_selected = select_last_positions(
                base, tokens["attention_mask"], last_n
            ).float()
            base_logp = F.log_softmax(base_selected, dim=-1)
            base_p = base_logp.exp()
            for dose in args.doses:
                wrapper.scale = float(dose)
                edited = model(**tokens, return_dict=True, use_cache=False).logits
                edited_selected = select_last_positions(
                    edited, tokens["attention_mask"], last_n
                ).float()
                edited_logp = F.log_softmax(edited_selected, dim=-1)
                kl = float((base_p * (base_logp - edited_logp)).sum(-1).mean())
                if not math.isfinite(kl):
                    raise RuntimeError(f"non-finite KL at dose {dose}, batch start {start}")
                batch_means[dose].append(kl)
            print(f"{min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    wrapper.scale = 0.0

    budget = float(config["edit"]["kl_budget"])
    measurements = []
    for dose in args.doses:
        means = batch_means[dose]
        subset = means[: int(args.subset_batches)]
        measurements.append(
            {
                "dose": float(dose),
                "kl_full_mean": float(sum(means) / len(means)),
                "kl_subset_mean": float(sum(subset) / len(subset)),
                "n_batches": len(means),
                "within_budget_full": bool(sum(means) / len(means) <= budget),
            }
        )
    result = {
        "schema_version": 1,
        "label": args.label,
        "kl_definition": "train_edit.validation_metrics benign KL, batch means averaged",
        "n_benign_validation_rows": len(rows),
        "batch_size": batch_size,
        "subset_batches": int(args.subset_batches),
        "retain_last_n": last_n,
        "kl_budget": budget,
        "norm_matched": bool(args.match_v_norm_from),
        "measurements": measurements,
        "adapter": file_record(args.adapter),
        "norm_reference": norm_reference_record,
        "prompt_splits": file_record(args.prompt_splits),
    }
    atomic_json_dump(result, args.out)
    print(json.dumps({"label": args.label, "measurements": measurements}, indent=2))


if __name__ == "__main__":
    main()
