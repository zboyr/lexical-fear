"""x81 Stage 2: score the full one-token fill distribution with Qwen."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x81_common import (
    BLANK,
    FILL_INSTRUCTION,
    MASK,
    MODEL_ID,
    RUN,
    TOP_K_FILLS,
    atomic_json,
    candidate_surface,
    file_record,
    format_user_chat,
    is_lexical_candidate,
    load_causal_lm,
    read_json,
    replace_span,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--audit", default=str(RUN / "x81_audit.json"))
    parser.add_argument("--lexicon", default=str(RUN / "x81_lexicon.json"))
    parser.add_argument("--out", default=str(RUN / "x81_fills.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def distribution_summary(logits: torch.Tensor, tokenizer, target_word: str) -> dict:
    values = logits.float()
    log_z = torch.logsumexp(values, dim=-1)
    log_probabilities = values - log_z
    probabilities = log_probabilities.exp()
    entropy = float(-(probabilities * log_probabilities).sum().item())
    top_probabilities, top_ids = torch.topk(probabilities, TOP_K_FILLS)
    candidates = []
    for rank, (probability, token_id) in enumerate(
        zip(top_probabilities.tolist(), top_ids.tolist()), start=1
    ):
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        surface = candidate_surface(decoded)
        candidates.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "decoded": decoded,
                "surface": surface,
                "probability": float(probability),
                "lexical_candidate": is_lexical_candidate(surface),
                "is_original_surface": surface.casefold() == target_word.casefold(),
                "is_blank": surface == BLANK,
            }
        )
    original = next((row for row in candidates if row["is_original_surface"]), None)
    blank = next((row for row in candidates if row["is_blank"]), None)
    return {
        "entropy_nats": entropy,
        "topk_mass": float(top_probabilities.sum().item()),
        "original_rank": None if original is None else original["rank"],
        "original_probability": None if original is None else original["probability"],
        "blank_rank": None if blank is None else blank["rank"],
        "blank_probability": None if blank is None else blank["probability"],
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("FATAL: --batch-size must be positive")
    audit_path = Path(args.audit)
    lexicon_path = Path(args.lexicon)
    if not audit_path.exists() or not lexicon_path.exists():
        raise SystemExit("FATAL: run x81 audit and lexicon stages first")
    audit = read_json(audit_path)
    lexicon = read_json(lexicon_path)
    if audit.get("status") != "pass" or lexicon.get("uses_evaluation_refusal_labels") is not False:
        raise SystemExit("FATAL: invalid upstream x81 manifests")
    if audit["model_id"] != args.model or lexicon["model_id"] != args.model:
        raise SystemExit("FATAL: model ID differs from upstream manifests")
    if args.model != MODEL_ID:
        raise SystemExit(f"FATAL: x81 is frozen to {MODEL_ID}")
    model_source = audit["model_snapshot"]["snapshot"]
    if (
        lexicon.get("model_source") != model_source
        or lexicon.get("model_manifest_sha256") != audit["model_snapshot"]["manifest_sha256"]
    ):
        raise SystemExit("FATAL: audited model snapshot differs from lexicon manifest")

    targets = list(lexicon["targets"])
    if args.limit:
        targets = targets[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_lm(model_source)

    rows = []
    for offset in range(0, len(targets), args.batch_size):
        batch = targets[offset : offset + args.batch_size]
        rendered = []
        masked_prompts = []
        for row in batch:
            masked = replace_span(row["prompt"], int(row["start"]), int(row["end"]), MASK)
            masked_prompts.append(masked)
            rendered.append(format_user_chat(tokenizer, FILL_INSTRUCTION.format(masked_prompt=masked)))
        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :]
        for source, masked, chat, one_logits in zip(batch, masked_prompts, rendered, logits):
            summary = distribution_summary(one_logits, tokenizer, source["target_word"])
            rows.append(
                {
                    "id": int(source["id"]),
                    "prompt": source["prompt"],
                    "prompt_sha256": sha256_text(source["prompt"]),
                    "target_word": source["target_word"],
                    "target_word_lower": source["target_word_lower"],
                    "start": int(source["start"]),
                    "end": int(source["end"]),
                    "masked_prompt": masked,
                    "rendered_chat_sha256": sha256_text(chat),
                    **summary,
                }
            )
        print(f"x81 fills: {min(offset + args.batch_size, len(targets))}/{len(targets)}", flush=True)

    output = {
        "experiment": "x81",
        "stage": 2,
        "model_id": args.model,
        "model_source": model_source,
        "model_manifest_sha256": audit["model_snapshot"]["manifest_sha256"],
        "uses_evaluation_refusal_labels": False,
        "config": {
            "top_k": TOP_K_FILLS,
            "sampling": False,
            "thinking": False,
            "probabilities": "raw full-vocabulary softmax",
            "entropy_units": "nats",
        },
        "inputs": {"audit": file_record(audit_path), "lexicon": file_record(lexicon_path)},
        "template_sha256": sha256_text(FILL_INSTRUCTION),
        "rows": rows,
    }
    atomic_json(args.out, output)
    finite = all(math.isfinite(row["entropy_nats"]) for row in rows)
    if not finite:
        raise SystemExit("FATAL: non-finite fill entropy written")
    print(f"x81 fill distributions: {len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
