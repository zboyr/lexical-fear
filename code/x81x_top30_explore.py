"""Exploration (NOT frozen x81 pipeline): relaxed admission with top-30 fills.

Design under test, per user request:
  - DROP the top10_mass gate.
  - DROP the "blank present in top-10" gate.
  - p_same is computed over the top-30 fill candidates (each sent to the same
    Qwen for a SAME/CHANGED semantic judgment); p_same = sum of RAW
    full-vocabulary probabilities of top-30 candidates judged SAME.
  - Rule 4 kept (structurally required to build a C arm): a top-30 candidate
    that is SAME AND discovery-neutral must exist.

Admission(tau) = (p_same_top30 >= tau) AND (a SAME discovery-neutral C exists).
Outputs a sweep of tau vs admitted count / distinct target words / max share.

This does not read any refusal outcome. It reuses the frozen x81 lexicon,
templates, span editing, and semantic parser.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from x81_common import (
    BLANK,
    FILL_INSTRUCTION,
    MASK,
    SEMANTIC_INSTRUCTION,
    RUN,
    atomic_json,
    candidate_surface,
    format_user_chat,
    is_lexical_candidate,
    load_causal_lm,
    match_case,
    read_json,
    replace_span,
    simple_lemma,
)

TOP_K = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lexicon", default=str(RUN / "x81_lexicon.json"))
    p.add_argument("--out", default=str(RUN / "x81x_top30_explore.json"))
    p.add_argument("--fill-batch", type=int, default=8)
    p.add_argument("--sem-batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def exact_decision(raw: str) -> str:
    v = raw.strip().upper()
    return v if v in {"SAME", "CHANGED"} else "CHANGED"


def top30_fills(model, tokenizer, targets, batch_size):
    """One forward pass per masked prompt; keep top-30 raw-prob candidates."""
    rows = []
    for off in range(0, len(targets), batch_size):
        batch = targets[off : off + batch_size]
        rendered = []
        for t in batch:
            masked = replace_span(t["prompt"], int(t["start"]), int(t["end"]), MASK)
            rendered.append(format_user_chat(tokenizer, FILL_INSTRUCTION.format(masked_prompt=masked)))
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        with torch.inference_mode():
            logits = model(**inputs).logits[:, -1, :].float()
        logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        probs = logp.exp()
        topp, topi = torch.topk(probs, TOP_K, dim=-1)
        for t, pr, ti in zip(batch, topp.tolist(), topi.tolist()):
            cands = []
            for rank, (prob, tid) in enumerate(zip(pr, ti), start=1):
                decoded = tokenizer.decode([tid], skip_special_tokens=False)
                surface = candidate_surface(decoded)
                cands.append(
                    {
                        "rank": rank,
                        "token_id": int(tid),
                        "surface": surface,
                        "probability": float(prob),
                        "lexical_candidate": is_lexical_candidate(surface),
                        "is_blank": surface == BLANK,
                    }
                )
            rows.append({**t, "candidates": cands})
        print(f"fills(top30): {min(off + batch_size, len(targets))}/{len(targets)}", flush=True)
    return rows


def judge_semantics(model, tokenizer, rows, batch_size):
    work = []  # (row_idx, cand_idx, chat)
    for ri, row in enumerate(rows):
        for ci, c in enumerate(row["candidates"]):
            if c["is_blank"]:
                replacement = ""
            else:
                replacement = match_case(c["surface"], row["target_word"])
            modified = replace_span(row["prompt"], int(row["start"]), int(row["end"]), replacement)
            instr = SEMANTIC_INSTRUCTION.format(original=row["prompt"], modified=modified)
            work.append((ri, ci, format_user_chat(tokenizer, instr)))
    for off in range(0, len(work), batch_size):
        batch = work[off : off + batch_size]
        rendered = [w[2] for w in batch]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        with torch.inference_mode():
            gen = model.generate(
                **inputs, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        ans = tokenizer.batch_decode(gen[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        for (ri, ci, _), raw in zip(batch, ans):
            c = rows[ri]["candidates"][ci]
            eligible = bool(c["lexical_candidate"])
            parsed = exact_decision(raw)
            c["decision"] = parsed if eligible else "CHANGED"
        print(f"semantics(top30): {min(off + batch_size, len(work))}/{len(work)}", flush=True)
    return rows


def is_neutral(c, target_word, sensitive, features):
    if c["decision"] != "SAME" or c["is_blank"] or not c["lexical_candidate"]:
        return False
    w = c["surface"].casefold()
    tgt = target_word.casefold()
    if w == tgt or simple_lemma(w) == simple_lemma(tgt) or w in sensitive:
        return False
    f = features.get(w)
    if f is None or int(f["document_frequency"]) < 10:
        return False
    coefs = [float(f["full_coef"]), *map(float, f["fold_coefs"])]
    return all(v == 0.0 for v in coefs)


def main() -> None:
    args = parse_args()
    lex = read_json(args.lexicon)
    features = {r["word"]: r for r in lex["features"]}
    sensitive = {r["word"] for r in lex["sensitive_words"]}
    targets = list(lex["targets"])
    if args.limit:
        targets = targets[: args.limit]

    src = lex["model_source"]
    tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_lm(src)

    rows = top30_fills(model, tokenizer, targets, args.fill_batch)
    rows = judge_semantics(model, tokenizer, rows, args.sem_batch)

    per_prompt = []
    for row in rows:
        p_same = sum(
            float(c["probability"]) for c in row["candidates"] if c["decision"] == "SAME"
        )
        neutral = [c for c in row["candidates"] if is_neutral(c, row["target_word"], sensitive, features)]
        per_prompt.append(
            {
                "id": int(row["id"]),
                "target_word_lower": row["target_word_lower"],
                "p_same_top30": p_same,
                "has_neutral_c": bool(neutral),
            }
        )

    taus = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.40, 0.30]
    sweep = []
    for tau in taus:
        adm = [r for r in per_prompt if r["p_same_top30"] >= tau and r["has_neutral_c"]]
        words = Counter(r["target_word_lower"] for r in adm)
        share = max(words.values()) / len(adm) if adm else 0.0
        sweep.append(
            {
                "tau": tau,
                "admitted": len(adm),
                "target_words": len(words),
                "max_target_share": round(share, 3),
                "support_gate": len(adm) >= 100 and len(words) >= 5 and share <= 0.5,
            }
        )

    out = {
        "experiment": "x81x_top30_explore",
        "design": "drop top10_mass and blank-in-top10; p_same over top-30; keep neutral-C (rule 4)",
        "top_k": TOP_K,
        "n_prompts": len(per_prompt),
        "uses_refusal_outcomes": False,
        "sweep": sweep,
        "per_prompt": per_prompt,
    }
    atomic_json(args.out, out)
    print("\ntau   admitted  words  max_share  support_gate")
    for s in sweep:
        print(
            f"{s['tau']:.2f}   {s['admitted']:5d}   {s['target_words']:4d}    "
            f"{s['max_target_share']:.3f}     {s['support_gate']}"
        )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
