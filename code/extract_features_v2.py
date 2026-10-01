"""Stage 3 (v2, shared by the Llama and Qwen lines): extract TBG + content_mean
hidden states and TBG logits for all prompts, with validated labels.

Supersedes `confusable_datasets/t3_extract_features.py` (Llama, tbg only) and
`q3_extract_features.py` (Qwen, superseded). One script for both lines so they
cannot drift apart again.

Per prompt, in ONE forward pass:
  - tbg_states:          L layers x [n, d]  (last/TBG token, fp16)
  - content_mean_states: L layers x [n, d]  (mean over USER-CONTENT tokens only;
                          chat-template scaffolding excluded via offset mapping)
  - tbg_logits:          [n, vocab] fp16 memmap (next-token logits at TBG)

Two deliberate differences from the original Llama extractor -- both make the
Llama line match the Qwen line, so the caches are no longer bit-comparable to the
pre-2026-07-20 Llama caches:

  1. `add_special_tokens=False`. The Llama chat template already emits
     <|begin_of_text|>, and the old extractor let the tokenizer prepend a second
     one -- every cached prompt carried a DUPLICATED BOS (ids [128000, 128000, ...]).
  2. `date_string` is pinned (Llama's template interpolates today's date into the
     system prompt, which would otherwise make the cache depend on the wall clock).

Labels come from a validated clustered file (see rejudge_validated.py): every
harmful row must carry a well-formed `llm_clustering`. Unlike the old scripts,
this one does NOT silently drop rows whose judge output failed -- it refuses to
run, because a dropped row is a prompt missing from the dataset.

Positional contract (unchanged): harmful (toxic, real T) first, then benign
(hard, T=0). Both sides sorted by id.
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATE_STRING = "09 Jul 2026"  # pinned: matches the original cache's build date


def content_token_idx(tok, prompt):
    """Return (templated_input_ids, list of content-token positions)."""
    kw = {}
    try:
        full = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False, date_string=DATE_STRING)
    except TypeError:  # template without one of the optional kwargs
        full = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, **kw)
    cstart = full.index(prompt)
    cend = cstart + len(prompt)
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"])
           if a >= cstart and b <= cend and b > a]
    if not idx:  # degenerate fallback: pool everything
        idx = list(range(len(enc["input_ids"])))
    return enc["input_ids"], idx


def load_labelled(path):
    """Load harmful rows, refusing anything without a well-formed clustering."""
    rows = [x for x in json.load(open(path)) if x.get("source") == "toxic"]
    bad = [x["id"] for x in rows
           if not x.get("llm_clustering") or "error" in x["llm_clustering"]
           or "clusters" not in x["llm_clustering"]]
    if bad:
        raise SystemExit(
            f"{len(bad)} harmful rows in {path} have no valid clustering "
            f"(first ids: {bad[:5]}). Run experiments/rejudge_validated.py first -- "
            f"dropping them would silently shrink the dataset.")
    rows.sort(key=lambda x: x["id"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--clustered", required=True, help="validated clustered json")
    ap.add_argument("--benign", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tox = load_labelled(args.clustered)
    ben = sorted([x for x in json.load(open(args.benign)) if x["source"] == "hard"],
                 key=lambda x: x["id"])

    prompts = [x["prompt"] for x in tox] + [x["prompt"] for x in ben]
    n_tox, n = len(tox), len(prompts)
    entropy = np.array([x["llm_clustering"]["safety_entropy"] for x in tox] + [0.0] * len(ben), dtype=np.float32)
    score = np.array([x["llm_clustering"]["safety_score"] for x in tox] + [0.0] * len(ben), dtype=np.float32)
    joint = np.array([x["llm_clustering"]["joint_risk_target"] for x in tox] + [0.0] * len(ben), dtype=np.float32)
    source = np.array([1] * n_tox + [0] * len(ben), dtype=np.int64)
    print(f"{n} prompts: {n_tox} harmful(real T) + {len(ben)} benign(T=0)", flush=True)
    print(f"T: min={joint.min():.3f} max={joint.max():.3f} mean={joint.mean():.3f} "
          f">0.005={(joint>0.005).mean()*100:.1f}% distinct={len(set(joint.round(4)))}", flush=True)

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    # X63_TRUST_REMOTE_CODE=0: force the native transformers implementation
    # when a repo ships stale custom code (e.g. Phi-4-mini's modeling_phi3.py
    # imports LossKwargs, removed from modern transformers)
    trust = os.environ.get("X63_TRUST_REMOTE_CODE", "1") != "0"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust).eval()
    except ValueError:
        # multimodal-wrapped text models (gemma-style) register under the
        # conditional-generation auto class instead
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=trust).eval()

    tcfg = model.config.get_text_config()  # flat config returns self
    n_layers = tcfg.num_hidden_layers + 1  # + embedding
    hdim, vocab = tcfg.hidden_size, tcfg.vocab_size
    gb = n_layers * n * hdim * 2 / 1e9
    print(f"{n_layers} layers x {hdim} dim, vocab {vocab} -- "
          f"{gb:.1f} GB per state array (x2), logits {n*vocab*2/1e9:.1f} GB on disk", flush=True)

    tbg = np.empty((n_layers, n, hdim), dtype=np.float16)
    cmean = np.empty((n_layers, n, hdim), dtype=np.float16)
    os.makedirs(args.cache_dir, exist_ok=True)
    logits_mm = np.lib.format.open_memmap(
        os.path.join(args.cache_dir, "tbg_logits.npy"), mode="w+",
        dtype=np.float16, shape=(n, vocab))

    for i in tqdm(range(n), desc="extract"):
        ids, cidx = content_token_idx(tok, prompts[i])
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        for L, hs in enumerate(out.hidden_states):
            row = hs[0]  # [seq, hdim]
            tbg[L, i] = row[-1].float().cpu().numpy()
            cmean[L, i] = row[cidx].float().mean(0).cpu().numpy()
        logits_mm[i] = out.logits[0, -1].float().cpu().numpy().astype(np.float16)

    logits_mm.flush()
    del model
    torch.cuda.empty_cache()

    cache = {"tbg_states": [torch.from_numpy(tbg[L]) for L in range(n_layers)],
             "content_mean_states": [torch.from_numpy(cmean[L]) for L in range(n_layers)],
             "metadata": {"model_name": args.model, "n_samples_tbg": n,
                          "extractor": "extract_features_v2", "date_string": DATE_STRING},
             "num_layers": n_layers, "hidden_dim": hdim, "num_samples_tbg": n}
    torch.save(cache, os.path.join(args.cache_dir, "hidden_states.pt"))
    np.save(os.path.join(args.cache_dir, "safety_entropy_labels_tbg.npy"), entropy)
    np.save(os.path.join(args.cache_dir, "safety_score_labels_tbg.npy"), score)
    np.save(os.path.join(args.cache_dir, "joint_risk_target_labels_tbg.npy"), joint)
    np.save(os.path.join(args.cache_dir, "source_labels_tbg.npy"), source)
    json.dump(prompts, open(os.path.join(args.cache_dir, "prompts.json"), "w"))
    print(f"saved cache -> {args.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
