"""Stage 1: sample N=10 responses per HARMFUL prompt from the probed model
(Llama-3.2-3B-Instruct), same generation config as the paper / orbench s1.

Only the harmful (source=toxic) side is sampled; benign (hard) gets T=0 in
stage 3 with no generation (project convention, see swse-judge-inverts-benign-t).

Input : data/pools/cp_prompts.json  (from t0; override with --in)
Output: data/pools/cp_responses.json  (harmful rows gain "llm_responses": [10x]),
        saved incrementally for resume (override with --out).
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=os.path.join(POOLS, "cp_prompts.json"))
ap.add_argument("--out", dest="out", default=os.path.join(POOLS, "cp_responses.json"))
ARGS = ap.parse_args()
IN, OUT = ARGS.inp, ARGS.out

MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
TOP_P = 0.9
TOP_K = 50
NUM_GENERATIONS = 10
MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEVICE = "cuda"


def main():
    rows = [r for r in json.load(open(IN)) if r["source"] == "toxic"]
    print(f"{len(rows)} harmful prompts to sample", flush=True)

    done = {}
    if os.path.exists(OUT):
        for o in json.load(open(OUT)):
            done[o["id"]] = o
        print(f"resuming: {len(done)} already done", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()
    device = torch.device(DEVICE)

    out = [done[i] for i in sorted(done)]
    for k, r in enumerate(rows):
        if r["id"] in done:
            continue
        chat = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        inputs = tok(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                num_return_sequences=NUM_GENERATIONS, pad_token_id=tok.eos_token_id)
        plen = inputs["input_ids"].shape[-1]
        r = dict(r)
        r["llm_responses"] = [tok.decode(g[plen:], skip_special_tokens=True).strip() for g in gen]
        out.append(r)
        if k % 25 == 0:
            print(f"  [{k}/{len(rows)}] {r['prompt'][:55]!r}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"saved {len(out)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
