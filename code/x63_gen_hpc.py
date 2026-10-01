"""x63 HPC generation: 10 untruncated responses per harmful prompt, sharded.

Generalizes the project's earlier single-model generator (x59, not part of this
release) to any HF chat model and adds --start/--end row sharding so many 2h
GPU jobs can split one model's 13,527 prompts.
Sampling config is identical to x59 (T=1.0, top_p 0.9, top_k 50, N=10,
max_new 2048 with a per-row 4096 retry, thinking off, torch.manual_seed(42),
left padding, OOM bisection). Resume-safe per shard output file.

Usage (directly, or via code/x63_gen.sbatch), from the repository root:
  python code/x63_gen_hpc.py --model Qwen/Qwen3.5-4B \
    --in data/prompts/x63_prompts13527.json \
    --out data/generations/x63_qwen_0_1150.json \
    --start 0 --end 1150
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", dest="out", required=True)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=0, help="exclusive; 0 = to the end")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows of the shard")
ARGS = ap.parse_args()

MAX_NEW, RETRY_NEW = 2048, 4096
TEMPERATURE, TOP_P, TOP_K, N = 1.0, 0.9, 50, 10


def eos_id_set(tok, model):
    ids = set()
    for v in (tok.eos_token_id, model.generation_config.eos_token_id):
        if v is None:
            continue
        ids.update(v if isinstance(v, (list, tuple)) else [v])
    return ids


def decode_row(seqs, tok, eos_ids, max_new):
    texts, lens, trunc = [], [], []
    for seq in seqs:
        ids = seq.tolist()
        hit = [i for i, t in enumerate(ids) if t in eos_ids]
        if hit:
            n_tok, tr = hit[0] + 1, False
        else:
            n_tok, tr = len(ids), len(ids) >= max_new
        texts.append(tok.decode(seq, skip_special_tokens=True).strip())
        lens.append(n_tok)
        trunc.append(tr)
    return texts, lens, trunc


def gen_batch(model, tok, eos_ids, chats, max_new):
    try:
        inputs = tok(chats, return_tensors="pt", padding=True,
                     add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=True,
                temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                num_return_sequences=N, pad_token_id=tok.pad_token_id)
        plen = inputs["input_ids"].shape[1]
        out = out[:, plen:].reshape(len(chats), N, -1)
        return [decode_row(out[i], tok, eos_ids, max_new) for i in range(len(chats))]
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(chats) == 1:
            raise
        mid = len(chats) // 2
        return gen_batch(model, tok, eos_ids, chats[:mid], max_new) + \
            gen_batch(model, tok, eos_ids, chats[mid:], max_new)


def load_model(name):
    try:
        return AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    except ValueError:
        # multimodal-wrapped text models (gemma-style) register under the
        # conditional-generation auto class instead
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()


def checkpoint(out):
    # atomic write so a mid-dump SIGKILL (2h wall) never corrupts the resume file
    json.dump(out, open(ARGS.out + ".tmp", "w"))
    os.replace(ARGS.out + ".tmp", ARGS.out)


def main():
    rows = json.load(open(ARGS.inp))
    end = ARGS.end or len(rows)
    rows = rows[ARGS.start:end]
    if ARGS.limit:
        rows = rows[:ARGS.limit]
    print(f"{ARGS.model}: shard [{ARGS.start},{end}) -> {len(rows)} prompts", flush=True)

    done = {}
    if os.path.exists(ARGS.out):
        try:
            prior = json.load(open(ARGS.out))
        except (ValueError, json.JSONDecodeError):
            print("resume file corrupt (killed mid-write); starting shard fresh", flush=True)
            prior = []
        for o in prior:
            # rows truncated at the 4096 retry cap are final; re-rolling them on
            # every resume would silently replace already-judged responses
            if len(o.get("llm_responses", [])) == N:
                done[o["id"]] = o
        print(f"resuming: {len(done)}", flush=True)

    tok = AutoTokenizer.from_pretrained(ARGS.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_model(ARGS.model)
    eos_ids = eos_id_set(tok, model)
    torch.manual_seed(42)

    todo = [r for r in rows if r["id"] not in done]
    for r in todo:
        r["_chat"] = tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    todo.sort(key=lambda r: len(r["_chat"]))

    out = list(done.values())
    for b in range(0, len(todo), ARGS.batch):
        batch = todo[b:b + ARGS.batch]
        results = gen_batch(model, tok, eos_ids, [r["_chat"] for r in batch], MAX_NEW)
        for r, (texts, lens, trunc) in zip(batch, results):
            if any(trunc):
                texts, lens, trunc = gen_batch(model, tok, eos_ids, [r["_chat"]], RETRY_NEW)[0]
            out.append({"id": r["id"], "prompt": r["prompt"], "model": ARGS.model,
                        "llm_responses": texts, "gen_tokens": lens, "truncated": trunc})
        print(f"  [{min(b + ARGS.batch, len(todo))}/{len(todo)}] done", flush=True)
        checkpoint(out)
    checkpoint(out)
    all_lens = [n for o in out for n in o["gen_tokens"]]
    n_trunc = sum(sum(o["truncated"]) for o in out)
    print(f"SHARD DONE {len(out)} rows, {len(all_lens)} resp, {n_trunc} truncated; "
          f"tok mean {sum(all_lens) // max(1, len(all_lens))}", flush=True)


if __name__ == "__main__":
    main()
