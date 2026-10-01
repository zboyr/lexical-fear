"""Stage 2: cluster each harmful prompt's 10 responses with the local Qwen judge
and derive continuous T. Byte-identical judge logic to orbench s2 (imported).

Input : data/pools/cp_responses.json  (from t1; override with --in)
Output: data/pools/cp_clustered.json  (each harmful row gains "llm_clustering"; --out)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "orbench"))
from s2_cluster_local import (  # noqa: E402
    build_chat, find_gguf, make_judge_prompt, parse_json, preload_cuda, score_from_clusters)

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default=os.path.join(POOLS, "cp_responses.json"))
ap.add_argument("--out", dest="out", default=os.path.join(POOLS, "cp_clustered.json"))
ARGS = ap.parse_args()
IN, OUT = ARGS.inp, ARGS.out


def main():
    data = json.load(open(IN))
    print(f"loaded {len(data)} harmful prompts with responses", flush=True)

    done = {}
    if os.path.exists(OUT):
        for o in json.load(open(OUT)):
            if "llm_clustering" in o and "error" not in o["llm_clustering"]:
                done[o["id"]] = o
        print(f"resuming: {len(done)} already clustered", flush=True)

    preload_cuda()
    from llama_cpp import Llama
    llm = Llama(model_path=find_gguf(), n_gpu_layers=-1, n_ctx=8192, seed=42, verbose=False)

    out = [done[i] for i in sorted(done)]
    n_err = 0
    for k, item in enumerate(data):
        if item["id"] in done:
            continue
        chat = build_chat(make_judge_prompt(item["llm_responses"], item["prompt"]))
        try:
            res = llm.create_completion(prompt=chat, max_tokens=512, temperature=0.0,
                                        stop=["<|im_end|>", "<|im_start|>"])
            parsed = parse_json(res["choices"][0]["text"])
            item = dict(item)
            item["llm_clustering"] = score_from_clusters(parsed, len(item["llm_responses"]))
        except Exception as e:
            n_err += 1
            item = dict(item)
            item["llm_clustering"] = {"error": str(e)}
        out.append(item)
        if k % 25 == 0:
            c = item["llm_clustering"]
            msg = f"T={c['joint_risk_target']:.3f}" if "error" not in c else "ERR"
            print(f"  [{k}/{len(data)}] {msg}", flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
    json.dump(out, open(OUT, "w"), indent=1)

    ok = [o for o in out if "error" not in o["llm_clustering"]]
    Ts = np.array([o["llm_clustering"]["joint_risk_target"] for o in ok])
    print(f"saved {len(out)} (errors {n_err}); harmful T mean={Ts.mean():.3f} "
          f">0.005={(Ts>0.005).mean()*100:.0f}% max={Ts.max():.3f}", flush=True)


if __name__ == "__main__":
    main()
