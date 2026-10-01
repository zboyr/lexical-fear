"""Stage 3: extract Llama-3.2-3B TBG hidden states for all 2000 prompts and cache
with real T on the harmful side (from t2) and T=0 on the benign side.

Positional contract: harmful (toxic, real T) first, then benign (hard, T=0).
Output: feature_caches/confusable_A_full/
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_features import extract_hidden_states_tbg, save_hidden_states  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
MODEL = "unsloth/Llama-3.2-3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="feature_caches/confusable_A_full")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clustered", default=os.path.join(POOLS, "cp_clustered.json"),
                    help="harmful prompts with llm_clustering (from t2)")
    ap.add_argument("--prompts", default=os.path.join(POOLS, "cp_prompts.json"),
                    help="file whose source=hard rows form the benign side")
    args = ap.parse_args()

    # harmful: real judged T (drop judge errors)
    tox = [x for x in json.load(open(args.clustered)) if x.get("source") == "toxic"
           and "llm_clustering" in x and "error" not in x["llm_clustering"]]
    tox.sort(key=lambda x: x["id"])
    tox_prompts = [x["prompt"] for x in tox]
    tox_entropy = [x["llm_clustering"]["safety_entropy"] for x in tox]
    tox_score = [x["llm_clustering"]["safety_score"] for x in tox]
    tox_joint = [x["llm_clustering"]["joint_risk_target"] for x in tox]

    # benign: T=0
    ben = [x for x in json.load(open(args.prompts)) if x["source"] == "hard"]
    ben.sort(key=lambda x: x["id"])
    ben_prompts = [x["prompt"] for x in ben]

    prompts = tox_prompts + ben_prompts
    entropy = np.array(tox_entropy + [0.0] * len(ben_prompts), dtype=np.float32)
    score = np.array(tox_score + [0.0] * len(ben_prompts), dtype=np.float32)
    joint = np.array(tox_joint + [0.0] * len(ben_prompts), dtype=np.float32)
    source = np.array([1] * len(tox_prompts) + [0] * len(ben_prompts), dtype=np.int64)

    print(f"{len(prompts)} prompts: {len(tox_prompts)} harmful(real T) + {len(ben_prompts)} benign(T=0)")
    print(f"T: min={joint.min():.3f} max={joint.max():.3f} mean={joint.mean():.3f} "
          f">0.005={(joint>0.005).mean()*100:.0f}%")

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()

    tbg = extract_hidden_states_tbg(prompts, model, tok, device, batch_size=1)

    os.makedirs(args.cache_dir, exist_ok=True)
    save_hidden_states({"tbg": tbg}, os.path.join(args.cache_dir, "hidden_states.pt"),
                       metadata={"model_name": MODEL, "n_samples_tbg": len(prompts)})
    np.save(os.path.join(args.cache_dir, "safety_entropy_labels_tbg.npy"), entropy)
    np.save(os.path.join(args.cache_dir, "safety_score_labels_tbg.npy"), score)
    np.save(os.path.join(args.cache_dir, "joint_risk_target_labels_tbg.npy"), joint)
    np.save(os.path.join(args.cache_dir, "source_labels_tbg.npy"), source)
    json.dump(prompts, open(os.path.join(args.cache_dir, "prompts.json"), "w"))
    print(f"saved cache -> {args.cache_dir} ({len(tbg)} layers, dim {tbg[0].shape[-1]})")


if __name__ == "__main__":
    main()
