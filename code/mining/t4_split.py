"""Split a full feature cache into train/test with a stratified split
(seed 42, stratify by source) matching what the eval uses.

Defaults: confusable_A_full -> confusable_A_{train,test}, test_size 0.3.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="feature_caches/confusable_A_full")
ap.add_argument("--train", default="feature_caches/confusable_A_train")
ap.add_argument("--test", default="feature_caches/confusable_A_test")
ap.add_argument("--test_size", type=float, default=0.3)
ARGS = ap.parse_args()
SRC, TRAIN, TEST = ARGS.src, ARGS.train, ARGS.test
MODEL = "unsloth/Llama-3.2-3B-Instruct"


def write_cache(out, idx, hs, entropy, score, joint, source, prompts):
    os.makedirs(out, exist_ok=True)
    tbg = [layer[idx] for layer in hs]
    cache = {"tbg_states": tbg,
             "metadata": {"model_name": MODEL, "n_samples_tbg": len(idx)},
             "num_layers": len(tbg), "hidden_dim": tbg[0].shape[-1],
             "num_samples_tbg": tbg[0].shape[0]}
    torch.save(cache, os.path.join(out, "hidden_states.pt"))
    np.save(os.path.join(out, "safety_entropy_labels_tbg.npy"), entropy[idx])
    np.save(os.path.join(out, "safety_score_labels_tbg.npy"), score[idx])
    np.save(os.path.join(out, "joint_risk_target_labels_tbg.npy"), joint[idx])
    np.save(os.path.join(out, "source_labels_tbg.npy"), source[idx])
    json.dump([prompts[i] for i in idx], open(os.path.join(out, "prompts.json"), "w"))
    print(f"  {out}: n={len(idx)} harmful={int(source[idx].sum())} "
          f"risk+={int((joint[idx]>0.005).sum())}")


def main():
    hs = torch.load(f"{SRC}/hidden_states.pt", weights_only=False)["tbg_states"]
    entropy = np.load(f"{SRC}/safety_entropy_labels_tbg.npy")
    score = np.load(f"{SRC}/safety_score_labels_tbg.npy")
    joint = np.load(f"{SRC}/joint_risk_target_labels_tbg.npy")
    source = np.load(f"{SRC}/source_labels_tbg.npy")
    prompts = json.load(open(f"{SRC}/prompts.json"))

    n = len(source)
    tr, te = train_test_split(np.arange(n), test_size=ARGS.test_size, random_state=42, stratify=source)
    print(f"n={n} -> train={len(tr)} / test={len(te)}")
    write_cache(TRAIN, tr, hs, entropy, score, joint, source, prompts)
    write_cache(TEST, te, hs, entropy, score, joint, source, prompts)


if __name__ == "__main__":
    main()
