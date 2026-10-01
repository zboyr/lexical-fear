"""x66: split_cache_v2 with an x61 row mask and optional benign subsampling.

Identical to split_cache_v2.py (90/10, stratified by source, same memory
discipline) except rows where the cache's x61_mask_tbg.npy is False are
dropped BEFORE the split -- those harmful prompts left the retained set in
x61 and have no v4/v5 labels. --benign_n N keeps only the first N rows of a
fixed seed-42 permutation of the benign rows (nested subsets across N, for
the x67 benign-count sweep). Also carries the v5 sidecar label files and
writes split_indices.json (original full-cache row ids of train/test) for
posthoc within-harmful reporting.
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

KEYS = ("tbg_states", "content_mean_states")
LABEL_FILES = ("safety_entropy_labels_tbg.npy", "safety_score_labels_tbg.npy",
               "joint_risk_target_labels_tbg.npy", "source_labels_tbg.npy")
OPTIONAL_LABEL_FILES = ("probt_labels_tbg.npy", "v5_score_labels_tbg.npy",
                        "v5_entropy_labels_tbg.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--benign_n", type=int, default=-1,
                    help="keep only the first N of a seed-42 permutation of "
                         "the benign rows (-1 = all)")
    args = ap.parse_args()

    labels = {n: np.load(f"{args.src}/{n}") for n in LABEL_FILES}
    for n in OPTIONAL_LABEL_FILES:
        if os.path.exists(f"{args.src}/{n}"):
            labels[n] = np.load(f"{args.src}/{n}")
            print(f"carrying optional label file {n}", flush=True)
    mask = np.load(f"{args.src}/x61_mask_tbg.npy")
    source = labels["source_labels_tbg.npy"]
    if args.benign_n >= 0:
        ben = np.flatnonzero(source == 0)
        drop = np.random.RandomState(42).permutation(ben)[args.benign_n:]
        mask = mask.copy()
        mask[drop] = False
        print(f"benign_n={args.benign_n}: kept benign "
              f"{int(mask[source == 0].sum())}", flush=True)
    keep = np.flatnonzero(mask)
    prompts = json.load(open(f"{args.src}/prompts.json"))
    print(f"mask keeps {len(keep)}/{len(mask)} rows "
          f"(harmful {int(source[keep].sum())})", flush=True)

    tr, te = train_test_split(keep, test_size=args.test_size,
                              random_state=args.seed, stratify=source[keep])
    print(f"-> train={len(tr)} / test={len(te)} (split seed {args.seed})", flush=True)

    for out, idx in ((args.train, tr), (args.test, te)):
        os.makedirs(out, exist_ok=True)
        for name, arr in labels.items():
            np.save(os.path.join(out, name), arr[idx])
        json.dump([prompts[i] for i in idx], open(os.path.join(out, "prompts.json"), "w"))
        json.dump(idx.tolist(), open(os.path.join(out, "split_indices.json"), "w"))
        risk = int((labels["joint_risk_target_labels_tbg.npy"][idx] > 0.005).sum())
        print(f"  {out}: n={len(idx)} harmful={int(source[idx].sum())} risk+={risk}", flush=True)

    src_lg = np.load(f"{args.src}/tbg_logits.npy", mmap_mode="r")
    for out, idx in ((args.train, tr), (args.test, te)):
        dst = np.lib.format.open_memmap(f"{out}/tbg_logits.npy", mode="w+",
                                        dtype=src_lg.dtype, shape=(len(idx), src_lg.shape[1]))
        for j0 in range(0, len(idx), 2048):
            sel = idx[j0:j0 + 2048]
            dst[j0:j0 + len(sel)] = src_lg[sel]
        dst.flush()
        del dst
    del src_lg
    print("  logits split", flush=True)

    hs = torch.load(f"{args.src}/hidden_states.pt", weights_only=False)
    tr_t, te_t = torch.from_numpy(tr), torch.from_numpy(te)
    sliced = {out: {k: [] for k in KEYS} for out in (args.train, args.test)}
    for k in KEYS:
        layers = hs[k]
        for L in range(len(layers)):
            sliced[args.train][k].append(layers[L][tr_t].clone())
            sliced[args.test][k].append(layers[L][te_t].clone())
            layers[L] = None
        hs[k] = None

    for out, idx in ((args.train, tr), (args.test, te)):
        cache = {k: sliced[out][k] for k in KEYS}
        cache.update({"metadata": {**hs.get("metadata", {}), "n_samples_tbg": len(idx),
                                   "split_from": args.src},
                      "num_layers": hs["num_layers"], "hidden_dim": hs["hidden_dim"],
                      "num_samples_tbg": len(idx)})
        torch.save(cache, f"{out}/hidden_states.pt")
        del cache, sliced[out]
    print("done", flush=True)


if __name__ == "__main__":
    main()
