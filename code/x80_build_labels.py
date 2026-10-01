"""x80 Stage 0b: per-prompt refusal/compliance counts in feature-cache row order.

Writes data/runs/x80/x80_labels.npz, the --labels input of
x80_fit_paired_probes.py:
  rows     row of each retained harmful prompt in the 30,000-row cache order
           (index into the id-sorted harmful side of cp_15k_clustered_v2.json)
  ids      the 13,527 retained harmful prompt ids (sorted)
  R_count  number of the 10 Qwen3.5-4B responses judged refusal2 = true (int8)
  H_count  number of the 10 responses judged comply2 = true (int8)

Extracted from the inline script that produced the released file; the logic is
unchanged. As a consistency check it prints whether the rows agree with the
retained harmful rows of data/x61_mask_tbg.npy.

Usage, from anywhere inside the checkout:
  python code/x80_build_labels.py [--out data/runs/x80/x80_labels.npz]
"""
import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")


def load(path, key):
    d = {}
    for line in open(path):
        r = json.loads(line)
        if not r.get("failed"):
            d[r["id"]] = sum(r[key])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(DATA, "runs", "x80", "x80_labels.npz"))
    args = ap.parse_args()

    pool_ids = sorted(r["id"] for r in json.load(open(os.path.join(DATA, "pools", "cp_15k_clustered_v2.json")))
                      if r["source"] == "toxic")
    row = {i: k for k, i in enumerate(pool_ids)}
    R = load(os.path.join(DATA, "judged", "x64_refusal2_qwen_s42.jsonl"), "refusal")
    H = load(os.path.join(DATA, "judged", "x64_comply2_qwen_s42.jsonl"), "complied")
    ids = sorted(int(json.loads(line)["source_id"]) for line in
                 open(os.path.join(DATA, "datasets", "swse_x61_majority_harmful_v1", "prompts.jsonl")))
    rows = np.array([row[i] for i in ids])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, rows=rows, ids=np.array(ids),
             R_count=np.array([R[i] for i in ids], dtype=np.int8),
             H_count=np.array([H[i] for i in ids], dtype=np.int8))
    m = np.load(os.path.join(DATA, "x61_mask_tbg.npy"))
    print("mask agrees", set(np.flatnonzero(m[:15000])) == set(rows), len(rows))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
