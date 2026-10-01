"""Build the final ~5000 harmful + 5000 benign combined pool and measure its AUC.

Strategy: allocate quota across the lowest-AUC within-source pairings (so that
source identity stays uncorrelated with the label — each source contributes to
both sides where possible), then report the TF-IDF LR 5-fold CV AUC of the
combined pool, plus a leave-one-component-out sensitivity check.

The component mix is data-driven: edit COMPONENTS after inspecting s2 results.
"""
import json
import os

import numpy as np
import pandas as pd

from s2_auc import cv_auc, SEED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
POOL = os.path.join(POOLS, "pool.parquet")
OUT_JSON = os.path.join(RESULTS, "combined_pool_auc.json")
OUT_POOL = os.path.join(POOLS, "combined_pool.json")

# (name, harmful (source, subtypes|None), benign (source, subtypes|None), quota per side)
# lowest-AUC within-source pairings from s2, topped up with WJB adversarial
COMPONENTS = [
    ("jbb", ("jbb", ["harmful"]), ("jbb", ["benign"]), 100),
    ("xstest", ("xstest", None), ("xstest", None), 200),
    ("orbench", ("orbench", ["toxic"]), ("orbench", ["hard"]), 655),
    ("wjb_adv", ("wjb", ["adversarial_harmful"]), ("wjb", ["adversarial_benign"]), 4045),
]


def take(df, source, subtypes, label, n, seed=SEED):
    m = (df["source"] == source) & (df["label"] == label)
    if subtypes is not None:
        m &= df["subtype"].isin(subtypes)
    d = df[m]
    return d.sample(n=min(n, len(d)), random_state=seed)


def build(df, components):
    parts = []
    for name, (hs, hsub), (bs, bsub), quota in components:
        h = take(df, hs, hsub, 1, quota)
        b = take(df, bs, bsub, 0, quota)
        n = min(len(h), len(b))  # keep each component internally balanced
        h, b = h.iloc[:n], b.iloc[:n]
        h = h.assign(component=name)
        b = b.assign(component=name)
        parts.append(pd.concat([h, b]))
        print(f"  component {name:12s}: {n} per side")
    return pd.concat(parts).reset_index(drop=True)


def main():
    df = pd.read_parquet(POOL)
    print("building combined pool:")
    pool = build(df, COMPONENTS)
    nh, nb = int((pool.label == 1).sum()), int((pool.label == 0).sum())
    print(f"total: {nh} harmful + {nb} benign")

    mean, std = cv_auc(pool["text"].tolist(), pool["label"].tolist())
    print(f"\ncombined pool TF-IDF LR AUC: {mean:.4f} ±{std:.4f}")

    report = {"n_harmful": nh, "n_benign": nb,
              "auc_mean": round(mean, 4), "auc_std": round(std, 4),
              "components": [c[0] for c in COMPONENTS], "loo": []}

    if len(COMPONENTS) > 1:
        print("\nleave-one-component-out:")
        for name, *_ in COMPONENTS:
            sub = pool[pool.component != name]
            m, s = cv_auc(sub["text"].tolist(), sub["label"].tolist())
            print(f"  without {name:12s}: AUC={m:.4f} ±{s:.4f}  (n={len(sub)})")
            report["loo"].append({"without": name, "auc_mean": round(m, 4), "auc_std": round(s, 4)})

    json.dump(report, open(OUT_JSON, "w"), indent=2)
    pool[["text", "label", "source", "subtype", "component"]].to_json(
        OUT_POOL, orient="records", indent=1)
    print(f"\nsaved report -> {OUT_JSON}\nsaved pool  -> {OUT_POOL}")


if __name__ == "__main__":
    main()
