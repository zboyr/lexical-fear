"""Extend pool A with a second 5000+5000 (pool A2) drawn from the same mining
run, keeping TF-IDF separability as low as possible.

Sources:
  harmful: random 5000 of the 5341 round-4 leftovers (random cut, per the
           lesson that hardness-ranked final cuts overshoot into anti-signal)
  benign : all 4837 round-4 leftovers + 163 top-up from round-3 dropped benign
           with the highest oof (= just below the round-3 keep cut, i.e. the
           least-easy of the dropped; avoids injecting easy examples)

Honest re-eval (fresh retrained 5-fold CV, word + char TF-IDF) on:
  A2 alone (5k+5k) and A+A2 combined (10k+10k).

Outputs: data/mined_pool_A2_rand.json, extend_pool_A2_auc.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
sys.path.insert(0, HERE)
from s4_adversarial_filter import cv_auc_kind  # noqa: E402

D = POOLS
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    pool_a = json.load(open(os.path.join(D, "mined_pool_A_rand.json")))
    used = {r["text"] for r in pool_a}
    r4 = pd.read_parquet(os.path.join(D, "round_A_rand_4.parquet"))
    r3 = pd.read_parquet(os.path.join(D, "round_A_rand_3.parquet"))

    rem = r4[~r4.text.isin(used)]
    h = rem[rem.label == 1].reset_index(drop=True)
    b = rem[rem.label == 0].reset_index(drop=True)
    print(f"round-4 remainder: harmful {len(h)}, benign {len(b)}")

    h_sel = h.iloc[rng.choice(len(h), 5000, replace=False)]

    n_top = 5000 - len(b)
    excl = used | set(r4.text)
    dropped_b = r3[(r3.label == 0) & (~r3.text.isin(excl))]
    top = dropped_b.nlargest(n_top, "oof")
    print(f"benign top-up: {n_top} from round-3 dropped (oof {top.oof.min():.3f}"
          f"-{top.oof.max():.3f}, dropped-pool max {dropped_b.oof.max():.3f})")
    b_sel = pd.concat([b, top], ignore_index=True)

    a2 = pd.concat([h_sel, b_sel], ignore_index=True)
    assert not a2.text.isin(used).any() and a2.text.nunique() == len(a2)
    a2[["text", "label", "source", "subtype", "oof"]].to_json(
        os.path.join(D, "mined_pool_A2_rand.json"), orient="records", indent=1)
    print(f"pool A2: {len(a2)} ({int((a2.label==1).sum())}+{int((a2.label==0).sum())})")

    results = {}
    texts2, labels2 = a2.text.tolist(), a2.label.tolist()
    texts10 = [r["text"] for r in pool_a] + texts2
    labels10 = [r["label"] for r in pool_a] + labels2
    for tag, tx, lb in [("A2 (5k+5k)", texts2, labels2),
                        ("A+A2 (10k+10k)", texts10, labels10)]:
        for kind in ("word", "char"):
            auc, sd = cv_auc_kind(tx, lb, kind)
            results[f"{tag} {kind}"] = round(auc, 4)
            print(f"{tag:16s} {kind:5s} CV AUC = {auc:.4f} (+-{sd:.4f})", flush=True)
    json.dump(results, open(os.path.join(RESULTS, "extend_pool_A2_auc.json"), "w"), indent=2)
    print("saved -> extend_pool_A2_auc.json")


if __name__ == "__main__":
    main()
