"""Extend the pool with a third 5000+5000 (pool A3), keeping TF-IDF low.

Round-4 leftovers after A+A2: 341 harmful, 0 benign — so A3 comes mostly from
the round-3 DROPPED half, selected by boundary extension: the least-easy of
the dropped side (oof closest to the round-3 keep cut). For harmful (label=1,
easy = high oof) that means lowest-oof dropped; for benign (easy = low oof),
highest-oof dropped. This extends the kept-half boundary rather than cherry-
picking extreme hardness, so it cannot overshoot into anti-signal.

Honest re-eval (fresh 5-fold CV, word + char) on A3 alone and A+A2+A3 (15k+15k).

Outputs: data/mined_pool_A3_bnd.json, extend_pool_A3_auc.json
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
    pool_a2 = json.load(open(os.path.join(D, "mined_pool_A2_rand.json")))
    used = {r["text"] for r in pool_a} | {r["text"] for r in pool_a2}
    r4 = pd.read_parquet(os.path.join(D, "round_A_rand_4.parquet"))
    r3 = pd.read_parquet(os.path.join(D, "round_A_rand_3.parquet"))

    r4_rem = r4[~r4.text.isin(used)]
    h_r4 = r4_rem[r4_rem.label == 1]
    print(f"round-4 leftovers: harmful {len(h_r4)}, benign {int((r4_rem.label==0).sum())}")

    dropped = r3[(~r3.text.isin(set(r4.text))) & (~r3.text.isin(used))]
    dh = dropped[dropped.label == 1].nsmallest(5000 - len(h_r4), "oof")
    db = dropped[dropped.label == 0].nlargest(5000, "oof")
    print(f"boundary picks: harmful {len(dh)} (oof {dh.oof.min():.3f}-{dh.oof.max():.3f}), "
          f"benign {len(db)} (oof {db.oof.min():.3f}-{db.oof.max():.3f})")

    a3 = pd.concat([h_r4, dh, db], ignore_index=True)
    assert not a3.text.isin(used).any() and a3.text.nunique() == len(a3)
    assert int((a3.label == 1).sum()) == 5000 and int((a3.label == 0).sum()) == 5000
    a3[["text", "label", "source", "subtype", "oof"]].to_json(
        os.path.join(D, "mined_pool_A3_bnd.json"), orient="records", indent=1)
    print(f"pool A3: {len(a3)}")

    results = {}
    tx3, lb3 = a3.text.tolist(), a3.label.tolist()
    tx15 = [r["text"] for r in pool_a + pool_a2] + tx3
    lb15 = [r["label"] for r in pool_a + pool_a2] + lb3
    for tag, tx, lb in [("A3 (5k+5k)", tx3, lb3), ("A+A2+A3 (15k+15k)", tx15, lb15)]:
        for kind in ("word", "char"):
            auc, sd = cv_auc_kind(tx, lb, kind)
            results[f"{tag} {kind}"] = round(auc, 4)
            print(f"{tag:18s} {kind:5s} CV AUC = {auc:.4f} (+-{sd:.4f})", flush=True)
    json.dump(results, open(os.path.join(RESULTS, "extend_pool_A3_auc.json"), "w"), indent=2)
    print("saved -> extend_pool_A3_auc.json")


if __name__ == "__main__":
    main()
