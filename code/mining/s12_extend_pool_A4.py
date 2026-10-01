"""Extend the pool with a fourth 5000+5000 (pool A4), same boundary-extension
recipe as A3: take the least-easy unused examples closest to the filtering
cuts. Harmful: 5000 lowest-oof of the remaining round-3 dropped (the slice
right after A3's picks). Benign: all 4675 remaining round-3 dropped + 325
highest-oof round-2 dropped.

Honest re-eval (fresh 5-fold CV, word + char) on A4 alone and A+A2+A3+A4
(20k+20k). Outputs: data/mined_pool_A4_bnd.json, extend_pool_A4_auc.json
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
sys.path.insert(0, HERE)
from s4_adversarial_filter import cv_auc_kind  # noqa: E402

D = POOLS
POOLS = ("mined_pool_A_rand.json", "mined_pool_A2_rand.json", "mined_pool_A3_bnd.json")


def main():
    prev = [json.load(open(os.path.join(D, f))) for f in POOLS]
    used = {r["text"] for pool in prev for r in pool}
    r2 = pd.read_parquet(os.path.join(D, "round_A_rand_2.parquet"))
    r3 = pd.read_parquet(os.path.join(D, "round_A_rand_3.parquet"))
    r4 = pd.read_parquet(os.path.join(D, "round_A_rand_4.parquet"))

    d3 = r3[(~r3.text.isin(set(r4.text))) & (~r3.text.isin(used))]
    dh = d3[d3.label == 1].nsmallest(5000, "oof")
    b3 = d3[d3.label == 0]
    d2 = r2[(~r2.text.isin(set(r3.text))) & (~r2.text.isin(used))]
    b2 = d2[d2.label == 0].nlargest(5000 - len(b3), "oof")
    print(f"harmful: 5000 r3-dropped (oof {dh.oof.min():.3f}-{dh.oof.max():.3f}); "
          f"benign: {len(b3)} r3-dropped + {len(b2)} r2-dropped "
          f"(oof {b2.oof.min():.3f}-{b2.oof.max():.3f})")

    a4 = pd.concat([dh, b3, b2], ignore_index=True)
    assert not a4.text.isin(used).any() and a4.text.nunique() == len(a4)
    assert int((a4.label == 1).sum()) == 5000 and int((a4.label == 0).sum()) == 5000
    a4[["text", "label", "source", "subtype", "oof"]].to_json(
        os.path.join(D, "mined_pool_A4_bnd.json"), orient="records", indent=1)
    print(f"pool A4: {len(a4)}")

    results = {}
    tx4, lb4 = a4.text.tolist(), a4.label.tolist()
    tx20 = [r["text"] for pool in prev for r in pool] + tx4
    lb20 = [r["label"] for pool in prev for r in pool] + lb4
    for tag, tx, lb in [("A4 (5k+5k)", tx4, lb4), ("A..A4 (20k+20k)", tx20, lb20)]:
        for kind in ("word", "char"):
            auc, sd = cv_auc_kind(tx, lb, kind)
            results[f"{tag} {kind}"] = round(auc, 4)
            print(f"{tag:16s} {kind:5s} CV AUC = {auc:.4f} (+-{sd:.4f})", flush=True)
    json.dump(results, open(os.path.join(HERE, "extend_pool_A4_auc.json"), "w"), indent=2)
    print("saved -> extend_pool_A4_auc.json")


if __name__ == "__main__":
    main()
