"""Iterative adversarial filtering (AFLite-style) to mine a maximally
TF-IDF-confusable 5000+5000 pool.

Single-round filtering (s4) only got the retrained AUC from 0.98 to 0.90:
the lexical signal is distributed, so features that survive round 1 become
useful in round 2. Iterate instead:

  round k: 5-fold OOF word-TFIDF-LR scores on the CURRENT pool
           -> per side, keep the hardest half (harmful with lowest score,
              benign with highest score)
  stop when both sides would drop below N_SIDE; final cut = exactly N_SIDE
  hardest per side.

After each round we report the pool OOF AUC (honest: every score comes from a
model that never saw the example, retrained on the current filtered pool).
Final subset additionally gets a fresh retrained word- and char-TFIDF CV AUC.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s2_auc import SEED
from s4_adversarial_filter import cv_auc_kind, oof_scores

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
POOL = os.path.join(POOLS, "pool.parquet")
N_SIDE = 5000
SHRINK = 0.5


def hardness(df):
    # high = harder for the classifier (harmful scored low / benign scored high)
    return np.where(df["label"] == 1, -df["oof"], df["oof"])


def run_variant(name, dfp, out_pool_path):
    dfp = dfp.reset_index(drop=True)
    nh, nb = int((dfp["label"] == 1).sum()), int((dfp["label"] == 0).sum())
    print(f"\n### variant {name}: start pool {len(dfp)} ({nh} harmful / {nb} benign)", flush=True)
    history = []
    rnd = 0
    while True:
        rnd += 1
        scores = oof_scores(dfp["text"].tolist(), dfp["label"].tolist())
        dfp = dfp.assign(oof=scores)
        auc = roc_auc_score(dfp["label"], scores)
        nh, nb = int((dfp["label"] == 1).sum()), int((dfp["label"] == 0).sum())
        print(f"  round {rnd}: n={len(dfp)} ({nh}h/{nb}b) OOF AUC={auc:.4f}", flush=True)
        history.append({"round": rnd, "n": len(dfp), "oof_auc": round(auc, 4)})

        next_h, next_b = int(nh * SHRINK), int(nb * SHRINK)
        last = next_h < N_SIDE or next_b < N_SIDE
        keep_h, keep_b = (N_SIDE, N_SIDE) if last else (next_h, next_b)
        dfp = dfp.assign(hard=hardness(dfp))
        dfp = pd.concat([dfp[dfp.label == 1].nlargest(keep_h, "hard"),
                         dfp[dfp.label == 0].nlargest(keep_b, "hard")]).reset_index(drop=True)
        if last:
            break

    # honest evaluation of the final subset with fresh retrained models
    res = {"variant": name, "history": history,
           "n_harmful": int((dfp["label"] == 1).sum()),
           "n_benign": int((dfp["label"] == 0).sum())}
    for kind in ["word", "char"]:
        m, s = cv_auc_kind(dfp["text"].tolist(), dfp["label"].tolist(), kind)
        print(f"  FINAL retrained {kind}-TFIDF CV AUC: {m:.4f} ±{s:.4f}", flush=True)
        res[f"final_{kind}_auc"] = round(m, 4)
        res[f"final_{kind}_std"] = round(s, 4)

    lh = dfp[dfp.label == 1]["text"].str.len()
    lb = dfp[dfp.label == 0]["text"].str.len()
    res["len_harmful_median"] = int(lh.median())
    res["len_benign_median"] = int(lb.median())
    comp = dfp.groupby(["label", "source", "subtype"]).size()
    print(comp.to_string(), flush=True)
    res["composition"] = {f"{l}/{s}/{t}": int(n) for (l, s, t), n in comp.items()}

    dfp[["text", "label", "source", "subtype", "oof"]].to_json(
        out_pool_path, orient="records", indent=1)
    print(f"  saved -> {out_pool_path}", flush=True)
    return res


def main():
    df = pd.read_parquet(POOL)
    results = []

    a = df[(df.source == "wjb") & (df.subtype.isin(["adversarial_harmful", "adversarial_benign"]))]
    results.append(run_variant("A_wjb_adv_iter", a,
                               os.path.join(POOLS, "mined_pool_A_iter.json")))

    parts = [g.sample(n=min(15000, len(g)), random_state=SEED)
             for _, g in df.groupby(["source", "label"])]
    b = pd.concat(parts).reset_index(drop=True)
    results.append(run_variant("B_multisource_iter", b,
                               os.path.join(POOLS, "mined_pool_B_iter.json")))

    out = os.path.join(RESULTS, "iterative_filter_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
