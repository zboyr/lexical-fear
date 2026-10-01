"""Iterative adversarial filtering with a RANDOM final cut.

s5 showed that iterative filtering drives the honest pool OOF AUC down to
~0.56-0.60 by the time the pool is 12-20k, but a final hardness-ranked cut to
5000+5000 overshoots into anti-correlated territory (retrained AUC bounced to
0.70 in variant A and INVERTED to 0.30 in variant B). Fix: once the pool is at
its last-round size, draw a RANDOM balanced 5000+5000 sample instead of taking
the hardness tail, then honestly re-evaluate with fresh retrained word- and
char-TFIDF models. Every round's pool is saved so alternative cuts can be
evaluated without re-mining.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from s2_auc import SEED
from s4_adversarial_filter import cv_auc_kind, oof_scores
from s5_iterative_filter import hardness

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
POOL = os.path.join(POOLS, "pool.parquet")
N_SIDE = 5000
SHRINK = 0.5


def run_variant(name, dfp):
    dfp = dfp.reset_index(drop=True)
    print(f"\n### variant {name}: start pool {len(dfp)}", flush=True)
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
        dfp[["text", "label", "source", "subtype", "oof"]].to_parquet(
            os.path.join(POOLS, f"round_{name}_{rnd}.parquet"))

        next_h, next_b = int(nh * SHRINK), int(nb * SHRINK)
        if next_h < N_SIDE or next_b < N_SIDE:
            break
        dfp = dfp.assign(hard=hardness(dfp))
        dfp = pd.concat([dfp[dfp.label == 1].nlargest(next_h, "hard"),
                         dfp[dfp.label == 0].nlargest(next_b, "hard")]).reset_index(drop=True)

    # final cut: RANDOM balanced sample from the last-round pool
    h = dfp[dfp.label == 1].sample(n=N_SIDE, random_state=SEED)
    b = dfp[dfp.label == 0].sample(n=N_SIDE, random_state=SEED)
    sel = pd.concat([h, b]).reset_index(drop=True)

    res = {"variant": name, "history": history, "n_harmful": len(h), "n_benign": len(b)}
    for kind in ["word", "char"]:
        m, s = cv_auc_kind(sel["text"].tolist(), sel["label"].tolist(), kind)
        print(f"  FINAL (random cut) retrained {kind}-TFIDF CV AUC: {m:.4f} ±{s:.4f}", flush=True)
        res[f"final_{kind}_auc"] = round(m, 4)
        res[f"final_{kind}_std"] = round(s, 4)

    lh, lb = h["text"].str.len(), b["text"].str.len()
    res["len_harmful_median"] = int(lh.median())
    res["len_benign_median"] = int(lb.median())
    comp = sel.groupby(["label", "source", "subtype"]).size()
    print(comp.to_string(), flush=True)
    res["composition"] = {f"{l}/{s}/{t}": int(n) for (l, s, t), n in comp.items()}

    out_pool = os.path.join(POOLS, f"mined_pool_{name}.json")
    sel[["text", "label", "source", "subtype", "oof"]].to_json(out_pool, orient="records", indent=1)
    print(f"  saved -> {out_pool}", flush=True)
    return res


def main():
    df = pd.read_parquet(POOL)
    results = []

    a = df[(df.source == "wjb") & (df.subtype.isin(["adversarial_harmful", "adversarial_benign"]))]
    results.append(run_variant("A_rand", a))

    parts = [g.sample(n=min(15000, len(g)), random_state=SEED)
             for _, g in df.groupby(["source", "label"])]
    results.append(run_variant("B_rand", pd.concat(parts).reset_index(drop=True)))

    out = os.path.join(RESULTS, "iterative_random_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
