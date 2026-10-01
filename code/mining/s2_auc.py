"""TF-IDF + LogisticRegression harmful-vs-benign AUC for each candidate pairing.

Same text baseline as the project (auc_summary.py): TfidfVectorizer(word 1-2gram,
min_df=2, sublinear_tf) + LogisticRegression(class_weight='balanced').
Metric: 5-fold stratified CV ROC AUC (vectorizer fit inside each fold).

Pairings are balanced-sampled to at most CAP per side (target pool is ~5000/5000),
seed fixed for reproducibility. Lower AUC = more lexically confusable.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
POOL = os.path.join(POOLS, "pool.parquet")
CAP = 5000
SEED = 42


def cv_auc(texts, labels, n_splits=5, seed=SEED):
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(texts, labels):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform(texts[tr])
        Xte = vec.transform(texts[te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, labels[tr])
        aucs.append(roc_auc_score(labels[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def sample_side(df, cap=CAP, seed=SEED):
    return df.sample(n=min(cap, len(df)), random_state=seed)


def pairing(df, name, harm_sel, ben_sel):
    h = df[harm_sel(df)]
    b = df[ben_sel(df)]
    h, b = sample_side(h), sample_side(b)
    sub = pd.concat([h, b])
    mean, std = cv_auc(sub["text"].tolist(), sub["label"].tolist())
    res = {"pairing": name, "n_harmful": len(h), "n_benign": len(b),
           "auc_mean": round(mean, 4), "auc_std": round(std, 4)}
    print(f"{name:45s} h={len(h):5d} b={len(b):5d}  AUC={mean:.4f} ±{std:.4f}")
    return res


def main():
    df = pd.read_parquet(POOL)
    print(f"pool: {len(df)} rows\n")
    S, T = "source", "subtype"
    results = []

    def sel(src, subs=None, label=None):
        def f(d):
            m = d[S] == src
            if subs is not None:
                m &= d[T].isin(subs)
            if label is not None:
                m &= d["label"] == label
            return m
        return f

    # within-source designed-confusable pairings
    results.append(pairing(df, "wjb vanilla_harmful vs vanilla_benign",
                           sel("wjb", ["vanilla_harmful"]), sel("wjb", ["vanilla_benign"])))
    results.append(pairing(df, "wjb adversarial_harmful vs adversarial_benign",
                           sel("wjb", ["adversarial_harmful"]), sel("wjb", ["adversarial_benign"])))
    results.append(pairing(df, "orbench toxic vs hard-1k",
                           sel("orbench", ["toxic"]), sel("orbench", ["hard"])))
    results.append(pairing(df, "orbench toxic vs 80k",
                           sel("orbench", ["toxic"]), sel("orbench", ["80k"])))
    results.append(pairing(df, "coconot original vs contrast (safety)",
                           sel("coconot", ["original_safety"]), sel("coconot", ["contrast_safety"])))
    results.append(pairing(df, "xstest unsafe vs safe",
                           lambda d: (d[S] == "xstest") & (d["label"] == 1),
                           lambda d: (d[S] == "xstest") & (d["label"] == 0)))
    results.append(pairing(df, "jbb harmful vs benign",
                           sel("jbb", ["harmful"]), sel("jbb", ["benign"])))
    results.append(pairing(df, "toxicchat toxic/jb vs clean",
                           sel("toxicchat", label=1), sel("toxicchat", label=0)))

    # cross-source: benign-only sets vs harmful-only sets
    for bsrc in ["falsereject", "phtest"]:
        for hname, hsel in [("salad base", sel("salad", ["base"])),
                            ("salad attack_enhanced", sel("salad", ["attack_enhanced"])),
                            ("alert plain", sel("alert", ["plain"])),
                            ("wjb vanilla_harmful", sel("wjb", ["vanilla_harmful"]))]:
            results.append(pairing(df, f"{bsrc} vs {hname}", hsel, sel(bsrc)))

    results.sort(key=lambda r: r["auc_mean"])
    out = os.path.join(RESULTS, "auc_pairings.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nranked (lowest first):")
    for r in results:
        print(f"  {r['auc_mean']:.4f}  {r['pairing']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
