"""Adversarial filtering: mine a 5000+5000 subset that a TF-IDF classifier
cannot separate, then HONESTLY re-evaluate it.

Protocol (AFLite-style, no train/select leakage):
  1. Mining pool = a large harmful+benign candidate pool.
  2. Out-of-fold scores: 5-fold CV, TF-IDF LR trained on 4 folds scores the 5th,
     so every prompt gets a score from a model that never saw it.
  3. Select the 5000 harmful with the LOWEST scores and 5000 benign with the
     HIGHEST scores (the ones that most fool the classifier).
  4. Honest re-evaluation: fresh 5-fold CV TF-IDF LR *retrained on the selected
     subset* (the retrained model can exploit any residual signal), plus a
     char 3-5gram TF-IDF LR as a second text model to check the subset is not
     just overfit to the word-level baseline.

Two mining pools:
  A: WJB adversarial_harmful vs adversarial_benign only (single-source, largest
     designed-confusable pair; per-pairing AUC 0.947).
  B: multi-source (every source capped at 15k per (source,label)).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from s2_auc import SEED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
POOL = os.path.join(POOLS, "pool.parquet")
N_SIDE = 5000


def make_vec(kind):
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)


def oof_scores(texts, labels, kind="word", n_splits=5, seed=SEED):
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    scores = np.zeros(len(texts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, te) in enumerate(skf.split(texts, labels)):
        vec = make_vec(kind)
        Xtr = vec.fit_transform(texts[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, labels[tr])
        scores[te] = clf.predict_proba(vec.transform(texts[te]))[:, 1]
        print(f"    fold {i+1}/5 done", flush=True)
    return scores


def cv_auc_kind(texts, labels, kind, n_splits=5, seed=SEED):
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(texts, labels):
        vec = make_vec(kind)
        Xtr = vec.fit_transform(texts[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, labels[tr])
        aucs.append(roc_auc_score(labels[te], clf.predict_proba(vec.transform(texts[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def run_variant(name, dfp, out_pool_path):
    print(f"\n### variant {name}: mining pool {len(dfp)} rows "
          f"({int((dfp.label==1).sum())} harmful / {int((dfp.label==0).sum())} benign)")
    print("  computing out-of-fold scores (word TF-IDF LR)...")
    dfp = dfp.reset_index(drop=True)
    scores = oof_scores(dfp["text"].tolist(), dfp["label"].tolist())
    pool_auc = roc_auc_score(dfp["label"], scores)
    print(f"  mining-pool OOF AUC: {pool_auc:.4f}")

    dfp = dfp.assign(oof=scores)
    hard_h = dfp[dfp.label == 1].nsmallest(N_SIDE, "oof")
    hard_b = dfp[dfp.label == 0].nlargest(N_SIDE, "oof")
    sel = pd.concat([hard_h, hard_b]).reset_index(drop=True)
    print(f"  selected {len(hard_h)} harmful + {len(hard_b)} benign")
    print(f"  by-construction AUC on selection (no retrain): "
          f"{roc_auc_score(sel['label'], sel['oof']):.4f}")

    res = {"variant": name, "pool_oof_auc": round(pool_auc, 4),
           "n_harmful": len(hard_h), "n_benign": len(hard_b)}
    for kind in ["word", "char"]:
        m, s = cv_auc_kind(sel["text"].tolist(), sel["label"].tolist(), kind)
        print(f"  HONEST retrained {kind}-TFIDF CV AUC on selection: {m:.4f} ±{s:.4f}")
        res[f"retrained_{kind}_auc"] = round(m, 4)
        res[f"retrained_{kind}_std"] = round(s, 4)

    # sanity: length confound check
    lh = sel[sel.label == 1]["text"].str.len()
    lb = sel[sel.label == 0]["text"].str.len()
    res["len_harmful_median"] = int(lh.median())
    res["len_benign_median"] = int(lb.median())
    print(f"  median char length harmful={lh.median():.0f} benign={lb.median():.0f}")
    print("  selection source composition:")
    comp = sel.groupby(["label", "source", "subtype"]).size()
    print(comp.to_string())
    res["composition"] = {f"{l}/{s}/{t}": int(n) for (l, s, t), n in comp.items()}

    sel[["text", "label", "source", "subtype", "oof"]].to_json(
        out_pool_path, orient="records", indent=1)
    print(f"  saved -> {out_pool_path}")
    return res


def main():
    df = pd.read_parquet(POOL)
    results = []

    # variant A: WJB adversarial pair only
    a = df[(df.source == "wjb") & (df.subtype.isin(["adversarial_harmful", "adversarial_benign"]))]
    results.append(run_variant("A_wjb_adv", a, os.path.join(POOLS, "mined_pool_A.json")))

    # variant B: multi-source, capped 15k per (source, label)
    b = (df.groupby(["source", "label"], group_keys=False)
           .apply(lambda g: g.sample(n=min(15000, len(g)), random_state=SEED)))
    results.append(run_variant("B_multisource", b, os.path.join(POOLS, "mined_pool_B.json")))

    out = os.path.join(RESULTS, "adversarial_filter_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
