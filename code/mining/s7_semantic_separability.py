"""Check whether the mined pools are only LEXICALLY confusable or also
SEMANTICALLY collapsed.

Two probes:
  1. Semantic separability: encode prompts with a sentence-embedding model
     (BAAI/bge-base-en-v1.5, CLS pooling, L2-normalized) and run the same
     5-fold CV LogisticRegression on the embeddings. If embedding AUC stays
     high while word/char TF-IDF sits near 0.57/0.60, the pool is what we
     want: lexically confusable but semantically separable. If embedding AUC
     also drops toward 0.5-0.6, harmful and benign meanings have collapsed.
  2. Cross-side nearest neighbours: for each harmful prompt, the most similar
     benign one by cosine. The top pairs are dumped for manual inspection --
     same-topic-different-intent pairs are fine, same-intent paraphrases mean
     label noise.
"""
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer

from s2_auc import SEED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
MODEL = "BAAI/bge-base-en-v1.5"
DEVICE = "cuda"
BATCH = 64


@torch.no_grad()
def encode(texts):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()
    out = []
    for i in range(0, len(texts), BATCH):
        batch = tok(texts[i:i + BATCH], padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(DEVICE)
        emb = model(**batch).last_hidden_state[:, 0]  # CLS
        out.append(torch.nn.functional.normalize(emb, dim=-1).cpu())
        if (i // BATCH) % 20 == 0:
            print(f"    encoded {i}/{len(texts)}", flush=True)
    return torch.cat(out).numpy()


def cv_auc_dense(X, y, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def run(pool_path, tag):
    pool = json.load(open(pool_path))
    texts = [r["text"] for r in pool]
    y = np.array([r["label"] for r in pool])
    print(f"\n### {tag}: {len(texts)} prompts", flush=True)
    X = encode(texts)

    m, s = cv_auc_dense(X, y)
    print(f"  embedding ({MODEL}) LR CV AUC: {m:.4f} ±{s:.4f}")

    # cross-side nearest neighbours
    Xh, Xb = X[y == 1], X[y == 0]
    th = [t for t, l in zip(texts, y) if l == 1]
    tb = [t for t, l in zip(texts, y) if l == 0]
    sim = Xh @ Xb.T
    nn_sim = sim.max(axis=1)
    nn_idx = sim.argmax(axis=1)
    print(f"  cross-side NN cosine: median={np.median(nn_sim):.3f} "
          f"p90={np.percentile(nn_sim, 90):.3f} max={nn_sim.max():.3f}")
    order = np.argsort(-nn_sim)[:30]
    pairs = [{"cosine": float(nn_sim[i]), "harmful": th[i][:400],
              "benign": tb[nn_idx[i]][:400]} for i in order]
    out = os.path.join(RESULTS, f"nn_pairs_{tag}.json")
    json.dump(pairs, open(out, "w"), indent=1)
    print(f"  top-30 most similar cross pairs -> {out}")
    return {"pool": tag, "embedding_auc": round(m, 4), "embedding_std": round(s, 4),
            "nn_cosine_median": round(float(np.median(nn_sim)), 4),
            "nn_cosine_p90": round(float(np.percentile(nn_sim, 90)), 4)}


def main():
    results = [run(os.path.join(POOLS, "mined_pool_A_rand.json"), "A"),
               run(os.path.join(POOLS, "mined_pool_B_rand.json"), "B")]
    json.dump(results, open(os.path.join(RESULTS, "semantic_separability.json"), "w"), indent=2)
    print("\nsaved -> semantic_separability.json")


if __name__ == "__main__":
    main()
