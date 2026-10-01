"""Stage 5: evaluate all predictors on the held-out confusable_A_test cache and
emit the summary table (detection AUC vs risk T = 1[T>0.005], and vs source).

Predictors:
  - MLP probe    best-layer / concat   (official train_probes.py, regress T)
  - Linear probe best-layer / concat   (official Ridge, regress T)
  - Embedding bag (MLP / Linear)        (mean-pooled layer-0 token embeddings)
  - TF-IDF + LogisticRegression         (raw prompt text)

Reuses orbench s6/s7/s8 logic. Run AFTER t4_split + train_probes.py on the train cache.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_probes import MLPProbe, predict_mlp, train_mlp_probe  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
ALPHA, THR = 0.7, 0.005
MODEL = "unsloth/Llama-3.2-3B-Instruct"


def load_labels(cache):
    entropy = np.load(f"{cache}/safety_entropy_labels_tbg.npy")
    score = np.load(f"{cache}/safety_score_labels_tbg.npy")
    source = np.load(f"{cache}/source_labels_tbg.npy")
    T = (ALPHA * score + (1 - ALPHA) * entropy).astype(np.float32)
    return T, (T > THR).astype(int), source


def mlp_pred(model, X):
    model = model.cpu().eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X)).numpy().ravel()


def official_probe_rows(run, test_cache):
    tbg = torch.load(f"{test_cache}/hidden_states.pt", weights_only=False)["tbg_states"]
    rows = []
    mlp = pickle.load(open(f"{run}/mlp/safety_probe.pkl", "rb"))
    cc = mlp["concatenated_layer_indices"]
    concat = np.concatenate([tbg[i].float().numpy() for i in cc], axis=1)
    bl = mlp["best_layer"]
    rows.append((f"MLP probe best-layer ({bl})", mlp_pred(mlp["all_models"][bl], tbg[bl].float().numpy())))
    rows.append(("MLP probe concat", mlp_pred(mlp["concatenated_model"], concat)))

    lin = pickle.load(open(f"{run}/linear/safety_probe.pkl", "rb"))
    bl = lin["best_layer"]
    rows.append((f"Linear probe best-layer ({bl})",
                 np.clip(lin["all_models"][bl].predict(tbg[bl].float().numpy()), 0, 1)))
    rows.append(("Linear probe concat",
                 np.clip(lin["concatenated_model"].predict(
                     np.concatenate([tbg[i].float().numpy() for i in lin["concatenated_layer_indices"]], axis=1)), 0, 1)))
    return rows


def embedding_bag_rows(full_cache, T, source, test_size=0.3):
    """Mean-pooled layer-0 token embeddings -> MLP + Ridge, same split as t4."""
    prompts = json.load(open(f"{full_cache}/prompts.json"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True).eval()
    emb = model.get_input_embeddings()
    feats = np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            if ids.shape[-1] == 0:
                ids = tok(p if p else " ", return_tensors="pt").input_ids.to(model.device)
            feats[i] = emb(ids)[0].float().mean(0).cpu().numpy()

    tr, te = train_test_split(np.arange(len(prompts)), test_size=test_size, random_state=42, stratify=source)
    tr2, va = train_test_split(tr, test_size=0.2, random_state=42)
    mlp = train_mlp_probe(feats[tr2], T[tr2], feats[va], T[va], hidden_dim=128, activation="silu",
                          dropout=0.4, learning_rate=5e-4, num_epochs=100, batch_size=128,
                          patience=20, device="cuda")
    s_mlp = predict_mlp(mlp, feats[te], batch_size=128, device="cuda").ravel()
    rid = Ridge(alpha=1.0).fit(feats[tr], T[tr])
    s_lin = np.clip(rid.predict(feats[te]), 0, 1)
    return [("Embedding bag (MLP)", s_mlp), ("Embedding bag (Linear)", s_lin)], te


def tfidf_row(full_cache, T, source, test_size=0.3):
    prompts = json.load(open(f"{full_cache}/prompts.json"))
    risk = (T > THR).astype(int)
    tr, te = train_test_split(np.arange(len(prompts)), test_size=test_size, random_state=42, stratify=source)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([prompts[i] for i in tr])
    Xte = vec.transform([prompts[i] for i in te])
    g = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, risk[tr])
    return [("TF-IDF", g.predict_proba(Xte)[:, 1])], te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<dir> with mlp/ and linear/ probes")
    ap.add_argument("--full_cache", default="feature_caches/confusable_A_full")
    ap.add_argument("--test_cache", default="feature_caches/confusable_A_test")
    ap.add_argument("--test_size", type=float, default=0.3, help="must match the t4 split")
    ap.add_argument("--out", default=os.path.join(RESULTS, "confusable_A_auc.json"))
    args = ap.parse_args()

    T_full, risk_full, source_full = load_labels(args.full_cache)
    T_test, risk_test, source_test = load_labels(args.test_cache)

    rows = official_probe_rows(args.run, args.test_cache)
    scored = [(n, roc_auc_score(source_test, s), roc_auc_score(risk_test, s), float(np.std(s)))
              for n, s in rows]

    eb_rows, te = embedding_bag_rows(args.full_cache, T_full, source_full, args.test_size)
    tf_rows, _ = tfidf_row(args.full_cache, T_full, source_full, args.test_size)
    for n, s in eb_rows + tf_rows:
        scored.append((n, roc_auc_score(source_full[te], s), roc_auc_score(risk_full[te], s), float(np.std(s))))

    print(f"\n{'predictor':32s}{'AUC vs source':>15s}{'AUC vs risk(T)':>16s}{'score std':>11s}")
    print("-" * 74)
    out = []
    for n, a_src, a_risk, std in scored:
        print(f"{n:32s}{a_src:>15.4f}{a_risk:>16.4f}{std:>11.4f}")
        out.append({"predictor": n, "auc_source": a_src, "auc_risk": a_risk, "score_std": std})
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
