"""x81 Stage 1: discover stable refusal-associated words on x66 train only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from transformers import AutoTokenizer

from x81_common import (
    MODEL_ID,
    N_RESPONSES,
    RUN,
    TEST_INDICES_PATH,
    TRAIN_INDICES_PATH,
    atomic_json,
    read_json,
    load_historical_refusals,
    load_split_rows,
    tokenizer_span_ids,
    word_spans,
)

CS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
MIN_DF = 10
N_FOLDS = 5
MAX_SENSITIVE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--audit", default=str(RUN / "x81_audit.json"))
    parser.add_argument("--out", default=str(RUN / "x81_lexicon.json"))
    return parser.parse_args()


def expanded_binomial(X, counts: np.ndarray, indices: np.ndarray):
    selected = X[indices]
    weights = np.concatenate((counts[indices], N_RESPONSES - counts[indices])).astype(float)
    labels = np.concatenate((np.ones(len(indices), dtype=int), np.zeros(len(indices), dtype=int)))
    matrix = sparse.vstack((selected, selected), format="csr")
    keep = weights > 0
    return matrix[keep], labels[keep], weights[keep]


def fit_model(X, counts: np.ndarray, indices: np.ndarray, c_value: float) -> LogisticRegression:
    matrix, labels, weights = expanded_binomial(X, counts, indices)
    model = LogisticRegression(
        penalty="l1",
        C=float(c_value),
        solver="liblinear",
        max_iter=5000,
        random_state=42,
    )
    model.fit(matrix, labels, sample_weight=weights)
    return model


def binomial_deviance(counts: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.clip(probabilities.astype(float), 1e-9, 1 - 1e-9)
    loss = -(counts * np.log(p) + (N_RESPONSES - counts) * np.log1p(-p))
    return float(loss.sum() / (N_RESPONSES * len(counts)))


def choose_c_and_fold_coefs(X, counts: np.ndarray):
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=42).split(np.arange(X.shape[0])))
    grid = []
    for c_value in CS:
        losses = []
        for train_index, validation_index in folds:
            model = fit_model(X, counts, train_index, c_value)
            probabilities = model.predict_proba(X[validation_index])[:, 1]
            losses.append(binomial_deviance(counts[validation_index], probabilities))
        grid.append({"C": c_value, "fold_deviance": losses, "mean_deviance": float(np.mean(losses))})
    best = min(grid, key=lambda row: (row["mean_deviance"], row["C"]))
    fold_coefs = []
    for train_index, _ in folds:
        fold_coefs.append(fit_model(X, counts, train_index, best["C"]).coef_[0].astype(float))
    full = fit_model(X, counts, np.arange(X.shape[0]), best["C"])
    return grid, full, np.stack(fold_coefs)


def main() -> None:
    args = parse_args()
    if not Path(args.audit).exists():
        raise SystemExit("FATAL: run x81_audit.py successfully first")
    audit = read_json(args.audit)
    if args.model != MODEL_ID or audit.get("model_id") != MODEL_ID or audit.get("status") != "pass":
        raise SystemExit(f"FATAL: x81 is frozen to an audited {MODEL_ID} snapshot")
    model_source = audit["model_snapshot"]["snapshot"]

    discovery_rows = load_split_rows(TRAIN_INDICES_PATH)
    evaluation_rows = load_split_rows(TEST_INDICES_PATH)
    discovery_ids = [int(row["id"]) for row in discovery_rows]
    refusals = load_historical_refusals(discovery_ids)
    counts = np.asarray([sum(refusals[prompt_id]) for prompt_id in discovery_ids], dtype=float)
    texts = [str(row["prompt"]) for row in discovery_rows]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 1),
        min_df=MIN_DF,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    document_frequency = np.asarray((X > 0).sum(axis=0)).ravel().astype(int)
    presence = (X > 0).astype(np.float64)
    rates = counts / N_RESPONSES
    sum_r_present = np.asarray(presence.T @ rates).ravel()
    total_r = float(rates.sum())
    grid, full_model, fold_coefs = choose_c_and_fold_coefs(X, counts)
    full_coefs = full_model.coef_[0].astype(float)

    feature_rows = []
    for index, name in enumerate(names):
        n_present = int(document_frequency[index])
        n_absent = len(rates) - n_present
        mean_present = float(sum_r_present[index] / n_present)
        mean_absent = float((total_r - sum_r_present[index]) / n_absent) if n_absent else None
        feature_rows.append(
            {
                "word": str(name),
                "document_frequency": n_present,
                "full_coef": float(full_coefs[index]),
                "fold_coefs": [float(value) for value in fold_coefs[:, index]],
                "mean_r_present": mean_present,
                "mean_r_absent": mean_absent,
                "univariate_delta_r": None if mean_absent is None else mean_present - mean_absent,
            }
        )

    stable = [
        row
        for row in feature_rows
        if row["full_coef"] > 0
        and all(value > 0 for value in row["fold_coefs"])
        and row["word"] not in ENGLISH_STOP_WORDS
        and not row["word"].isdigit()
    ]
    stable.sort(key=lambda row: (-row["full_coef"], row["word"]))
    preliminary = stable[:MAX_SENSITIVE]

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
    selected_targets = []
    preliminary_by_word = {row["word"]: row for row in preliminary}
    for row in evaluation_rows:
        prompt = str(row["prompt"])
        eligible = []
        occurring = []
        for word_row in preliminary:
            word = word_row["word"]
            spans = word_spans(prompt, word)
            if not spans:
                continue
            occurring.append(word)
            if len(spans) != 1:
                continue
            start, end = spans[0]
            try:
                token_ids = tokenizer_span_ids(tokenizer, prompt, start, end)
            except (NotImplementedError, ValueError):
                raise SystemExit("FATAL: Qwen tokenizer does not provide usable offsets")
            if len(token_ids) != 1:
                continue
            if token_ids[0] in tokenizer.all_special_ids:
                continue
            eligible.append((-float(word_row["full_coef"]), start, word, end, token_ids[0]))
        if not eligible:
            continue
        _, start, word, end, token_id = min(eligible)
        selected_targets.append(
            {
                "id": int(row["id"]),
                "prompt": prompt,
                "target_word": prompt[start:end],
                "target_word_lower": word,
                "start": int(start),
                "end": int(end),
                "target_token_id": int(token_id),
                "discovery_coef": preliminary_by_word[word]["full_coef"],
                "other_sensitive_words": [value for value in occurring if value != word],
            }
        )

    sensitive = preliminary

    output = {
        "experiment": "x81",
        "stage": 1,
        "model_id": args.model,
        "model_source": model_source,
        "model_manifest_sha256": audit["model_snapshot"]["manifest_sha256"],
        "uses_evaluation_refusal_labels": False,
        "config": {
            "features": "lowercase word-unigram TF-IDF",
            "min_df": MIN_DF,
            "sublinear_tf": True,
            "penalty": "l1",
            "solver": "liblinear",
            "C_grid": list(CS),
            "n_folds": N_FOLDS,
            "max_sensitive": MAX_SENSITIVE,
        },
        "counts": {
            "discovery_prompts": len(discovery_rows),
            "evaluation_prompts": len(evaluation_rows),
            "vocabulary": len(names),
            "stable_positive": len(stable),
            "sensitive_words": len(sensitive),
            "target_words_observed_token_eligible": len(
                {row["target_word_lower"] for row in selected_targets}
            ),
            "evaluation_targets": len(selected_targets),
        },
        "selected_C": float(full_model.C),
        "C_grid_results": grid,
        "intercept": float(full_model.intercept_[0]),
        "features": feature_rows,
        "sensitive_words": sensitive,
        "targets": selected_targets,
    }
    atomic_json(args.out, output)
    print(
        f"x81 lexicon: vocab={len(names)} stable={len(stable)} "
        f"sensitive={len(sensitive)} targets={len(selected_targets)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
