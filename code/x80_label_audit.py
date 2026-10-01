"""x80 Stage 0: paired-label lineage audit (gates G0, G1).

Read-only. Pairs the x64 refusal2 (R) and comply2 (H, amended contract:
substantive compliance, NOT judged harm) labels on the same ten x63
responses per Qwen3.5-4B prompt, verifies lineage, and reports the
response-level 2x2 table, prompt-level distributions, association, the
non-complementary fraction (G1), stochasticity counts and label-sampling
floors, separately on outer-train/inner-val/outer-test for x66 seeds
42/123/7. Selects, repairs, or relabels nothing.

Writes data/runs/x80/x80_label_manifest.json and
data/results/x80_label_audit.json. Exit 0 = G0 and G1 pass.
"""
import collections, hashlib, inspect, json, os, sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/code")
import x58_methods  # noqa: E402

TAG = "qwen"
JUDGED = {"R": (f"{ROOT}/data/judged/x64_refusal2_{TAG}_s42.jsonl", "refusal"),
          "H": (f"{ROOT}/data/judged/x64_comply2_{TAG}_s42.jsonl", "complied")}
POOL = f"{ROOT}/data/pools/cp_15k_clustered_v2.json"
PROMPTS = f"{ROOT}/data/datasets/swse_x61_majority_harmful_v1/prompts.jsonl"
SPLITS = f"{ROOT}/data/runs/x66"
SEEDS = (42, 123, 7)
OUT_M = f"{ROOT}/data/runs/x80/x80_label_manifest.json"
OUT_R = f"{ROOT}/data/results/x80_label_audit.json"
G1_MIN = 0.05


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_judged(path, key):
    rows = [json.loads(l) for l in open(path)]
    failed = sorted({r["id"] for r in rows if r.get("failed")})
    ok = collections.defaultdict(list)
    for r in rows:
        if not r.get("failed"):
            ok[r["id"]].append(r[key])
    dup = sorted(k for k, v in ok.items() if len(v) > 1)
    bad = sorted(k for k, v in ok.items() if any(len(x) != 10 or not all(isinstance(b, bool) for b in x) for x in v))
    lab = {k: np.array(v[0], dtype=np.int8) for k, v in ok.items()}
    info = {"file": os.path.relpath(path, ROOT), "sha256": sha(path), "field": key,
            "rows_total": len(rows), "failed_attempt_rows": len(rows) - sum(len(v) for v in ok.values()),
            "failed_ids_later_rerun": failed, "unique_ids": len(ok), "duplicate_ids": dup, "malformed_ids": bad}
    return lab, info


def floor_binomial(counts):
    """E|X/10 - p_hat| under Binomial(10, p_hat) per prompt, averaged (exact)."""
    from scipy.stats import binom
    out = []
    for c in counts:
        p = c / 10.0
        k = np.arange(11)
        out.append(float(np.sum(binom.pmf(k, 10, p) * np.abs(k / 10.0 - p))))
    return float(np.mean(out))


def audit_rows(R, H, name):
    r, h = R.mean(1), H.mean(1)
    cells = {"(R=1,H=0) safe_refusal": int(((R == 1) & (H == 0)).sum()),
             "(R=1,H=1) refuse_then_comply": int(((R == 1) & (H == 1)).sum()),
             "(R=0,H=1) compliant_nonrefusal": int(((R == 0) & (H == 1)).sum()),
             "(R=0,H=0) neither": int(((R == 0) & (H == 0)).sum())}
    n_resp = R.size
    noncomp = (cells["(R=1,H=1) refuse_then_comply"] + cells["(R=0,H=0) neither"]) / n_resp
    sr, sh = R.std(1) > 0, H.std(1) > 0
    return {"name": name, "prompts": int(len(R)), "responses": int(n_resp),
            "cells": cells, "cell_fractions": {k: v / n_resp for k, v in cells.items()},
            "noncomplementary_fraction": noncomp,
            "mean_r": float(r.mean()), "mean_h": float(h.mean()),
            "r_hist": np.bincount(R.sum(1), minlength=11).tolist(),
            "h_hist": np.bincount(H.sum(1), minlength=11).tolist(),
            "r_plus_h_hist": np.bincount((R.sum(1) + H.sum(1)), minlength=21).tolist(),
            "pearson_r_h": float(pearsonr(r, h)[0]), "spearman_r_h": float(spearmanr(r, h)[0]),
            "stochastic": {"R_only": int((sr & ~sh).sum()), "H_only": int((~sr & sh).sum()),
                           "both": int((sr & sh).sum()), "neither": int((~sr & ~sh).sum())},
            "label_floor_mae_r": floor_binomial(R.sum(1)), "label_floor_mae_h": floor_binomial(H.sum(1))}


def main():
    frozen = sorted(int(json.loads(l)["source_id"]) for l in open(PROMPTS))
    pool_ids = sorted(r["id"] for r in json.load(open(POOL)) if r["source"] == "toxic")
    assert len(pool_ids) == 15000
    row_of_id = {i: k for k, i in enumerate(pool_ids)}  # positional contract: harmful rows sorted by id
    labs, infos = {}, {}
    for k, (p, key) in JUDGED.items():
        labs[k], infos[k] = load_judged(p, key)
        infos[k]["ids_match_frozen_13527"] = set(labs[k]) == set(frozen)
    for k in labs:
        assert infos[k]["ids_match_frozen_13527"] and not infos[k]["malformed_ids"]
    # response-order alignment: both methods judged the same x63 row's llm_responses in index order
    rub = {"refusal2": hashlib.sha256(inspect.getsource(x58_methods.prompt_refusal2).encode()).hexdigest(),
           "comply2": hashlib.sha256(inspect.getsource(x58_methods.prompt_comply2).encode()).hexdigest()}
    R = np.stack([labs["R"][i] for i in frozen]); H = np.stack([labs["H"][i] for i in frozen])
    g0 = True
    manifest = {"experiment": "x80", "stage": 0, "model": "Qwen/Qwen3.5-4B", "tag": TAG,
                "R": dict(infos["R"], construct="explicit refusal (x58 refusal2), y_R = refusal_count/10 = 1 - v6"),
                "H": dict(infos["H"], construct="substantive compliance (x58 comply2) -- NOT judged harm; amended contract 2026-08-21"),
                "rubric_source_sha256": rub,
                "judge": "deepseek/deepseek-v4-flash-0731, effort low, seed 42",
                "responses": "x63 regeneration, 10 per prompt, index order shared by both methods (x64_run.py judges row['llm_responses'] verbatim)",
                "prompt_order": "frozen 13,527 source_ids ascending; cache row = rank among the 15,000 toxic pool ids",
                "G0": g0}
    json.dump(manifest, open(OUT_M, "w"), indent=1)

    per_split = {}
    src_full = np.zeros(30000, dtype=np.int8); src_full[:15000] = 1
    rows_frozen = np.array([row_of_id[i] for i in frozen])
    pos = {r: k for k, r in enumerate(rows_frozen)}
    for seed in SEEDS:
        tr = np.array(json.load(open(f"{SPLITS}/{TAG}_seed{seed}_train_indices.json")))
        te = np.array(json.load(open(f"{SPLITS}/{TAG}_seed{seed}_test_indices.json")))
        itr, iva = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=42, stratify=src_full[tr])
        parts = {"outer_train_all": tr, "inner_train": tr[itr], "inner_val": tr[iva], "outer_test": te}
        per_split[seed] = {}
        for name, idx in parts.items():
            hrows = [pos[r] for r in idx if r < 15000]
            assert all(r in pos for r in idx if r < 15000), "split contains a non-retained harmful row"
            per_split[seed][name] = audit_rows(R[hrows], H[hrows], name)
    full = audit_rows(R, H, "all_13527")
    g1 = min(per_split[s]["outer_train_all"]["noncomplementary_fraction"] for s in SEEDS) >= G1_MIN
    res = {"G0": g0, "G1": g1, "G1_threshold": G1_MIN, "all": full, "by_seed": per_split,
           "H_caveat": "H is comply2 substantive compliance, not judged harm"}
    json.dump(res, open(OUT_R, "w"), indent=1)
    s42 = per_split[42]["outer_train_all"]
    print(json.dumps({"G0": g0, "G1": g1, "all_cells": full["cells"], "all_noncomp": round(full["noncomplementary_fraction"], 4),
                      "pearson": round(full["pearson_r_h"], 4), "spearman": round(full["spearman_r_h"], 4),
                      "stochastic": full["stochastic"], "floors": [round(full["label_floor_mae_r"], 4), round(full["label_floor_mae_h"], 4)],
                      "train42_noncomp": round(s42["noncomplementary_fraction"], 4)}, indent=1))
    sys.exit(0 if (g0 and g1) else 3)


if __name__ == "__main__":
    main()
