"""x66: build the v4/v5 label-overlay caches for one model tag.

Creates {tag}_A15k_v4_full and {tag}_A15k_v5_full next to the model's
v2-style full cache. Features (hidden_states.pt, tbg_logits.npy,
prompts.json, source_labels_tbg.npy, probt when present) are symlinked;
only the label npys are real files, in the v2/v3 file convention so
probes_optimal_v2.py --t_variant v2 reproduces each construct verbatim:

  v4: safety_score  = 1 - refusal_rate
      safety_entropy = H2(refusal_rate)
      => 0.7*s + 0.3*e = T_v4 (x65 formula)
  v5: safety_score  = p_hc + 0.5*p_dh
      safety_entropy = 3-class entropy / log 3
      => 0.7*s + 0.3*e = T_v5 (the historical construct, mapping B)

Row order: the 30k cache rows are the 15k harmful prompts of
cp_15k_clustered_v2.json (row r <-> its "id" field) followed by 15k benign.
Benign rows get 0 everywhere (never judged, v2 convention). The 1,473
harmful rows outside the x61-retained 13,527 get 0 in the label arrays and
False in x61_mask_tbg.npy; they must be DROPPED via the mask before any
split or training -- they have no v4/v5 judgment. The v4 overlay additionally
carries the v5 arrays as sidecars (v5_score/v5_entropy_labels_tbg.npy) so
one feature split can serve both trainings.

Fail-closed: requires the x65 label jsons with "complete": true.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# X66_* env overrides let the same script run on a cluster, where the feature
# caches and label jsons may live outside the repository.
FC = os.environ.get("SWSE_FC", os.environ.get("X66_FC", os.path.join(ROOT, "feature_caches")))
DATA = os.environ.get("X66_DATA", os.path.join(ROOT, "data"))
CLUSTERED = os.environ.get(
    "X66_CLUSTERED",
    os.path.join(ROOT, "data", "pools", "cp_15k_clustered_v2.json"))

SRC_CACHE = {"qwen": "qwen35_4b_A15k_v2_full", "llama": "llama32_A15k_v2_full",
             "smollm3": "smollm3_A15k_v2style_full", "gemma": "gemma_A15k_v2style_full",
             "phi": "phi_A15k_v2style_full",
             "qwen08b": "qwen08b_A15k_v2style_full", "qwen2b": "qwen2b_A15k_v2style_full",
             "qwen9b": "qwen9b_A15k_v2style_full", "qwen27b": "qwen27b_A15k_v2style_full"}
SYMLINK_FILES = ("hidden_states.pt", "tbg_logits.npy", "prompts.json",
                 "source_labels_tbg.npy", "probt_labels_tbg.npy")
N_ROWS, N_HARM = 30000, 15000


def build_arrays(tag):
    order = [r["id"] for r in json.load(open(CLUSTERED))]
    assert len(order) == N_HARM
    row_of = {int(i): r for r, i in enumerate(order)}

    out = {}
    for ver in ("v4", "v5"):
        path = os.path.join(DATA, "labels", f"{tag}_labels_{ver}.json")
        if not os.path.exists(path):
            sys.exit(f"FATAL: missing {path}")
        obj = json.load(open(path))
        if not obj.get("complete"):
            sys.exit(f"FATAL: {path} is not a complete build")
        s = np.zeros(N_ROWS, dtype=np.float32)
        e = np.zeros(N_ROWS, dtype=np.float32)
        mask = np.zeros(N_ROWS, dtype=bool)
        mask[N_HARM:] = True  # benign rows always kept, T=0
        for pid, row in obj["rows"].items():
            r = row_of[int(pid)]
            if ver == "v4":
                s[r] = 1.0 - row["refusal_rate"]
                e[r] = row["refusal_entropy"]
            else:
                s[r] = row["safety_score"]
                e[r] = row["safety_entropy"]
            mask[r] = True
        n_lab = int(mask[:N_HARM].sum())
        if n_lab != len(obj["rows"]):
            sys.exit(f"FATAL: {ver} mapped {n_lab} rows for {len(obj['rows'])} labels")
        out[ver] = (s, e, mask)
        T = 0.7 * s + 0.3 * e
        print(f"{tag} {ver}: labeled harmful rows {n_lab}, "
              f"T mean(within-labeled) {T[mask & (np.arange(N_ROWS) < N_HARM)].mean():.4f}, "
              f"zeros {(T[mask & (np.arange(N_ROWS) < N_HARM)] == 0).mean()*100:.1f}%")
    if not np.array_equal(out["v4"][2], out["v5"][2]):
        sys.exit("FATAL: v4/v5 row masks differ")
    return out


def make_overlay(tag, ver, s, e, mask, extra):
    src = os.path.join(FC, SRC_CACHE[tag])
    dst = os.path.join(FC, f"{tag}_A15k_{ver}_full")
    os.makedirs(dst, exist_ok=True)
    for name in SYMLINK_FILES:
        sp, lp = os.path.join(src, name), os.path.join(dst, name)
        if not os.path.exists(sp):
            continue  # probt exists only for qwen
        if os.path.islink(lp) or os.path.exists(lp):
            os.remove(lp)
        os.symlink(sp, lp)
    np.save(os.path.join(dst, "safety_score_labels_tbg.npy"), s)
    np.save(os.path.join(dst, "safety_entropy_labels_tbg.npy"), e)
    np.save(os.path.join(dst, "joint_risk_target_labels_tbg.npy"),
            (0.7 * s + 0.3 * e).astype(np.float32))
    np.save(os.path.join(dst, "x61_mask_tbg.npy"), mask)
    for name, arr in extra.items():
        np.save(os.path.join(dst, name), arr)
    print(f"wrote {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(SRC_CACHE))
    args = ap.parse_args()
    arrs = build_arrays(args.tag)
    s4, e4, mask = arrs["v4"]
    s5, e5, _ = arrs["v5"]
    make_overlay(args.tag, "v4", s4, e4, mask,
                 {"v5_score_labels_tbg.npy": s5, "v5_entropy_labels_tbg.npy": e5})
    make_overlay(args.tag, "v5", s5, e5, mask, {})


if __name__ == "__main__":
    main()
