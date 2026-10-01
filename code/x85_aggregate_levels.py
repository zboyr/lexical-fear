"""x85 supplement: prompt- and word-level aggregation of the latent/behavior link.

Descriptive companion to x85_analyze.py. It reads the same frozen shards and
judgments and reports (a) per-word and per-prompt mean delta_refusal with the
matching ANOVA component projections, (b) per-category and per-stratum
behavior means, and (c) a per-word sanity check on the neutral control. No
gate is computed here; the gates live in x85_analyze.py.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from x85_analyze import (
    behavior_arrays,
    cell_index_grids,
    load_selected_features,
    parse_int_list,
    parse_position_list,
)
from x85_common import PRIMARY_STATES, RESULTS, RUN, atomic_json, cohort_path, read_json, vector_anova


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=str(cohort_path()))
    parser.add_argument("--features-glob", default=str(RUN / "features" / "x85_hidden_shard_*.pt"))
    parser.add_argument("--run-dir", default=str(RUN))
    parser.add_argument("--states", default=",".join(str(value) for value in PRIMARY_STATES))
    parser.add_argument("--positions", default="tbg_minus1,tbg")
    parser.add_argument("--out", default=str(RESULTS / "x85_crossed_keyword_levels.json"))
    return parser.parse_args()


def rho(a: np.ndarray, b: np.ndarray) -> dict:
    result = spearmanr(a, b)
    return {"n": int(len(a)), "rho": float(result.correlation), "p": float(result.pvalue)}


def main() -> None:
    args = parse_args()
    states = parse_int_list(args.states)
    positions = parse_position_list(args.positions)
    manifest = read_json(Path(args.cohort))
    behavior = behavior_arrays(manifest, Path(args.run_dir))
    if behavior is None:
        raise SystemExit("FATAL: behavior judgments incomplete")
    feature_paths = [Path(value) for value in sorted(glob.glob(args.features_glob))]
    features, _ = load_selected_features(feature_paths, manifest, states, positions)
    harm_grid, neutral_grid = cell_index_grids(manifest)
    words = manifest["words"]
    prompts = manifest["prompts"]
    n_prompts, n_words = len(prompts), len(words)

    # Behavior aggregates at each level (sparse graph: 6 words/prompt, 24 prompts/word).
    by_word = defaultdict(list)
    by_prompt = defaultdict(list)
    for row in behavior["rows"]:
        by_word[row["word_index"]].append(row)
        by_prompt[row["prompt_index"]].append(row)
    word_delta = np.asarray([np.mean([r["delta_H_minus_N"] for r in by_word[w]]) for w in range(n_words)])
    word_rH = np.asarray([np.mean([r["r_H"] for r in by_word[w]]) for w in range(n_words)])
    word_rN = np.asarray([np.mean([r["r_N"] for r in by_word[w]]) for w in range(n_words)])
    word_rO = np.asarray([np.mean([r["r_O"] for r in by_word[w]]) for w in range(n_words)])
    prompt_delta = np.asarray([np.mean([r["delta_H_minus_N"] for r in by_prompt[p]]) for p in range(n_prompts)])
    prompt_rH = np.asarray([np.mean([r["r_H"] for r in by_prompt[p]]) for p in range(n_prompts)])
    prompt_rN = np.asarray([np.mean([r["r_N"] for r in by_prompt[p]]) for p in range(n_prompts)])
    prompt_rO = np.asarray([np.mean([r["r_O"] for r in by_prompt[p]]) for p in range(n_prompts)])

    levels = {}
    for position_offset, position in enumerate(positions):
        levels[position] = {}
        for state_offset, state in enumerate(states):
            deltas = (
                features[harm_grid, position_offset, state_offset]
                - features[neutral_grid, position_offset, state_offset]
            )
            parts = vector_anova(deltas)
            mu = parts["grand"]
            q = mu / np.linalg.norm(mu)
            word_proj = (mu + parts["word"]) @ q  # word-mean projection (dense, 96 prompts)
            prompt_proj = (mu + parts["prompt"]) @ q  # prompt-mean projection (dense, 24 words)
            word_norm = np.linalg.norm(parts["word"], axis=1)
            prompt_norm = np.linalg.norm(parts["prompt"], axis=1)
            levels[position][str(state)] = {
                "word_level": {
                    "projection_vs_delta": rho(word_proj, word_delta),
                    "effect_norm_vs_delta": rho(word_norm, word_delta),
                    "projection_vs_r_H": rho(word_proj, word_rH),
                },
                "prompt_level": {
                    "projection_vs_delta": rho(prompt_proj, prompt_delta),
                    "effect_norm_vs_delta": rho(prompt_norm, prompt_delta),
                    "projection_vs_r_O": rho(prompt_proj, prompt_rO),
                    "historical_rate_vs_delta": rho(
                        np.asarray([row["historical_refusal_rate"] for row in prompts]), prompt_delta
                    ),
                },
                "per_word_projection": [float(value) for value in word_proj],
                "per_prompt_projection": [float(value) for value in prompt_proj],
            }

    word_table = [
        {
            "word_index": w,
            "category": words[w]["category"],
            "harm": words[w]["harm"],
            "neutral": words[w]["neutral"],
            "n_prompts": len(by_word[w]),
            "mean_r_O": float(word_rO[w]),
            "mean_r_N": float(word_rN[w]),
            "mean_r_H": float(word_rH[w]),
            "delta_H_minus_N": float(word_delta[w]),
        }
        for w in range(n_words)
    ]
    category_table = {}
    for category in dict.fromkeys(row["category"] for row in words):
        rows = [r for w, rs in by_word.items() if words[w]["category"] == category for r in rs]
        category_table[category] = {
            "n_edges": len(rows),
            "delta_H_minus_N": float(np.mean([r["delta_H_minus_N"] for r in rows])),
            "delta_H_minus_O": float(np.mean([r["delta_H_minus_O"] for r in rows])),
        }
    stratum_table = {}
    for stratum in ("floor", "low", "high"):
        idx = [p for p in range(n_prompts) if prompts[p]["stratum"] == stratum]
        rows = [r for p in idx for r in by_prompt[p]]
        stratum_table[stratum] = {
            "n_prompts": len(idx),
            "mean_r_O": float(np.mean(prompt_rO[idx])),
            "mean_r_N": float(np.mean(prompt_rN[idx])),
            "mean_r_H": float(np.mean(prompt_rH[idx])),
            "delta_H_minus_N": float(np.mean([r["delta_H_minus_N"] for r in rows])),
            "prompts_with_r_O_zero_pushed_to_majority_H": int(
                sum(1 for p in idx if prompt_rO[p] == 0 and prompt_rH[p] >= 0.5)
            ),
            "prompts_with_r_O_zero": int(sum(1 for p in idx if prompt_rO[p] == 0)),
        }
    edge_delta = behavior["delta"]
    output = {
        "experiment": "x85",
        "kind": "descriptive_supplement",
        "note": "No gate here; gates are in x85_crossed_keyword_latent.json. Word level pools 24 prompts per word, prompt level pools 6 words per prompt (behavior graph), while projections use the dense 96 x 24 grid.",
        "edge_summary": {
            "n_edges": int(len(edge_delta)),
            "delta_H_minus_N_quantiles": {
                str(qv): float(np.quantile(edge_delta, qv)) for qv in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "fraction_edges_delta_positive": float((edge_delta > 0).mean()),
            "fraction_edges_delta_negative": float((edge_delta < 0).mean()),
        },
        "levels": levels,
        "words": word_table,
        "categories": category_table,
        "strata": stratum_table,
    }
    atomic_json(Path(args.out), output)
    print(json.dumps({"edge_summary": output["edge_summary"], "strata": stratum_table}, indent=1))
    for position in positions:
        for state in states:
            entry = levels[position][str(state)]
            print(
                position, state,
                "word proj~delta rho=%.2f" % entry["word_level"]["projection_vs_delta"]["rho"],
                "prompt proj~delta rho=%.2f" % entry["prompt_level"]["projection_vs_delta"]["rho"],
                "prompt hist~delta rho=%.2f" % entry["prompt_level"]["historical_rate_vs_delta"]["rho"],
            )
    for row in sorted(word_table, key=lambda r: r["delta_H_minus_N"]):
        print("%-10s %-20s rO=%.2f rN=%.2f rH=%.2f d=%+.2f" % (row["harm"], row["category"], row["mean_r_O"], row["mean_r_N"], row["mean_r_H"], row["delta_H_minus_N"]))
    for category, row in category_table.items():
        print("%-22s n=%d dHN=%+.3f dHO=%+.3f" % (category, row["n_edges"], row["delta_H_minus_N"], row["delta_H_minus_O"]))


if __name__ == "__main__":
    main()
