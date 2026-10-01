#!/usr/bin/env python
"""Build the x61 collapsed-majority harmful prompt dataset.

Rule (user decision 2026-08-15):
  1. Collapse each judge's six-way x61 code: P,X -> PX ; A,C,O -> ACO ; N -> N.
  2. Drop prompts where the three judges' collapsed labels are all different
     (i.e. exactly {N, PX, ACO}).
  3. Drop prompts whose collapsed majority (>=2 of 3) label is PX.
  4. Every remaining prompt (collapsed majority N or ACO) is retained; the
     retained set is the new harmful prompt dataset.

Inputs are the legacy 15,000-prompt harmful source and the three validated
x61 judge outputs (Luna / DeepSeek / GLM). The script fails closed on any
hash, coverage, or duplicate-id problem. It does NOT touch
datasets/CURRENT.json.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "pools"

SOURCE_JSON = DATA / "qwen_clustered_v2.json"
SOURCE_SHA256 = "3f2f70a50a5d9e8a9a7b7844b27ab640023e6315609fc97137bfe0232a294905"

JUDGE_FILES = {
    "luna": DATA / "x61_sixway_gpt56_luna_max.jsonl",
    "deepseek": DATA / "x61_sixway_deepseek_v4_0731_max.jsonl",
    "glm": DATA / "x61_sixway_glm_5_2_low.jsonl",
}

V1_DIR = REPO / "data" / "datasets" / "swse_feasibility_clean_b075_v1"

COLLAPSE = {"N": "N", "P": "PX", "X": "PX", "A": "ACO", "C": "ACO", "O": "ACO"}

DATASET_NAME = "swse_x61_majority_harmful_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_judge(path: Path, source_by_id: dict) -> dict:
    codes = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pid = r["id"]
            if pid in codes:
                sys.exit(f"FATAL: duplicate id {pid} in {path}")
            if pid not in source_by_id:
                sys.exit(f"FATAL: id {pid} in {path} not in source")
            if r["code"] not in COLLAPSE:
                sys.exit(f"FATAL: invalid code {r['code']!r} for id {pid} in {path}")
            expect = hashlib.sha256(
                source_by_id[pid]["prompt"].encode("utf-8")
            ).hexdigest()
            if r["prompt_sha256"] != expect:
                sys.exit(f"FATAL: prompt hash mismatch for id {pid} in {path}")
            codes[pid] = r["code"]
    if len(codes) != len(source_by_id):
        sys.exit(f"FATAL: {path} has {len(codes)} rows, expected {len(source_by_id)}")
    return codes


def old_feasibility_status(v1_dir: Path) -> dict:
    """source_id -> 'feasible' | 'infeasible' (unreviewed ids are absent)."""
    status = {}
    with open(v1_dir / "feasibility_decisions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            status[r["source_id"]] = (
                "feasible" if r["feasible_from_prompt_alone"] else "infeasible"
            )
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "datasets" / DATASET_NAME)
    ap.add_argument("--force", action="store_true", help="allow non-empty --out")
    args = ap.parse_args()

    if sha256_file(SOURCE_JSON) != SOURCE_SHA256:
        sys.exit("FATAL: source prompt file hash mismatch")
    source = json.load(open(SOURCE_JSON))
    source_by_id = {r["id"]: r for r in source}
    if len(source_by_id) != 15000:
        sys.exit("FATAL: source does not contain 15,000 unique ids")

    judge_codes = {name: load_judge(path, source_by_id) for name, path in JUDGE_FILES.items()}
    old_status = old_feasibility_status(V1_DIR)
    unknown_old = set(old_status) - set(source_by_id)
    if unknown_old:
        sys.exit(f"FATAL: {len(unknown_old)} feasibility source_ids not in source")

    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        sys.exit(f"FATAL: {args.out} is not empty; pass --force to overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    retained, excluded = [], []
    decision_by_id = {}
    for row_idx, r in enumerate(source):
        pid = r["id"]
        codes = {name: judge_codes[name][pid] for name in JUDGE_FILES}
        collapsed = {name: COLLAPSE[c] for name, c in codes.items()}
        values = list(collapsed.values())
        counts = Counter(values)
        if len(counts) == 3:
            decision, majority = "dropped_all_collapsed_different", None
        else:
            majority = counts.most_common(1)[0][0]
            decision = "dropped_majority_PX" if majority == "PX" else "retained"
        decision_by_id[pid] = decision
        out_row = {
            "source_id": pid,
            "source_row": row_idx,
            "prompt": r["prompt"],
            "source": r["source"],
            "subtype": r["subtype"],
            "prompt_sha256": hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest(),
            "code_luna": codes["luna"],
            "code_deepseek": codes["deepseek"],
            "code_glm": codes["glm"],
            "collapsed_luna": collapsed["luna"],
            "collapsed_deepseek": collapsed["deepseek"],
            "collapsed_glm": collapsed["glm"],
            "collapsed_majority": majority,
            "collapsed_unanimous": len(counts) == 1,
            "old_feasibility_status": old_status.get(pid, "unreviewed"),
        }
        if decision == "retained":
            retained.append(out_row)
        else:
            excluded.append({**out_row, "exclusion_reason": decision})

    prompts_path = args.out / "prompts.jsonl"
    excluded_path = args.out / "excluded.jsonl"
    for path, rows in ((prompts_path, retained), (excluded_path, excluded)):
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    decision_counts = Counter(decision_by_id.values())
    majority_breakdown = Counter(r["collapsed_majority"] for r in retained)
    unanimous_retained = sum(r["collapsed_unanimous"] for r in retained)
    crosstab = Counter(
        (old_status.get(pid, "unreviewed"), decision) for pid, decision in decision_by_id.items()
    )

    git_commit = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    manifest = {
        "dataset_name": DATASET_NAME,
        "schema_version": 1,
        "built_by": "scripts/build_x61_majority_harmful_promptset.py",
        "git_commit": git_commit,
        "rule": {
            "collapse": COLLAPSE,
            "drop_if": [
                "three collapsed labels all different ({N, PX, ACO})",
                "collapsed majority (>=2 of 3) is PX",
            ],
            "keep": "collapsed majority is N or ACO",
        },
        "inputs": {
            "source_prompts": {
                "path": str(SOURCE_JSON.relative_to(REPO)),
                "sha256": SOURCE_SHA256,
                "rows": 15000,
            },
            "judges": {
                name: {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
                for name, path in JUDGE_FILES.items()
            },
            "old_feasibility_decisions": {
                "path": str((V1_DIR / "feasibility_decisions.jsonl").relative_to(REPO)),
                "sha256": sha256_file(V1_DIR / "feasibility_decisions.jsonl"),
            },
        },
        "counts": {
            "source_total": 15000,
            "retained": len(retained),
            "dropped_all_collapsed_different": decision_counts[
                "dropped_all_collapsed_different"
            ],
            "dropped_majority_PX": decision_counts["dropped_majority_PX"],
            "retained_majority_breakdown": dict(majority_breakdown),
            "retained_collapsed_unanimous": unanimous_retained,
        },
        "old_feasibility_crosstab": {
            f"{old}|{dec}": n for (old, dec), n in sorted(crosstab.items())
        },
        "outputs": {
            "prompts.jsonl": {"sha256": sha256_file(prompts_path), "rows": len(retained)},
            "excluded.jsonl": {"sha256": sha256_file(excluded_path), "rows": len(excluded)},
        },
        "notes": [
            "Harmful-source prompts only; no benign prompts are included.",
            "Retained rows with collapsed majority N are NOT certified benign;"
            " they are kept by the user's rule, which only removes PX-majority"
            " and unstable (all-different) rows.",
            "datasets/CURRENT.json was intentionally not modified.",
        ],
    }
    manifest_path = args.out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps({"manifest_sha256": sha256_file(manifest_path), **manifest["counts"]}, indent=2))
    print("old_feasibility_crosstab:")
    for key, n in sorted(crosstab.items()):
        print(f"  {key[0]:>10} | {key[1]:<30} {n:>6}")


if __name__ == "__main__":
    main()
