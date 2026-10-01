"""Prepare the 10k+10k pool A+A2 labeled set for feature extraction.

Harmful: merge all three clustered batches (pilot ids 0-999, rest 2000-5999,
A2 20000-24999). Benign: pool A benign (5000) + pool A2 benign (5000), T=0,
ids 30000+.

Outputs: data/cp_10k_clustered.json, data/cp_10k_benign.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
D = POOLS


def main():
    merged = []
    for f in ("cp_clustered.json", "cp_rest_clustered.json", "cp_a2_clustered.json"):
        merged += json.load(open(os.path.join(D, f)))
    merged.sort(key=lambda x: x["id"])
    ids = [x["id"] for x in merged]
    assert len(ids) == len(set(ids)), "id collision"
    ok = sum(1 for x in merged if "error" not in x.get("llm_clustering", {"error": 1}))
    json.dump(merged, open(os.path.join(D, "cp_10k_clustered.json"), "w"), indent=1)
    print(f"harmful: {len(merged)} merged ({ok} valid)")

    ben = [r for r in json.load(open(os.path.join(D, "mined_pool_A_rand.json"))) if r["label"] == 0]
    ben += [r for r in json.load(open(os.path.join(D, "mined_pool_A2_rand.json"))) if r["label"] == 0]
    rows = [{"id": 30000 + i, "prompt": r["text"], "source": "hard", "subtype": r["subtype"]}
            for i, r in enumerate(ben)]
    assert len({r["prompt"] for r in rows}) == len(rows), "duplicate benign text"
    json.dump(rows, open(os.path.join(D, "cp_10k_benign.json"), "w"), indent=1)
    print(f"benign: {len(rows)} (T=0)")


if __name__ == "__main__":
    main()
