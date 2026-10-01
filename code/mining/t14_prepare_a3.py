"""Prepare pool A3's 5000 harmful prompts for SWSE labeling (ids 40000+,
disjoint from pilot 0-1999, rest 2000-5999, A-benign 6000+, A2 20000+,
10k-benign 30000+).

Output: data/cp_a3_prompts.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")


def main():
    a3 = json.load(open(os.path.join(POOLS, "mined_pool_A3_bnd.json")))
    rows = [{"id": 40000 + i, "prompt": r["text"], "source": "toxic", "subtype": r["subtype"]}
            for i, r in enumerate(r for r in a3 if r["label"] == 1)]
    out = os.path.join(POOLS, "cp_a3_prompts.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"saved {len(rows)} harmful (ids {rows[0]['id']}..{rows[-1]['id']}) -> {out}")


if __name__ == "__main__":
    main()
