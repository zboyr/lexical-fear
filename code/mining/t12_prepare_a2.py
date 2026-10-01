"""Prepare pool A2's 5000 harmful prompts for SWSE labeling (t1 -> t2).

Ids start at 20000 to stay disjoint from the pool-A pilot (0-1999), rest
(2000-5999) and full-benign (6000-10999) id ranges, so all clustered files
can be merged safely later.

Output: data/cp_a2_prompts.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")


def main():
    a2 = json.load(open(os.path.join(POOLS, "mined_pool_A2_rand.json")))
    rows = [{"id": 20000 + i, "prompt": r["text"], "source": "toxic", "subtype": r["subtype"]}
            for i, r in enumerate(r for r in a2 if r["label"] == 1)]
    out = os.path.join(POOLS, "cp_a2_prompts.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"saved {len(rows)} harmful (ids {rows[0]['id']}..{rows[-1]['id']}) -> {out}")


if __name__ == "__main__":
    main()
