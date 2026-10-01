"""Prepare the REMAINING 4000 harmful prompts of mined pool A for SWSE labeling
(the pilot t0 used 1000 harmful + 1000 benign; this covers the harmful rest).

Ids continue after the pilot (2000, 2001, ...) so the two labeled sets can be
merged later without collisions. Only source=toxic rows — benign keeps T=0 by
project convention and needs no generation.

Output: data/pools/cp_rest_prompts.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
POOL = os.path.join(POOLS, "mined_pool_A_rand.json")
PILOT = os.path.join(POOLS, "cp_prompts.json")
OUT = os.path.join(POOLS, "cp_rest_prompts.json")


def main():
    pool = json.load(open(POOL))
    pilot = json.load(open(PILOT))
    used = {r["prompt"] for r in pilot if r["source"] == "toxic"}
    start_id = max(r["id"] for r in pilot) + 1  # 2000

    rows = []
    for r in pool:
        if r["label"] == 1 and r["text"] not in used:
            rows.append({"id": start_id + len(rows), "prompt": r["text"],
                         "source": "toxic", "subtype": r["subtype"]})
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"saved {len(rows)} remaining harmful (ids {rows[0]['id']}..{rows[-1]['id']}) -> {OUT}")


if __name__ == "__main__":
    main()
