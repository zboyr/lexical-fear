"""Stage 0: select a balanced 1000 harmful + 1000 benign subset of mined pool A
and write it in the downstream positional contract (harmful first, benign second).

Output: data/pools/cp_prompts.json
  list of {"id", "prompt", "source": "toxic"|"hard", "orig_idx", "subtype"}
  ids are contiguous 0..N-1, harmful (toxic=1) first then benign (hard=0).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
POOLS = os.path.join(ROOT, "data", "pools")
RESULTS = os.path.join(ROOT, "data", "results")
POOL = os.path.join(POOLS, "mined_pool_A_rand.json")
OUT = os.path.join(POOLS, "cp_prompts.json")
N_PER = 1000
SEED = 42


def main():
    pool = json.load(open(POOL))
    rng = np.random.default_rng(SEED)
    h = [r for r in pool if r["label"] == 1]
    b = [r for r in pool if r["label"] == 0]
    hp = [h[i] for i in rng.choice(len(h), N_PER, replace=False)]
    bp = [b[i] for i in rng.choice(len(b), N_PER, replace=False)]

    rows = []
    for r in hp:
        rows.append({"prompt": r["text"], "source": "toxic", "subtype": r["subtype"]})
    for r in bp:
        rows.append({"prompt": r["text"], "source": "hard", "subtype": r["subtype"]})
    for i, r in enumerate(rows):
        r["id"] = i

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"saved {len(rows)} ({N_PER} toxic + {N_PER} hard) -> {OUT}")


if __name__ == "__main__":
    main()
