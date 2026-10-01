"""x64: v4/v5 label judging over the x63 regenerated responses.

Judge: deepseek/deepseek-v4-flash-0731 @ reasoning low (x62 verdict), single
pass (seed 42), x58-verbatim refusal2 + comply2 prompts, retry ladder plus the
x62 schema repair (indexed-keys / bool-array salvage, records tagged
"repaired"). Incremental and resume-safe: judges whatever complete rows have
landed in the local staging dir; re-run any time as more x63 shards arrive.

Usage:
  python code/x64_run.py --tag qwen
"""
import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
QDIR = os.path.join(HERE, "lib")
sys.path.insert(0, QDIR)
sys.path.insert(0, HERE)
from q19_run_judges import CostMeter, call, load_api_key  # noqa: E402
from rubric_common import extract_json_object  # noqa: E402
from x58_methods import METHODS as M58  # noqa: E402

METHODS = {"refusal2": M58["refusal2"], "comply2": M58["comply2"]}
LABEL_KEY = {"refusal2": "refusal", "comply2": "complied"}
DATA = os.path.join(ROOT, "data")
JUDGED = os.path.join(DATA, "judged")
STAGING = os.path.join(DATA, "generations")
N_EXPECTED = 13527

JUDGE = dict(model="deepseek/deepseek-v4-flash-0731",
             reasoning={"effort": "low"}, extra_tokens=15000)
PASS_SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True,
                choices=["qwen", "llama", "smollm3", "gemma", "phi",
                         "qwen08b", "qwen2b", "qwen9b", "qwen27b"])
ap.add_argument("--methods", nargs="*", default=list(METHODS))
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cap", type=float, default=40.0)
ap.add_argument("--threads", type=int, default=500)
ARGS = ap.parse_args()


def load_landed_rows(tag):
    """Complete (10-response) rows from landed x63 shards.

    Rows still truncated after the 4096-token retry are kept: unlike the old
    200-token bug, a 4096-token cut leaves refusal/comply fully judgeable, and
    excluding them would make the 13,527 full-coverage gate unreachable."""
    rows = {}
    n_trunc = 0
    for path in sorted(glob.glob(os.path.join(STAGING, f"x63_{tag}_*_*.json"))):
        try:
            shard = json.load(open(path))
        except (ValueError, json.JSONDecodeError):
            print(f"[skip] {os.path.basename(path)}: mid-write/corrupt, will retry later")
            continue
        for o in shard:
            if len(o.get("llm_responses", [])) == 10:
                if o["id"] in rows:
                    sys.exit(f"FATAL: duplicate id {o['id']} across shards")
                if any(o["truncated"]):
                    n_trunc += 1
                rows[o["id"]] = o
    if n_trunc:
        print(f"[note] {n_trunc} rows contain >=1 response truncated at the "
              f"4096 retry cap; judged as-is", flush=True)
    return list(rows.values())


def repair_labels(parsed, content, n, label_key):
    if isinstance(parsed, dict) and set(parsed) == {str(i) for i in range(n)} \
            and all(isinstance(parsed[str(i)], bool) for i in range(n)):
        return {label_key: [parsed[str(i)] for i in range(n)]}, "indexed_keys"
    rx = re.compile(r"\[(?:\s*(?:true|false)\s*,){%d}\s*(?:true|false)\s*\]" % (n - 1))
    m = rx.search(content)
    if m:
        arr = json.loads(m.group(0))
        if len(arr) == n:
            return {label_key: arr}, "bool_array"
    return None, None


def judge_row(session, key, mcfg, row, meter, label_key):
    prompt = mcfg["prompt"](row["llm_responses"], row["prompt"])
    n = len(row["llm_responses"])
    reasons = []
    for attempt_seed in (PASS_SEED, PASS_SEED + 1000, PASS_SEED + 2000):
        if meter.aborted:
            return None
        try:
            res = call(session, key, JUDGE["model"], prompt, attempt_seed,
                       JUDGE["reasoning"], mcfg["max_tokens"] + JUDGE["extra_tokens"])
        except RuntimeError as e:
            reasons.append(f"transport: {e}"[:100])
            continue
        meter.add(res.get("usage", {}))
        content = res["choices"][0]["message"].get("content")
        if not content:
            reasons.append(f"empty_content: finish={res['choices'][0].get('finish_reason')}")
            continue
        try:
            parsed = extract_json_object(content)
        except (ValueError, json.JSONDecodeError):
            reasons.append(f"json_parse: {content[:80]!r}")
            continue
        rec = mcfg["validate"](parsed, n)
        repaired = None
        if rec is None:
            rec, repaired = repair_labels(parsed, content, n, label_key)
        if rec is not None:
            rec["id"] = row["id"]
            rec["attempt_seed"] = attempt_seed
            if repaired:
                rec["repaired"] = repaired
            return rec
        reasons.append(f"validate: {json.dumps(parsed)[:120]}")
    return {"id": row["id"], "failed": True, "fail_reasons": reasons}


def main():
    rows = load_landed_rows(ARGS.tag)
    print(f"{ARGS.tag}: {len(rows)}/{N_EXPECTED} complete rows landed", flush=True)
    if not rows:
        return
    if ARGS.limit:
        rows = rows[:ARGS.limit]
    key = load_api_key()
    meter = CostMeter(ARGS.cap)
    for method in ARGS.methods:
        mcfg = METHODS[method]
        out_path = os.path.join(JUDGED, f"x64_{method}_{ARGS.tag}_s{PASS_SEED}.jsonl")
        done = set()
        if os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    rec = json.loads(line)
                    if not rec.get("failed"):
                        done.add(rec["id"])
        todo = [r for r in rows if r["id"] not in done]
        if not todo:
            print(f"{method}/{ARGS.tag}: nothing new ({len(done)} done)", flush=True)
            continue
        usd0 = meter.snapshot()[0]
        session = requests.Session()
        n_ok = n_fail = 0
        with open(out_path, "a") as f, ThreadPoolExecutor(ARGS.threads) as ex:
            futs = {ex.submit(judge_row, session, key, mcfg, r, meter,
                              LABEL_KEY[method]): r for r in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                if rec is None:
                    continue
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if rec.get("failed"):
                    n_fail += 1
                else:
                    n_ok += 1
        usd, pt, ct = meter.snapshot()
        print(f"{method}/{ARGS.tag}: +{n_ok} ok, {n_fail} failed "
              f"(${usd - usd0:.2f} this method; ${usd:.2f} this run)", flush=True)
        if meter.aborted:
            print("COST CAP HIT -- aborting", flush=True)
            return
    print(f"RUN DONE ${meter.snapshot()[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
