"""q19 runner: label the 100 disagreement rows with two OpenRouter judges under
each labeling-process method, to find which process maximizes GPT<->DeepSeek
agreement (and T-AUC consistency).

Judges: openai/gpt-5.6-terra, deepseek/deepseek-v4-flash.
Methods: holistic | criteria | severity | criteria_sev (see q19_methods).

JSON-object response mode for BOTH judges (identical interface = fair
comparison), robust parse + validation, <=3 attempts with per-attempt seeds,
resume-safe JSONL. Cost meter with hard abort.

Usage:
  .venv/bin/python code/lib/q19_run_judges.py --judge gpt --method severity --limit 2
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from q19_methods import METHODS, prompt_shared_crit  # noqa: E402
from rubric_common import extract_json_object  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
ATTEMPT_SEEDS = (42, 123, 7)
DEFAULT_SAMPLE = os.path.join(HERE, "data", "disagree100.json")

# extra_tokens = reasoning-token headroom on top of the method's output budget
# (DeepSeek-V4 is a reasoning model: reasoning tokens are billed as completion
# tokens and consume max_tokens BEFORE the JSON, so a tight cap returns
# content=None with finish_reason=length).
JUDGES = {
    "gpt": dict(model="openai/gpt-5.6-terra", reasoning={"effort": "low"}, extra_tokens=0),
    "deepseek": dict(model="deepseek/deepseek-v4-flash", reasoning={"effort": "low"}, extra_tokens=9000),
    "glm": dict(model="z-ai/glm-5.2", reasoning={"effort": "low"}, extra_tokens=9000),
}


def load_api_key():
    """Load the OpenRouter credential from <repo>/.env (a line
    OPENROUTER_API_KEY=...), falling back to the OPENROUTER_API_KEY
    environment variable."""
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1]
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    raise RuntimeError("OPENROUTER_API_KEY not found in .env or the environment")


class CostMeter:
    """Thread-safe use of provider-reported cost with a conservative fallback."""

    def __init__(self, cap):
        self.cap = float(cap)
        self.lock = threading.Lock()
        self.usd = 0.0
        self.pt = 0
        self.ct = 0
        self.aborted = False

    def add(self, usage):
        cost = usage.get("cost")
        if cost is None:
            # This fallback matches the historical GPT utility.  OpenRouter's
            # response normally supplies the model-specific exact cost.
            cost = usage.get("prompt_tokens", 0) * 1.25e-6 + usage.get(
                "completion_tokens", 0
            ) * 7.5e-6
        with self.lock:
            self.usd += float(cost)
            self.pt += int(usage.get("prompt_tokens", 0))
            self.ct += int(usage.get("completion_tokens", 0))
            if self.usd > self.cap:
                self.aborted = True

    def snapshot(self):
        with self.lock:
            return self.usd, self.pt, self.ct


def call(session, key, model, prompt, seed, reasoning, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "seed": seed,
        "usage": {"include": True},
    }
    if reasoning:
        body["reasoning"] = reasoning
    for backoff in (1, 4, 15, 60, 120):
        try:
            r = session.post(URL, json=body, timeout=180,
                             headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException:
            time.sleep(backoff)
            continue
        if r.status_code == 200:
            try:
                res = r.json()
            except (ValueError, json.JSONDecodeError):
                time.sleep(backoff)  # malformed 200 body (provider artifact) -> retry
                continue
            if "error" in res:
                time.sleep(backoff)
                continue
            return res
        if r.status_code in (408, 429, 500, 502, 503, 524):
            time.sleep(backoff)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("retries exhausted")


def judge_row(item, key, jcfg, mcfg, method, meter, criteria=None, seeds=ATTEMPT_SEEDS):
    n = len(item["llm_responses"])
    if method == "shared_crit":
        crit = (criteria or {}).get(item["id"], "")
        prompt = prompt_shared_crit(item["llm_responses"], item["prompt"], crit)
    else:
        prompt = mcfg["prompt"](item["llm_responses"], item["prompt"])
    session = requests.Session()
    why = "no attempt"
    for attempt, seed in enumerate(seeds):
        if meter.aborted:
            return None
        try:
            res = call(session, key, jcfg["model"], prompt, seed,
                       jcfg["reasoning"], mcfg["max_tokens"] + jcfg["extra_tokens"])
        except RuntimeError as e:
            why = str(e); continue
        meter.add(res.get("usage", {}))
        try:
            ch = res["choices"][0]
            content = ch["message"].get("content")
            if not content:
                why = f"empty content (finish={ch.get('finish_reason')})"; continue
            parsed = extract_json_object(content)
        except (ValueError, json.JSONDecodeError, KeyError, IndexError) as e:
            why = f"parse: {e}"; continue
        why = mcfg["validate"](parsed, n)
        if why is None:
            lc = mcfg["derive"](parsed, n, f"{method}_{jcfg['model']}")
            return {"id": item["id"], "llm_clustering": lc,
                    "judge": {"attempt": attempt + 1, "seed": seed}}
    return {"id": item["id"], "failed": True, "reason": str(why)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=sorted(JUDGES), required=True)
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--sample", default=DEFAULT_SAMPLE, help="path to the prompt sample json")
    ap.add_argument("--tag", default="disagree100", help="output-file prefix")
    ap.add_argument("--seed", type=int, default=42,
                    help="primary sampling seed (retries use the other seeds)")
    ap.add_argument("--effort", choices=("low", "medium", "high"), default=None,
                    help="override reasoning effort for this run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_cost", type=float, default=15.0)
    args = ap.parse_args()

    key = load_api_key()
    jcfg, mcfg = dict(JUDGES[args.judge]), METHODS[args.method]
    if args.effort:
        jcfg["reasoning"] = {"effort": args.effort}
    out_path = os.path.join(HERE, "data", f"{args.tag}_{args.method}_{args.judge}.jsonl")
    todo = json.load(open(args.sample))
    if args.limit:
        todo = todo[:args.limit]
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            if not r.get("failed"):
                done.add(r["id"])
    todo = [it for it in todo if it["id"] not in done]
    print(f"{args.judge}/{args.method}: {len(todo)} rows ({len(done)} done) -> {out_path}", flush=True)
    if not todo:
        return

    criteria = None
    if args.method == "shared_crit":
        cpath = os.path.join(HERE, "data", f"{args.tag}_criteria.json")
        criteria = {int(k): v for k, v in json.load(open(cpath)).items()}

    meter = CostMeter(args.max_cost)
    t0 = time.time(); n_ok = n_fail = 0
    with open(out_path, "a") as fh, ThreadPoolExecutor(args.concurrency) as ex:
        seeds = tuple([args.seed] + [s for s in (42, 123, 7, 99) if s != args.seed][:2])
        fut2id = {ex.submit(judge_row, it, key, jcfg, mcfg, args.method, meter, criteria, seeds): it["id"]
                  for it in todo}
        for k, fut in enumerate(as_completed(fut2id), 1):
            try:
                rec = fut.result()
            except Exception as e:  # a crashing row must not kill the grid
                rec = {"id": fut2id[fut], "failed": True, "reason": f"exception: {e}"}
            if rec is None:
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            n_ok += 0 if rec.get("failed") else 1
            n_fail += 1 if rec.get("failed") else 0
            if k % 25 == 0 or k == len(fut2id):
                usd, pt, ct = meter.snapshot()
                print(f"  {k}/{len(fut2id)} ok={n_ok} fail={n_fail} ${usd:.2f} {time.time()-t0:.0f}s", flush=True)
    usd, pt, ct = meter.snapshot()
    print(f"done {args.judge}/{args.method}: ok={n_ok} fail={n_fail} ${usd:.2f} aborted={meter.aborted}")


if __name__ == "__main__":
    main()
