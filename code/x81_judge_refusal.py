"""x81 Stage 4b: apply the exact x64 refusal2 judge to one new arm."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from x58_methods import METHODS as M58  # noqa: E402
from x81_common import N_RESPONSES, RUN, read_json, sha256_file  # noqa: E402

JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"
REASONING = {"effort": "low"}
EXTRA_TOKENS = 15000
ATTEMPT_SEEDS = (42, 1042, 2042)
METHOD = M58["refusal2"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def extract_json_object(text: str):
    """The string/escape-aware balanced-object parser used by x64."""
    value = text.split("</think>")[-1]
    start = value.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(value)):
        character = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return json.loads(value[start : index + 1])
    raise ValueError("unbalanced JSON object")


def load_api_key() -> str:
    with (HERE.parent / ".env").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("OPENROUTER_API_KEY not found in .env")


class CostMeter:
    """The x64 provider-cost accounting contract without legacy imports."""

    def __init__(self, cap: float):
        self.cap = float(cap)
        self.lock = threading.Lock()
        self.usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.aborted = False

    def add(self, usage: dict) -> None:
        cost = usage.get("cost")
        if cost is None:
            cost = usage.get("prompt_tokens", 0) * 1.25e-6 + usage.get(
                "completion_tokens", 0
            ) * 7.5e-6
        with self.lock:
            self.usd += float(cost)
            self.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.completion_tokens += int(usage.get("completion_tokens", 0))
            if self.usd > self.cap:
                self.aborted = True

    def snapshot(self):
        with self.lock:
            return self.usd, self.prompt_tokens, self.completion_tokens


def call(session, key: str, model: str, prompt: str, seed: int, reasoning: dict, max_tokens: int):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "seed": seed,
        "usage": {"include": True},
        "reasoning": reasoning,
    }
    for backoff in (1, 4, 15, 60, 120):
        try:
            response = session.post(
                OPENROUTER_URL,
                json=body,
                timeout=180,
                headers={"Authorization": f"Bearer {key}"},
            )
        except requests.RequestException:
            time.sleep(backoff)
            continue
        if response.status_code == 200:
            try:
                parsed = response.json()
            except (ValueError, json.JSONDecodeError):
                time.sleep(backoff)
                continue
            if "error" in parsed:
                time.sleep(backoff)
                continue
            return parsed
        if response.status_code in (408, 429, 500, 502, 503, 524):
            time.sleep(backoff)
            continue
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    raise RuntimeError("retries exhausted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=("A", "B", "C", "P", "D", "E"))
    parser.add_argument("--generations", help="default x81 generation path for condition")
    parser.add_argument("--out", help="default x81 refusal JSONL path for condition")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cap", type=float, default=40.0)
    parser.add_argument("--threads", type=int, default=200)
    return parser.parse_args()


def repair_labels(parsed, content: str, n: int):
    if (
        isinstance(parsed, dict)
        and set(parsed) == {str(index) for index in range(n)}
        and all(isinstance(parsed[str(index)], bool) for index in range(n))
    ):
        return {"refusal": [parsed[str(index)] for index in range(n)]}, "indexed_keys"
    pattern = re.compile(
        r"\[(?:\s*(?:true|false)\s*,){%d}\s*(?:true|false)\s*\]" % (n - 1)
    )
    match = pattern.search(content)
    if match:
        values = json.loads(match.group(0))
        if len(values) == n:
            return {"refusal": values}, "bool_array"
    return None, None


def judge_row(api_key: str, row: dict, meter: CostMeter, generation_sha: str) -> dict | None:
    prompt = METHOD["prompt"](row["llm_responses"], row["prompt"])
    reasons = []
    session = requests.Session()
    for attempt_seed in ATTEMPT_SEEDS:
        if meter.aborted:
            return None
        try:
            response = call(
                session,
                api_key,
                JUDGE_MODEL,
                prompt,
                attempt_seed,
                REASONING,
                METHOD["max_tokens"] + EXTRA_TOKENS,
            )
        except RuntimeError as error:
            reasons.append(f"transport: {error}"[:100])
            continue
        meter.add(response.get("usage", {}))
        try:
            choice = response["choices"][0]
            content = choice["message"].get("content")
        except (KeyError, IndexError, TypeError) as error:
            reasons.append(f"schema: {error}"[:100])
            continue
        if not content:
            reasons.append(f"empty_content: finish={choice.get('finish_reason')}")
            continue
        try:
            parsed = extract_json_object(content)
        except (ValueError, json.JSONDecodeError) as error:
            reasons.append(f"json_parse: {error}"[:100])
            continue
        record = METHOD["validate"](parsed, N_RESPONSES)
        repaired = None
        if record is None:
            record, repaired = repair_labels(parsed, content, N_RESPONSES)
        if record is not None:
            record.update(
                {
                    "id": int(row["id"]),
                    "condition": row["condition"],
                    "prompt_sha256": row["prompt_sha256"],
                    "response_sha256": row["response_sha256"],
                    "arms_sha256": row["arms_sha256"],
                    "generation_sha256": generation_sha,
                    "judge_model": JUDGE_MODEL,
                    "attempt_seed": attempt_seed,
                }
            )
            if repaired:
                record["repaired"] = repaired
            return record
        reasons.append(f"validate: {json.dumps(parsed)[:120]}")
    return {
        "id": int(row["id"]),
        "condition": row["condition"],
        "prompt_sha256": row["prompt_sha256"],
        "response_sha256": row["response_sha256"],
        "arms_sha256": row["arms_sha256"],
        "generation_sha256": generation_sha,
        "judge_model": JUDGE_MODEL,
        "failed": True,
        "fail_reasons": reasons,
    }


def main() -> None:
    args = parse_args()
    generation_path = (
        Path(args.generations)
        if args.generations
        else RUN / f"x81_generations_{args.condition}.json"
    )
    out_path = Path(args.out) if args.out else RUN / f"x81_refusal_{args.condition}.jsonl"
    if not generation_path.exists():
        raise SystemExit(f"FATAL: generation file not found: {generation_path}")
    generations = read_json(generation_path)
    if generations.get("condition") != args.condition:
        raise SystemExit("FATAL: generation condition mismatch")
    rows = list(generations["rows"])
    if args.limit:
        rows = rows[: args.limit]
    by_id = {int(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit("FATAL: duplicate generation IDs")
    for row in rows:
        if (
            row.get("condition") != args.condition
            or row.get("arms_sha256") != generations["arms_sha256"]
            or len(row.get("llm_responses", [])) != N_RESPONSES
            or len(row.get("response_sha256", [])) != N_RESPONSES
        ):
            raise SystemExit(f"FATAL: invalid generation row {row.get('id')}")

    generation_sha = sha256_file(generation_path)
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("failed"):
                    continue
                prompt_id = int(record["id"])
                source = by_id.get(prompt_id)
                if source is None:
                    continue
                if (
                    record.get("condition") != args.condition
                    or record.get("prompt_sha256") != source["prompt_sha256"]
                    or record.get("response_sha256") != source["response_sha256"]
                    or record.get("arms_sha256") != generations["arms_sha256"]
                    or len(record.get("refusal", [])) != N_RESPONSES
                    or not all(isinstance(value, bool) for value in record["refusal"])
                ):
                    raise SystemExit(f"FATAL: invalid successful judge resume row {prompt_id}")
                if prompt_id in done:
                    raise SystemExit(f"FATAL: duplicate successful judge row {prompt_id}")
                done.add(prompt_id)

    todo = [row for row in rows if int(row["id"]) not in done]
    if not todo:
        print(f"x81 refusal {args.condition}: nothing new ({len(done)} done)")
        return
    api_key = load_api_key()
    meter = CostMeter(args.cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        futures = {
            pool.submit(judge_row, api_key, row, meter, generation_sha): row for row in todo
        }
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if record.get("failed"):
                n_failed += 1
            else:
                n_ok += 1
    usd, prompt_tokens, completion_tokens = meter.snapshot()
    print(
        f"x81 refusal {args.condition}: +{n_ok} ok, {n_failed} failed; "
        f"${usd:.2f}, prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
    )
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()
