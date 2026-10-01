"""x86 Stage 5: exact x64 refusal2 or comply2 judge for one condition."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from x58_methods import METHODS
from x81_judge_refusal import (
    ATTEMPT_SEEDS,
    EXTRA_TOKENS,
    JUDGE_MODEL,
    REASONING,
    CostMeter,
    call,
    extract_json_object,
    load_api_key,
)
from x86_common import RUN, file_record, read_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--method", required=True, choices=("refusal2", "comply2"))
    parser.add_argument("--generations")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cap", type=float, default=40.0)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--judge-model", default=None, help="override JUDGE_MODEL (e.g. comply2 reliability)")
    parser.add_argument("--per-response", action="store_true", help="judge one response per API call (reliable for comply2 on long responses)")
    return parser.parse_args()


def repair_labels(parsed, content: str, n: int, method: str):
    field = "refusal" if method == "refusal2" else "complied"
    if (
        isinstance(parsed, dict)
        and set(parsed) == {str(index) for index in range(n)}
        and all(isinstance(parsed[str(index)], bool) for index in range(n))
    ):
        return {field: [parsed[str(index)] for index in range(n)]}, "indexed_keys"
    # Unambiguous shape drift under the correct field name (dominant flash
    # failure mode in per-response mode): a bare bool instead of a length-1
    # list, or 0/1 integers instead of bools. Accepting these is a strict
    # extension -- nothing previously accepted changes meaning.
    value = parsed.get(field) if isinstance(parsed, dict) else None
    if n == 1 and isinstance(value, bool):
        return {field: [value]}, "bare_bool"
    if (
        isinstance(value, list)
        and len(value) == n
        and all(isinstance(x, (bool, int)) and int(x) in (0, 1) for x in value)
    ):
        return {field: [bool(x) for x in value]}, "int_list"
    pattern = re.compile(
        r"\[(?:\s*(?:true|false)\s*,){%d}\s*(?:true|false)\s*\]" % (n - 1)
    )
    match = pattern.search(content)
    if match:
        values = json.loads(match.group(0))
        if len(values) == n:
            return {field: values}, "bool_array"
    return None, None


def _judge_chunk(session, api_key, judge_model, method, responses, query, meter, reasons):
    """One API call judging `responses`; returns (field_values, repaired) or (None, None)."""
    contract = METHODS[method]
    field = "refusal" if method == "refusal2" else "complied"
    n = len(responses)
    prompt = contract["prompt"](responses, query)
    for attempt_seed in ATTEMPT_SEEDS:
        if meter.aborted:
            return None, None
        try:
            response = call(
                session, api_key, judge_model, prompt, attempt_seed, REASONING,
                contract["max_tokens"] + EXTRA_TOKENS,
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
        record = contract["validate"](parsed, n)
        repaired = None
        if record is None:
            record, repaired = repair_labels(parsed, content, n, method)
        if record is not None:
            return record[field], repaired
        reasons.append(f"validate: {json.dumps(parsed)[:120]}")
    return None, None


def judge_one(api_key, row, method, meter, generation_sha, judge_model=JUDGE_MODEL, per_response=False):
    field = "refusal" if method == "refusal2" else "complied"
    responses = row["llm_responses"]
    query = row["prompt"]
    reasons = []
    import requests

    session = requests.Session()
    repaired_any = None
    if per_response:
        # One context per response: judge each response alone (small, reliable input),
        # then assemble the length-n label array. Any single unjudgeable response
        # fails the whole row (it will be retried on the next resume pass).
        values = []
        for index, response in enumerate(responses):
            chunk, repaired = _judge_chunk(
                session, api_key, judge_model, method, [response], query, meter, reasons
            )
            if chunk is None:
                values = None
                reasons.append(f"per_response_failed_at={index}")
                break
            repaired_any = repaired_any or repaired
            values.append(bool(chunk[0]))
    else:
        values, repaired_any = _judge_chunk(
            session, api_key, judge_model, method, responses, query, meter, reasons
        )

    base = {
        "id": int(row["id"]),
        "condition": row["condition"],
        "scope": row["scope"],
        "method": method,
        "prompt_sha256": row["prompt_sha256"],
        "response_sha256": row["response_sha256"],
        "run_hash": row["run_hash"],
        "generation_sha256": generation_sha,
        "judge_model": judge_model,
    }
    if values is None:
        if meter.aborted:
            return None
        return {**base, "failed": True, "fail_reasons": reasons}
    record = {**base, field: list(values), "per_response": per_response}
    if repaired_any:
        record["repaired"] = repaired_any
    return record


def main() -> None:
    args = parse_args()
    generation_path = (
        Path(args.generations)
        if args.generations
        else RUN / f"x86_generations_{args.condition}.json"
    )
    out_path = (
        Path(args.out)
        if args.out
        else RUN / f"x86_{args.method}_{args.condition}.jsonl"
    )
    generations = read_json(generation_path)
    if generations.get("experiment") != "x86" or generations.get("condition") != args.condition:
        raise SystemExit("FATAL: invalid x86 generation artifact")
    rows = list(generations["rows"])
    if args.limit:
        rows = rows[: args.limit]
    by_id = {int(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit("FATAL: duplicate generation IDs")
    generation_sha = sha256_file(generation_path)
    field = "refusal" if args.method == "refusal2" else "complied"
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("failed"):
                    continue
                row_id = int(record["id"])
                source = by_id.get(row_id)
                if source is None:
                    continue
                if (
                    record.get("condition") != args.condition
                    or record.get("method") != args.method
                    or record.get("response_sha256") != source["response_sha256"]
                    or record.get("run_hash") != source["run_hash"]
                    or len(record.get(field, [])) != len(source["llm_responses"])
                    or not all(isinstance(value, bool) for value in record[field])
                ):
                    raise SystemExit(f"FATAL: invalid judge resume row {row_id}")
                if row_id in done:
                    raise SystemExit(f"FATAL: duplicate successful judge row {row_id}")
                done.add(row_id)
    todo = [row for row in rows if int(row["id"]) not in done]
    if not todo:
        print(f"x86 {args.method} {args.condition}: nothing new ({len(done)} done)")
        return
    api_key = load_api_key()
    judge_model = args.judge_model or JUDGE_MODEL
    meter = CostMeter(args.cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        futures = {
            pool.submit(judge_one, api_key, row, args.method, meter, generation_sha, judge_model, args.per_response): row
            for row in todo
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
        f"x86 {args.method} {args.condition}: +{n_ok} ok, {n_failed} failed; "
        f"${usd:.2f}, prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
    )
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()

