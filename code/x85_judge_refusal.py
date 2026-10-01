"""x85 Stage 3: exact x83/x64 refusal2 judge for one sparse behavior arm."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from x81_judge_refusal import CostMeter, judge_row, load_api_key
from x85_common import CONDITIONS, N_RESPONSES, RUN, read_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--generations")
    parser.add_argument("--out")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cap", type=float, default=40.0)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_path = (
        Path(args.generations)
        if args.generations
        else RUN / f"x85_generations_{args.condition}.json"
    )
    out_path = Path(args.out) if args.out else RUN / f"x85_refusal_{args.condition}.jsonl"
    generations = read_json(generation_path)
    if generations.get("experiment") != "x85" or generations.get("condition") != args.condition:
        raise SystemExit("FATAL: generation condition/experiment mismatch")
    rows = list(generations["rows"])
    if args.limit:
        rows = rows[: args.limit]
    by_id = {int(row["id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit("FATAL: duplicate behavior IDs")
    for row in rows:
        if (
            row.get("condition") != args.condition
            or row.get("arms_sha256") != generations["cohort_sha256"]
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
                behavior_id = int(record["id"])
                source = by_id.get(behavior_id)
                if source is None:
                    continue
                if (
                    record.get("cell_id") != source["cell_id"]
                    or record.get("condition") != args.condition
                    or record.get("prompt_sha256") != source["prompt_sha256"]
                    or record.get("response_sha256") != source["response_sha256"]
                    or record.get("arms_sha256") != generations["cohort_sha256"]
                    or len(record.get("refusal", [])) != N_RESPONSES
                    or not all(isinstance(value, bool) for value in record["refusal"])
                ):
                    raise SystemExit(f"FATAL: invalid successful judge resume row {behavior_id}")
                if behavior_id in done:
                    raise SystemExit(f"FATAL: duplicate successful judge row {behavior_id}")
                done.add(behavior_id)

    todo = [row for row in rows if int(row["id"]) not in done]
    if not todo:
        print(f"x85 refusal {args.condition}: nothing new ({len(done)} done)")
        return
    api_key = load_api_key()
    meter = CostMeter(args.cap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_failed = 0
    with out_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.threads) as pool:
        futures = {pool.submit(judge_row, api_key, row, meter, generation_sha): row for row in todo}
        for future in as_completed(futures):
            source = futures[future]
            record = future.result()
            if record is None:
                continue
            record.update(
                {
                    "cell_id": source["cell_id"],
                    "prompt_id": source["prompt_id"],
                    "word_index": source["word_index"],
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if record.get("failed"):
                n_failed += 1
            else:
                n_ok += 1
    usd, prompt_tokens, completion_tokens = meter.snapshot()
    print(
        f"x85 refusal {args.condition}: +{n_ok} ok, {n_failed} failed; "
        f"${usd:.2f}, prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}"
    )
    if meter.aborted:
        raise SystemExit("COST CAP HIT")


if __name__ == "__main__":
    main()

