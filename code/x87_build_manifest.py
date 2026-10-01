"""Freeze OR-Bench and StrongREJECT rows plus the exact Qwen snapshot."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

from x81_common import model_snapshot_record
from x87_common import (
    BENCHMARKS,
    CONDITIONS,
    MODEL_ID,
    ORBENCH_REPO,
    ORBENCH_REVISION,
    RUN,
    STRONGREJECT_REVISION,
    STRONGREJECT_URL,
    atomic_json,
    file_record,
    read_json,
    sha256_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--x86-manifest", default="data/runs/x86/x86_manifest.json")
    parser.add_argument("--out", default=str(RUN / "x87_manifest.json"))
    parser.add_argument("--orbench-revision", default=ORBENCH_REVISION)
    return parser.parse_args()


def row_id(benchmark: str, index: int, prompt: str) -> str:
    return f"{benchmark}:{index:05d}:{sha256_text(prompt)[:12]}"


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    x86_manifest = read_json(args.x86_manifest)
    if x86_manifest.get("status") != "frozen" or x86_manifest.get("model_id") != MODEL_ID:
        raise RuntimeError("x86 manifest is not a frozen Qwen3.5-4B artifact")
    requested_snapshot = Path(args.model_snapshot).resolve()
    x86_snapshot_name = Path(x86_manifest["model_snapshot"]).name
    if requested_snapshot.name != x86_snapshot_name:
        raise RuntimeError(
            f"model snapshot differs from x86: {requested_snapshot.name} != {x86_snapshot_name}"
        )

    rows = []
    for config, benchmark in (
        ("or-bench-hard-1k", "orbench_hard"),
        ("or-bench-toxic", "orbench_toxic"),
    ):
        dataset = load_dataset(
            ORBENCH_REPO, config, split="train", revision=args.orbench_revision
        )
        for index, example in enumerate(dataset):
            prompt = str(example["prompt"])
            rows.append(
                {
                    "id": row_id(benchmark, index, prompt),
                    "benchmark": benchmark,
                    "source_index": index,
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "category": str(example.get("category", "")),
                }
            )

    strongreject = load_dataset("csv", data_files=STRONGREJECT_URL, split="train")
    for index, example in enumerate(strongreject):
        prompt = str(example["forbidden_prompt"])
        rows.append(
            {
                "id": row_id("strongreject", index, prompt),
                "benchmark": "strongreject",
                "source_index": index,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "category": str(example.get("category", "")),
                "source": str(example.get("source", "")),
            }
        )
    counts = {name: sum(row["benchmark"] == name for row in rows) for name in BENCHMARKS}
    if counts["orbench_hard"] < 1_000 or counts["strongreject"] != 313:
        raise RuntimeError(f"unexpected benchmark sizes: {counts}")
    try:
        lm_eval_version = importlib.metadata.version("lm-eval")
    except importlib.metadata.PackageNotFoundError:
        lm_eval_version = None
    manifest = {
        "schema_version": 1,
        "experiment": "x87",
        "status": "frozen",
        "model_id": MODEL_ID,
        "model_snapshot": model_snapshot_record(requested_snapshot),
        "x86_manifest": file_record(args.x86_manifest),
        "conditions": list(CONDITIONS),
        "benchmarks": {
            "orbench": {
                "repo": ORBENCH_REPO,
                "revision": args.orbench_revision,
                "configs": ["or-bench-hard-1k", "or-bench-toxic"],
            },
            "strongreject": {
                "source_revision": STRONGREJECT_REVISION,
                "url": STRONGREJECT_URL,
                "evaluator": "strongreject_finetuned",
            },
            "ifeval": {
                "implementation": "lm-evaluation-harness/ifeval",
                "lm_eval_version": lm_eval_version,
                "primary_metric": "prompt_level_strict_acc",
            },
        },
        "counts": counts,
        "rows_sha256": sha256_text(
            "\n".join(f"{row['id']}\t{row['prompt_sha256']}" for row in rows)
        ),
        "rows": rows,
    }
    atomic_json(args.out, manifest)
    print({"out": args.out, "counts": counts, "lm_eval": lm_eval_version})


if __name__ == "__main__":
    main()
