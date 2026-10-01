"""Run lm-evaluation-harness GSM8K under one x87 condition (post-hoc extension).

GSM8K was not pre-registered in the x87 plan; outputs from this script are
descriptive capability context only and never feed the x87 gate file.
"""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
from pathlib import Path

from transformers import AutoTokenizer

from x87_common import (
    ALL_CONDITIONS,
    MODEL_ID,
    RUN,
    assert_qwen_boundary,
    atomic_json,
    file_record,
    format_user_chat,
    load_causal_lm,
    public_condition_metadata,
    read_json,
    setup_condition,
)

TASK = "gsm8k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=ALL_CONDITIONS)
    parser.add_argument("--manifest", default=str(RUN / "x87_manifest.json"))
    parser.add_argument("--x86-directions", default="data/runs/x86/x86_directions.npz")
    parser.add_argument("--x86-calibration", default="data/runs/x86/x86_kl_calibration.json")
    parser.add_argument("--x75-adapter", default="data/runs/x87/x75_qwen_refusal_weighted_true.pt")
    parser.add_argument("--out")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "frozen" or manifest.get("model_id") != MODEL_ID:
        raise SystemExit("FATAL: invalid x87 manifest")
    actual_version = importlib.metadata.version("lm-eval")
    expected_version = manifest["benchmarks"]["ifeval"]["lm_eval_version"]
    if expected_version is not None and actual_version != expected_version:
        raise SystemExit(
            f"FATAL: lm-eval changed after manifest freeze: {actual_version} != {expected_version}"
        )
    model_source = manifest["model_snapshot"]["snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    assert_qwen_boundary(tokenizer, format_user_chat(tokenizer, "x87 boundary audit"))
    tokenizer.apply_chat_template = functools.partial(
        tokenizer.apply_chat_template, enable_thinking=False
    )
    model = load_causal_lm(model_source)
    setup = setup_condition(
        model,
        args.condition,
        x86_directions=args.x86_directions,
        x86_calibration=args.x86_calibration,
        x75_adapter=args.x75_adapter,
    )
    model.eval()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    result = lm_eval.simple_evaluate(
        model=lm,
        tasks=[TASK],
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        log_samples=True,
        random_seed=8700,
        numpy_random_seed=8700,
        torch_random_seed=8700,
        fewshot_random_seed=8700,
    )
    payload = {
        "schema_version": 1,
        "experiment": "x87",
        "condition": args.condition,
        "model_id": MODEL_ID,
        "manifest": file_record(manifest_path),
        "lm_eval_version": actual_version,
        "intervention": public_condition_metadata(setup),
        "task": TASK,
        "post_hoc": True,
        "results": result.get("results"),
        "versions": result.get("versions"),
        "n-samples": result.get("n-samples"),
        "samples": result.get("samples", {}).get(TASK, []),
    }
    output = Path(args.out) if args.out else RUN / f"x87_gsm8k_{args.condition}.json"
    atomic_json(output, payload)
    print(json.dumps(result.get("results"), default=str, indent=1))


if __name__ == "__main__":
    main()
