"""Run lm-evaluation-harness IFEval or GSM8K under one x88 model/condition."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import json
from pathlib import Path

from x88_common import (
    CONDITIONS,
    MODELS,
    RUN,
    atomic_json,
    default_paths,
    file_record,
    load_model,
    load_tokenizer,
    public_condition_metadata,
    read_json,
    setup_condition,
    tag_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("ifeval", "gsm8k"))
    parser.add_argument("--model", required=True, choices=tuple(MODELS))
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--manifest", default=str(RUN / "x88_manifest.json"))
    parser.add_argument("--out")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "frozen" or manifest.get("experiment") != "x88":
        raise SystemExit("FATAL: invalid x88 manifest")
    actual_version = importlib.metadata.version("lm-eval")
    expected_version = manifest["benchmarks"][args.task]["lm_eval_version"]
    if expected_version is not None and actual_version != expected_version:
        raise SystemExit(
            f"FATAL: lm-eval changed after manifest freeze: {actual_version} != {expected_version}"
        )
    entry = manifest["models"][args.model]
    paths = default_paths(args.model)
    tokenizer = load_tokenizer(entry["snapshot"], args.model)
    tokenizer.apply_chat_template = functools.partial(
        tokenizer.apply_chat_template, enable_thinking=False
    )
    model = load_model(entry["snapshot"], args.model)
    setup = setup_condition(
        model,
        args.model,
        args.condition,
        directions=paths["directions"],
        adapter_path=paths["adapter"],
    ) if args.condition != "base" else setup_condition(model, args.model, "base")
    model.eval()

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    kwargs = {"fewshot_as_multiturn": True} if args.task == "gsm8k" else {}
    result = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.task],
        apply_chat_template=True,
        log_samples=True,
        random_seed=8800,
        numpy_random_seed=8800,
        torch_random_seed=8800,
        fewshot_random_seed=8800,
        **kwargs,
    )
    samples = result.get("samples", {}).get(args.task, [])
    if args.task == "gsm8k":
        for sample in samples:
            sample.pop("doc", None)
            sample.pop("arguments", None)
    payload = {
        "schema_version": 1,
        "experiment": "x88",
        "model_tag": args.model,
        "condition": args.condition,
        "model_id": MODELS[args.model]["model_id"],
        "manifest": file_record(manifest_path),
        "lm_eval_version": actual_version,
        "intervention": public_condition_metadata(setup),
        "task": args.task,
        "samples_slimmed": args.task == "gsm8k",
        "results": result.get("results"),
        "versions": result.get("versions"),
        "n-samples": result.get("n-samples"),
        "samples": samples,
    }
    output = (
        Path(args.out)
        if args.out
        else tag_dir(args.model) / f"x88_{args.task}_{args.condition}.json"
    )
    atomic_json(output, payload)
    print(json.dumps(result.get("results"), default=str, indent=1))


if __name__ == "__main__":
    main()
