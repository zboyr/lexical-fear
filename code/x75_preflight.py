"""Validate frozen x75 hashes, environment, offline cache, and task inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

from edit_common import atomic_json_dump, file_record, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-offline", action="store_true")
    parser.add_argument("--audit-model-cache", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--load-eval-data", action="store_true")
    return parser.parse_args()


def _walk_records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        else:
            for child in value.values():
                yield from _walk_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_records(child)


def _versions() -> dict:
    packages = (
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "huggingface-hub",
        "lm-eval",
        "numpy",
        "scipy",
        "scikit-learn",
    )
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _cache_audit(model_id: str) -> dict:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo_id=model_id, local_files_only=True))
    files = []
    total = 0
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            resolved = path.resolve()
            size = resolved.stat().st_size
            total += size
            files.append({
                "relative_path": str(path.relative_to(snapshot)),
                "resolved_path": str(resolved),
                "size_bytes": size,
            })
    if not files or total <= 0:
        raise RuntimeError("resolved model snapshot contains no files")
    return {"snapshot": str(snapshot), "n_files": len(files), "total_size_bytes": total, "files": files}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if not config.get("x75", {}).get("design_frozen"):
        raise ValueError("x75 config is not marked frozen")
    if args.require_offline and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is required")

    checked = []
    for record in _walk_records(config["x75"]["inherited_inputs"]):
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"frozen input hash mismatch: {path}")
        checked.append(str(path))
    for name, record in config["x75"]["code"].items():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"frozen code hash mismatch: {name}")

    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    versions = _versions()
    if versions["accelerate"] is None:
        raise RuntimeError("accelerate is an explicit x75 dependency")

    cache = _cache_audit(config["model_id"]) if args.audit_model_cache else None
    model_load = None
    if args.load_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        model_load = {
            "tokenizer_class": type(tokenizer).__name__,
            "model_class": type(model).__name__,
            "device": str(next(model.parameters()).device),
        }

    eval_data = None
    if args.load_eval_data:
        from datasets import load_dataset
        from lm_eval.tasks import TaskManager

        loaded = {
            "wikitext": len(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")),
            "gsm8k": len(load_dataset("openai/gsm8k", "main", split="test")),
            "ifeval": len(load_dataset("google/IFEval", split="train")),
            "humaneval": len(load_dataset("openai/openai_humaneval", split="test")),
        }
        manager = TaskManager()
        for task in ("ifeval", "humaneval"):
            manager.load_task_or_group(task)
        eval_data = {"datasets": loaded, "lm_eval_tasks": ["ifeval", "humaneval"]}

    result = {
        "schema_version": 1,
        "experiment": "x75",
        "status": "pass",
        "python": sys.executable,
        "versions": versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "config": file_record(config_path),
        "n_frozen_inputs_checked": len(checked),
        "model_cache": cache,
        "model_load": model_load,
        "eval_data": eval_data,
    }
    atomic_json_dump(result, args.out)
    print(json.dumps({key: result[key] for key in (
        "status", "python", "versions", "cuda_available", "n_frozen_inputs_checked"
    )}, indent=2))


if __name__ == "__main__":
    main()
