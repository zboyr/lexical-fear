"""Shared utilities for the Llama refusal-probe weight-edit experiment.

Shared by the x71 edit chain and the x70 capability evaluation, which is why
these two modules carry no x-prefix.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


DEFAULT_CONFIG = HERE.parents[1] / "data" / "runs" / "x71" / "x71_config.json"


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Frozen config of the edit experiment; x70 reads model_id + chat date."""
    target = Path(path) if path else DEFAULT_CONFIG
    return json.loads(target.read_text())


def atomic_json_dump(obj: Any, path: str | os.PathLike[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, target)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | os.PathLike[str], hash_contents: bool = True) -> dict[str, Any]:
    target = Path(path).resolve()
    stat = target.stat()
    record: dict[str, Any] = {
        "path": str(target),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_contents:
        record["sha256"] = sha256_file(target)
    return record


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_user_chat(tokenizer: Any, prompt: str, date_string: str) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "date_string": date_string,
    }
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    # Newer multimodal-wrapped text checkpoints (including Qwen3.5 in recent
    # transformers) put the text decoder one level deeper.  Resolve it by the
    # text-config layer count, while rejecting ambiguous ModuleLists.
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers = int(config.num_hidden_layers)
    matches = [
        module
        for _, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList)
        and len(module) == n_layers
        and all(hasattr(layer, "mlp") for layer in module)
    ]
    if len(matches) == 1:
        return matches[0]
    raise TypeError(f"Could not uniquely locate {n_layers} decoder layers; found {len(matches)}")


def load_causal_lm(model_id: str, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    from transformers import AutoModelForCausalLM

    # X63_TRUST_REMOTE_CODE=0 forces the native transformers implementation
    # when a repo ships stale custom code (extract_features_v2 convention;
    # needed for Phi-4-mini under transformers 5.x).
    trust = os.environ.get("X63_TRUST_REMOTE_CODE", "1") != "0"
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=trust,
        )
    except ValueError:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=trust,
        )
