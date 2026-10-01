"""Shared contracts for x88: eight-model transfer of the deep-dose edits.

Exploratory extension of x87 (no pre-registration, descriptive only).
Model-agnostic mechanics (boundary hook, rank-one wrapper, bootstrap) are
imported from x87_common; everything model-specific lives in the registry
and the x88 manifest.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import torch

from x87_common import (  # noqa: F401  (re-exported for x88 scripts)
    ExternalBoundaryHook,
    MODEL_ID as QWEN_MODEL_ID,
    _decoder_layers,
    atomic_json,
    consensus_x86_vectors,
    file_record,
    format_user_chat,
    install_rank_one_qwen,
    paired_arrays,
    paired_bootstrap,
    read_json,
    sha256_file,
    sha256_text,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
RUN = DATA / "runs" / "x88"
RESULTS = DATA / "results"

# The x66 campaign tags minus smollm3 (near-zero refusal) and minus qwen
# (Qwen3.5-4B, imported from x87 rather than rerun).
MODELS: dict[str, dict[str, Any]] = {
    "qwen08b": {"model_id": "Qwen/Qwen3.5-0.8B", "trust_remote_code": True},
    "qwen2b": {"model_id": "Qwen/Qwen3.5-2B", "trust_remote_code": True},
    "qwen9b": {"model_id": "Qwen/Qwen3.5-9B", "trust_remote_code": True},
    "qwen27b": {"model_id": "Qwen/Qwen3.5-27B", "trust_remote_code": True},
    "llama": {"model_id": "unsloth/Llama-3.2-3B-Instruct", "trust_remote_code": True},
    "gemma": {"model_id": "google/gemma-4-e2b-it", "trust_remote_code": True},
    # Phi-4-mini ships stale custom code (modeling_phi3.py) that breaks on
    # transformers 5.x; the campaign always ran it on the native impl.
    "phi": {"model_id": "microsoft/Phi-4-mini-instruct", "trust_remote_code": False},
}
TAGS = tuple(MODELS)
IMPORTED_TAG = "qwen"  # x87 rows, never rerun here

CONDITIONS = ("base", "x86_m150", "x75_m6", "x75_m3")
DOSES = {"x86_m150": -1.5, "x75_m6": -6.0, "x75_m3": -3.0}

SEED = 8800
MAX_NEW_TOKENS = 1024
ALPHA = 0.05
BOOTSTRAP = 20_000

# Frozen relative-depth rule for the boundary-write hook: recovers block 29
# (state 30) on Qwen3.5-4B, where x85/x86 located the shared boundary shift.
HOOK_NUMERATOR, HOOK_DENOMINATOR = 29, 33


def hook_block_for(n_blocks: int) -> int:
    return round(HOOK_NUMERATOR * n_blocks / HOOK_DENOMINATOR)


def stable_seed(*parts: object) -> int:
    payload = "x88|" + "|".join(str(value) for value in (SEED, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big") & 0x7FFFFFFF


def tag_dir(tag: str) -> Path:
    if tag not in MODELS:
        raise ValueError(f"unknown x88 tag {tag}")
    return RUN / tag


def load_tokenizer(source: str, tag: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source, trust_remote_code=MODELS[tag]["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(source: str, tag: str) -> torch.nn.Module:
    from lib.edit_common import load_causal_lm as lib_load

    os.environ["X63_TRUST_REMOTE_CODE"] = (
        "1" if MODELS[tag]["trust_remote_code"] else "0"
    )
    return lib_load(source)


def n_blocks_of(model: torch.nn.Module) -> int:
    return int(model.config.get_text_config().num_hidden_layers)


def assert_boundary(tokenizer, rendered: str, expected_ids: list[int], tag: str) -> None:
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if len(ids) < 2 or list(ids[-2:]) != list(expected_ids):
        raise RuntimeError(
            f"{tag}: rendered chat does not end with the audited boundary ids "
            f"{expected_ids}; got {ids[-2:]}"
        )


def validate_x88_adapter(adapter: dict, tag: str) -> None:
    if adapter.get("model_id") != MODELS[tag]["model_id"]:
        raise ValueError(
            f"adapter model_id {adapter.get('model_id')!r} is not {tag}'s model"
        )
    metadata = adapter.get("metadata", {})
    if metadata.get("probe_kind") != "true":
        raise ValueError("x88 requires the true, not permuted, probe adapter")
    if metadata.get("probe_arm") != "refusal_weighted":
        raise ValueError("x88 is frozen to the refusal_weighted arm")


def setup_condition(
    model: torch.nn.Module,
    tag: str,
    condition: str,
    *,
    directions: str | Path | None = None,
    adapter_path: str | Path | None = None,
) -> dict[str, Any]:
    """Install exactly one intervention and return its auditable metadata."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown x88 condition {condition}")
    if condition == "base":
        return {"condition": condition, "model_tag": tag, "kind": "identity", "dose": 0.0}
    if condition == "x86_m150":
        if not directions:
            raise ValueError("x86_m150 requires a per-model directions artifact")
        minus1, tbg = consensus_x86_vectors(directions)
        block = hook_block_for(n_blocks_of(model))
        hook = ExternalBoundaryHook(DOSES[condition], minus1, tbg)
        handle = _decoder_layers(model)[block].register_forward_hook(hook)
        return {
            "condition": condition,
            "model_tag": tag,
            "kind": "x88_consensus_boundary_write",
            "dose": DOSES[condition],
            "positions": ["tbg_minus1", "tbg"],
            "hook_block": block,
            "state_index": block + 1,
            "directions": file_record(directions),
            "hook": hook,
            "handle": handle,
        }
    if not adapter_path:
        raise ValueError(f"{condition} requires a per-model adapter")
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    validate_x88_adapter(adapter, tag)
    trained_source = adapter.get("metadata", {}).get("model_source")
    loaded_source = getattr(model.config, "_name_or_path", None)
    if (
        not trained_source
        or not loaded_source
        or Path(trained_source).name != Path(loaded_source).name
    ):
        raise ValueError(
            f"adapter/model snapshot mismatch: {trained_source!r} vs {loaded_source!r}"
        )
    wrapper = install_rank_one_qwen(
        model,
        int(adapter["block_index"]),
        adapter["q_unit"].float(),
        DOSES[condition],
    )
    wrapper.right.data.copy_(adapter["right"].float().to(wrapper.right.device))
    wrapper.right.requires_grad_(False)
    return {
        "condition": condition,
        "model_tag": tag,
        "kind": "x88_refusal_weighted_rank1",
        "dose": DOSES[condition],
        "state_index": int(adapter["state_index"]),
        "block_index": int(adapter["block_index"]),
        "adapter": file_record(adapter_path),
        "wrapper": wrapper,
    }


def public_condition_metadata(setup: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in setup.items() if k not in ("hook", "handle", "wrapper")}


def default_paths(tag: str) -> dict[str, Path]:
    d = tag_dir(tag)
    return {
        "directions": d / "x88_directions.npz",
        "adapter": d / f"x75_{tag}_refusal_weighted_true.pt",
        "config": d / "x88_x75_config.json",
    }
