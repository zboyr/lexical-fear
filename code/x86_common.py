"""Shared frozen contracts for x86's held-out response-boundary intervention.

x86 rebuilds four fold-specific x85 boundary shifts (state 30 = output of
decoder block 29, positions ``</think>`` and the following ``"\n\n"``), and
tests whether signed activation writes of those shifts move new refusal
behavior on the strict x85 fold diagonal, against KL-matched null directions.

Loaders, hashing, chat rendering, and model loading are reused from the x85 →
x83 → x81 lineage so the checkpoint and template are identical to x85.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from x85_common import (  # noqa: F401  (re-exported for the stage scripts)
    DATA,
    MODEL_ID,
    RESULTS,
    atomic_json,
    file_record,
    format_user_chat,
    load_causal_lm,
    read_json,
    sha256_file,
    sha256_text,
)

RUN = DATA / "runs" / "x86"

# --- frozen scalar contracts -------------------------------------------------
N_RESPONSES = 10          # target responses per (cell, condition)
BENIGN_RESPONSES = 5      # benign responses per (prompt, condition)
BOUNDARY_POSITIONS = ("tbg_minus1", "tbg")
INTERVENTION_STATE = 30   # HF hidden-state index; output of decoder block 29
HOOK_BLOCK = INTERVENTION_STATE - 1  # decoder block whose output is state 30

DOSE_GRID = (0.125, 0.25, 0.5, 1.0)
KL_CAP = 0.10
KL_SLACK = 1e-6

# Seeds, all frozen by the plan.
TARGET_GRAPH_SEED = 8604
BENIGN_CALIB_SEED = 8600
BENIGN_EVAL_SEED = 8601
RANDOM_NULL_SEED = 8602
PERMUTATION_NULL_SEED = 8603
DOSE_ZERO_AUDIT_SEED = 8605
GENERATION_SEED_BASE = 8600

STRATA = ("floor", "low", "high")
PROMPTS_PER_FOLD_STRATUM = 8
WORDS_PER_FOLD = 6
PROMPT_DEGREE = 3
WORD_DEGREE_IN_BLOCK = 4

CODE_FILES = (
    "x86_common.py",
    "x86_build_manifest.py",
    "x86_calibrate_kl.py",
    "x86_generate.py",
    "x86_judge.py",
    "x86_analyze.py",
)


def cohort_path() -> Path:
    return RUN / "x86_manifest.json"


def _derived_seed(*parts) -> int:
    payload = "x86|" + "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def stable_seed(*parts, base: int = GENERATION_SEED_BASE) -> int:
    """32-bit torch-safe seed derived from an ordered tuple of string parts."""
    payload = "x86|" + "|".join(str(part) for part in (base, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big") & 0x7FFFFFFF


def code_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {name: sha256_file(here / name) for name in CODE_FILES if (here / name).exists()}


# --- fold-specific directions ------------------------------------------------
def fold_direction(
    deltas: np.ndarray, prompt_folds: Iterable[int], word_folds: Iterable[int], fold: int
) -> np.ndarray:
    """Mean H-minus-N vector over cells whose prompt and word are BOTH out of `fold`."""
    p_mask = np.asarray(list(prompt_folds), dtype=int) != fold
    w_mask = np.asarray(list(word_folds), dtype=int) != fold
    if not p_mask.any() or not w_mask.any():
        raise ValueError("fold_direction training set is empty")
    return deltas[np.ix_(p_mask, w_mask)].mean(axis=(0, 1))


def orthogonal_random_null(
    true_vector: np.ndarray, fold: int, position: str, base_seed: int = RANDOM_NULL_SEED
) -> np.ndarray:
    """Gaussian vector orthogonalized against `true_vector` and scaled to its norm."""
    true = np.asarray(true_vector, dtype=np.float64)
    true_norm = float(np.linalg.norm(true))
    if true_norm == 0.0:
        raise ValueError("cannot build a null against a zero direction")
    rng = np.random.default_rng(_derived_seed(base_seed, fold, position))
    gaussian = rng.standard_normal(true.shape)
    gaussian = gaussian - (gaussian @ true) / (true @ true) * true
    norm = float(np.linalg.norm(gaussian))
    if norm == 0.0:
        raise ValueError("degenerate random null")
    return gaussian / norm * true_norm


def permutation_sign_null(
    deltas: np.ndarray,
    prompt_folds: Iterable[int],
    word_folds: Iterable[int],
    fold: int,
    prompt_signs: np.ndarray,
    word_signs: np.ndarray,
) -> np.ndarray:
    """Recompute the fold training mean after flipping each H-N pair by prompt×word sign.

    The result is scaled to the true fold-direction norm; its raw cosine to the
    true direction is a reported diagnostic, never an outcome-selected quantity.
    """
    p_folds = np.asarray(list(prompt_folds), dtype=int)
    w_folds = np.asarray(list(word_folds), dtype=int)
    p_mask = p_folds != fold
    w_mask = w_folds != fold
    signs = np.outer(
        np.asarray(prompt_signs, dtype=np.float64)[p_mask],
        np.asarray(word_signs, dtype=np.float64)[w_mask],
    )
    block = deltas[np.ix_(p_mask, w_mask)]
    flipped_mean = (signs[:, :, None] * block).mean(axis=(0, 1))
    true = fold_direction(deltas, p_folds, w_folds, fold)
    true_norm = float(np.linalg.norm(true))
    flipped_norm = float(np.linalg.norm(flipped_mean))
    if flipped_norm == 0.0:
        raise ValueError("degenerate permutation null")
    return flipped_mean / flipped_norm * true_norm


def load_direction_vectors(manifest: dict) -> dict[str, np.ndarray]:
    """Load the x86-own fold direction npz named in the manifest, as float32."""
    path = manifest["directions_npz"]["path"]
    with np.load(path) as data:
        return {name: data[name].astype(np.float32) for name in data.files}


# --- target graph ------------------------------------------------------------
def _biregular_block(
    prompt_indices: list[int],
    word_indices: list[int],
    prompt_degree: int,
    word_degree: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """A simple (prompt_degree, word_degree)-biregular bipartite graph by rejection."""
    word_stubs = [w for w in word_indices for _ in range(word_degree)]
    prompt_stubs = [p for p in prompt_indices for _ in range(prompt_degree)]
    if len(word_stubs) != len(prompt_stubs):
        raise ValueError("stub counts disagree; degrees do not balance the block")
    for _ in range(100_000):
        order = rng.permutation(len(word_stubs))
        edges: set[tuple[int, int]] = set()
        ok = True
        for position, prompt in enumerate(prompt_stubs):
            word = word_stubs[int(order[position])]
            edge = (int(prompt), int(word))
            if edge in edges:
                ok = False
                break
            edges.add(edge)
        if ok:
            return sorted(edges)
    raise RuntimeError("failed to draw a simple biregular block")


def target_graph(fold_manifest: dict, seed: int = TARGET_GRAPH_SEED) -> list[dict]:
    """Strict fold-diagonal target cells: per fold and stratum, an (3,4)-biregular block."""
    prompts = fold_manifest["prompts"]
    words = fold_manifest["words"]
    words_by_fold: dict[int, list[int]] = {}
    for word in words:
        words_by_fold.setdefault(int(word["word_fold"]), []).append(int(word["word_index"]))
    edges: list[dict] = []
    for fold in range(4):
        fold_words = sorted(words_by_fold.get(fold, []))
        if len(fold_words) != WORDS_PER_FOLD:
            raise ValueError(f"fold {fold} has {len(fold_words)} words, expected {WORDS_PER_FOLD}")
        for stratum_index, stratum in enumerate(STRATA):
            block_prompts = sorted(
                int(row["prompt_index"])
                for row in prompts
                if int(row["prompt_fold"]) == fold and row["stratum"] == stratum
            )
            if len(block_prompts) != PROMPTS_PER_FOLD_STRATUM:
                raise ValueError(
                    f"fold {fold} stratum {stratum} has {len(block_prompts)} prompts"
                )
            rng = np.random.default_rng([seed, fold, stratum_index])
            block = _biregular_block(
                block_prompts, fold_words, PROMPT_DEGREE, WORD_DEGREE_IN_BLOCK, rng
            )
            for prompt_index, word_index in block:
                edges.append(
                    {
                        "prompt_index": prompt_index,
                        "word_index": word_index,
                        "fold": fold,
                        "stratum": stratum,
                    }
                )
    return sorted(edges, key=lambda row: (row["fold"], row["stratum"], row["prompt_index"], row["word_index"]))


# --- conditions --------------------------------------------------------------
def _both() -> list[str]:
    return list(BOUNDARY_POSITIONS)


def target_conditions() -> list[dict]:
    """The fourteen frozen target conditions; H uses negative dose, N positive."""
    rows: list[dict] = []
    specs = (
        ("0", "zero", "true", _both()),
        ("both_half", "half", "true", _both()),
        ("both_full", "full", "true", _both()),
        ("minus1_full", "full", "true", ["tbg_minus1"]),
        ("tbg_full", "full", "true", ["tbg"]),
        ("random_full", "null_full", "random", _both()),
        ("permutation_full", "null_full", "permutation", _both()),
    )
    for arm, sign in (("H", -1.0), ("N", +1.0)):
        for suffix, dose_kind, family, positions in specs:
            rows.append(
                {
                    "condition": f"{arm}_{suffix}",
                    "arm": arm,
                    "sign": sign,
                    "dose_kind": dose_kind,
                    "family": family,
                    "positions": positions,
                }
            )
    return rows


def benign_conditions() -> list[dict]:
    """The five frozen benign conditions, all at positive sign."""
    specs = (
        ("0", "zero", "true", _both()),
        ("both_half", "half", "true", _both()),
        ("both_full", "full", "true", _both()),
        ("random_full", "null_full", "random", _both()),
        ("permutation_full", "null_full", "permutation", _both()),
    )
    return [
        {
            "condition": f"B_{suffix}",
            "arm": "B",
            "sign": +1.0,
            "dose_kind": dose_kind,
            "family": family,
            "positions": positions,
        }
        for suffix, dose_kind, family, positions in specs
    ]


# --- activation hook ---------------------------------------------------------
class BoundaryHook:
    """Prefill-only two-position residual write at frozen absolute indices.

    Acts only when it sees the full prompt prefill (``sequence_length > 1``) and
    only at the two audited absolute positions. Dose zero is an exact identity
    (returns the same tensor object). Cached one-token decode steps and every
    other position, layer, or tensor are untouched.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.signed_dose = 0.0
        self.minus1_idx = -1
        self.tbg_idx = -1
        self.vec_minus1: torch.Tensor | None = None
        self.vec_tbg: torch.Tensor | None = None
        self.reset_counters()

    def reset_counters(self) -> None:
        self.prefill_calls = 0
        self.decode_calls = 0
        self.writes_minus1 = 0
        self.writes_tbg = 0

    def configure(
        self,
        signed_dose: float,
        minus1_idx: int,
        tbg_idx: int,
        vec_minus1: torch.Tensor | None,
        vec_tbg: torch.Tensor | None,
    ) -> None:
        self.signed_dose = float(signed_dose)
        self.minus1_idx = int(minus1_idx)
        self.tbg_idx = int(tbg_idx)
        self.vec_minus1 = vec_minus1
        self.vec_tbg = vec_tbg

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        hidden = output
        seq_len = int(hidden.shape[1])
        if seq_len <= 1:
            self.decode_calls += 1
            return output
        self.prefill_calls += 1
        if self.signed_dose == 0.0:
            return output  # exact identity: same object, no copy
        edited = hidden.clone()
        if self.vec_minus1 is not None:
            vector = self.vec_minus1.to(dtype=edited.dtype, device=edited.device)
            edited[:, self.minus1_idx, :] = edited[:, self.minus1_idx, :] + self.signed_dose * vector
            self.writes_minus1 += 1
        if self.vec_tbg is not None:
            vector = self.vec_tbg.to(dtype=edited.dtype, device=edited.device)
            edited[:, self.tbg_idx, :] = edited[:, self.tbg_idx, :] + self.signed_dose * vector
            self.writes_tbg += 1
        return edited


def _decoder_layers(model) -> torch.nn.ModuleList:
    n_layers = int(model.config.num_hidden_layers)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == n_layers:
            return module
    raise RuntimeError("could not locate the decoder ModuleList for the hook")


def install_hook(model, hook: BoundaryHook):
    """Register the boundary hook on decoder block 29's output (state 30)."""
    block = _decoder_layers(model)[HOOK_BLOCK]

    def forward_hook(module, args, output):
        if isinstance(output, tuple):
            new_hidden = hook(module, args, output[0])
            if new_hidden is output[0]:
                return output
            return (new_hidden,) + tuple(output[1:])
        return hook(module, args, output)

    return block.register_forward_hook(forward_hook)
