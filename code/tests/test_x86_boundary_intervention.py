"""Contract tests for x86's held-out boundary intervention."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x86_analyze import causal_metrics, repeated_fourgram_dominated  # noqa: E402
from x86_calibrate_kl import admit_true_dose, monotone  # noqa: E402
from x86_common import (  # noqa: E402
    BOUNDARY_POSITIONS,
    BoundaryHook,
    fold_direction,
    orthogonal_random_null,
    target_conditions,
    target_graph,
)
from x86_generate import resolve_condition  # noqa: E402
from x86_judge import repair_labels  # noqa: E402


def synthetic_fold_manifest():
    prompts = []
    for fold in range(4):
        for stratum in ("floor", "low", "high"):
            for _ in range(8):
                prompts.append(
                    {
                        "prompt_index": len(prompts),
                        "prompt_fold": fold,
                        "stratum": stratum,
                    }
                )
    words = [
        {"word_index": index, "word_fold": index // 6} for index in range(24)
    ]
    return {"prompts": prompts, "words": words}


def test_target_graph_is_regular_and_stratified():
    edges = target_graph(synthetic_fold_manifest(), seed=8604)
    assert len(edges) == 288
    assert set(Counter(row["prompt_index"] for row in edges).values()) == {3}
    assert set(Counter(row["word_index"] for row in edges).values()) == {12}
    for word in range(24):
        assert Counter(row["stratum"] for row in edges if row["word_index"] == word) == {
            "floor": 4,
            "low": 4,
            "high": 4,
        }


def test_fold_direction_excludes_prompt_and_word_fold():
    deltas = np.zeros((8, 8, 2), dtype=float)
    prompt_folds = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    word_folds = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    deltas[:, :, 0] = np.arange(8)[:, None]
    deltas[:, :, 1] = np.arange(8)[None, :]
    got = fold_direction(deltas, prompt_folds, word_folds, fold=0)
    expected = deltas[np.ix_(prompt_folds != 0, word_folds != 0)].mean(axis=(0, 1))
    assert np.allclose(got, expected)


def test_null_is_orthogonal_and_norm_matched():
    true = np.asarray([1.0, 2.0, 3.0, 4.0])
    null = orthogonal_random_null(true, fold=2, position="tbg")
    assert abs(float(null @ true)) < 1e-10
    assert np.isclose(np.linalg.norm(null), np.linalg.norm(true))


def test_boundary_hook_writes_only_two_prefill_positions():
    hook = BoundaryHook()
    hook.enabled = True
    hook.configure(2.0, 3, 4, torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
    base = torch.zeros((2, 5, 3))
    edited = hook(None, None, base)
    assert torch.equal(base, torch.zeros_like(base))
    assert torch.all(edited[:, 3, 0] == 2) and torch.all(edited[:, 4, 1] == 2)
    assert hook.prefill_calls == 1 and hook.writes_minus1 == 1 and hook.writes_tbg == 1
    decode = torch.zeros((2, 1, 3))
    assert torch.equal(hook(None, None, decode), decode)
    assert hook.decode_calls == 1


def test_boundary_hook_dose_zero_is_exact_identity():
    hook = BoundaryHook()
    hook.enabled = True
    hook.configure(0.0, 1, 2, torch.ones(4), torch.ones(4))
    base = torch.randn(1, 3, 4)
    output = hook(None, None, base)
    assert output is base and torch.equal(output, base)
    assert hook.writes_minus1 == 0 and hook.writes_tbg == 0


def test_dose_admission_requires_both_signs_and_folds():
    records = {"minus": [], "plus": []}
    for magnitude, kl in zip((0.125, 0.25, 0.5, 1.0), (0.01, 0.02, 0.08, 0.12)):
        for sign in records:
            records[sign].append(
                {
                    "magnitude": magnitude,
                    "mean_kl": kl,
                    "fold_mean_kl": {str(fold): kl for fold in range(4)},
                }
            )
    assert admit_true_dose(records) == 0.5
    assert monotone(records["minus"])


def test_condition_resolution_uses_calibrated_doses():
    manifest = {
        "target_conditions": target_conditions(),
        "benign_conditions": [],
    }
    calibration = {
        "admitted_a_star": 0.5,
        "null_calibration": {
            "random": {
                "minus": {"status": "pass", "matched_signed_dose": -0.7},
                "plus": {"status": "pass", "matched_signed_dose": 0.8},
            }
        },
    }
    assert resolve_condition(manifest, calibration, "H_both_half")[2] == -0.25
    assert resolve_condition(manifest, calibration, "N_both_full")[2] == 0.5
    assert resolve_condition(manifest, calibration, "H_random_full")[2] == -0.7


def test_generic_judge_repair_uses_method_field():
    parsed = {"0": True, "1": False}
    refusal, _ = repair_labels(parsed, "", 2, "refusal2")
    complied, _ = repair_labels(parsed, "", 2, "comply2")
    assert refusal == {"refusal": [True, False]}
    assert complied == {"complied": [True, False]}


def test_causal_metrics_and_repetition_audit():
    n = 6
    rates = {
        "H_0": np.full(n, 0.8),
        "H_both_half": np.full(n, 0.7),
        "H_both_full": np.full(n, 0.6),
        "H_minus1_full": np.full(n, 0.7),
        "H_tbg_full": np.full(n, 0.7),
        "H_random_full": np.full(n, 0.79),
        "H_permutation_full": np.full(n, 0.8),
        "N_0": np.full(n, 0.2),
        "N_both_half": np.full(n, 0.3),
        "N_both_full": np.full(n, 0.4),
        "N_minus1_full": np.full(n, 0.3),
        "N_tbg_full": np.full(n, 0.3),
        "N_random_full": np.full(n, 0.21),
        "N_permutation_full": np.full(n, 0.2),
    }
    result = causal_metrics(rates, np.ones(n))
    assert np.isclose(result["suppress_H_full"], 0.2)
    assert np.isclose(result["induce_N_full"], 0.2)
    assert np.isclose(result["bidirectional_full"], 0.2)
    assert result["true_minus_random"] > 0
    assert repeated_fourgram_dominated("a a a a a a a a a a a a")
    assert not repeated_fourgram_dominated("a b c d e f g h")
