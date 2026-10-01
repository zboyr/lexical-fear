import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from x87_common import (  # noqa: E402
    MODEL_ID,
    ExternalBoundaryHook,
    consensus_x86_vectors,
    paired_bootstrap,
    validate_x75_adapter,
    x86_full_dose,
)
from x87_score import parse_orbench_label  # noqa: E402


def test_external_boundary_hook_writes_only_last_two_prefill_positions():
    hook = ExternalBoundaryHook(-0.5, torch.tensor([2.0, 0.0]), torch.tensor([0.0, 4.0]))
    original = torch.zeros((2, 5, 2))
    edited = hook(None, (), original)
    assert torch.equal(original, torch.zeros_like(original))
    assert torch.equal(edited[:, :-2], torch.zeros((2, 3, 2)))
    assert torch.equal(edited[:, -2], torch.tensor([[-1.0, 0.0], [-1.0, 0.0]]))
    assert torch.equal(edited[:, -1], torch.tensor([[0.0, -2.0], [0.0, -2.0]]))
    assert hook.prefill_calls == 1

    decode = torch.ones((2, 1, 2))
    assert hook(None, (), decode) is decode
    assert hook.decode_calls == 1


def test_consensus_x86_vectors_averages_all_folds(tmp_path):
    payload = {}
    for fold in range(4):
        payload[f"true_f{fold}_tbg_minus1"] = np.asarray([fold, 2 * fold], dtype=np.float32)
        payload[f"true_f{fold}_tbg"] = np.asarray([10 + fold, 20 + fold], dtype=np.float32)
    path = tmp_path / "directions.npz"
    np.savez(path, **payload)
    minus1, tbg = consensus_x86_vectors(path)
    assert torch.allclose(minus1, torch.tensor([1.5, 3.0]))
    assert torch.allclose(tbg, torch.tensor([11.5, 21.5]))


def test_x86_full_dose_is_negative(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"status": "pass", "model_id": MODEL_ID, "admitted_a_star": 0.5}))
    assert x86_full_dose(path) == -0.5


def test_original_llama_x75_adapter_is_rejected():
    with pytest.raises(ValueError, match="not Qwen3.5-4B"):
        validate_x75_adapter(
            {"model_id": "unsloth/Llama-3.2-3B-Instruct", "metadata": {"probe_kind": "true"}}
        )


def test_qwen_adapter_from_another_probe_arm_is_rejected():
    with pytest.raises(ValueError, match="refusal_weighted"):
        validate_x75_adapter(
            {
                "model_id": MODEL_ID,
                "metadata": {"probe_kind": "true", "probe_arm": "boundary"},
            }
        )


def test_paired_bootstrap_reports_edited_minus_base():
    result = paired_bootstrap(
        [1, 1, 1, 1], [0, 0, 0, 0], alpha=0.05, seed=1, n_boot=100
    )
    assert result["mean_delta"] == -1.0
    assert result["one_sided_upper"] == -1.0
    assert result["one_sided_lower"] == -1.0


@pytest.mark.parametrize(
    ("text", "expected", "mode"),
    [
        ("Reason. [[direct_answer]]", "direct_answer", "bracketed"),
        ("Reason. [[direct_refusal]]", "direct_refusal", "bracketed"),
        ("Reason. [[indirect_refusal]]", "indirect_refusal", "bracketed"),
        ("direct_answer", "direct_answer", "bare_final_line"),
        ("Reason.\n\nindirect_refusal\n", "indirect_refusal", "bare_final_line"),
    ],
)
def test_parse_orbench_label(text, expected, mode):
    assert parse_orbench_label(text) == (expected, mode)


@pytest.mark.parametrize(
    "text",
    [
        "[[direct_answer]] and [[direct_refusal]]",
        "no label here",
        "direct_answer or indirect_refusal",
        "Reason mentioning direct_answer in prose.",
    ],
)
def test_parse_orbench_label_fails_closed_on_ambiguous_output(text):
    with pytest.raises(ValueError):
        parse_orbench_label(text)
