from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x81_analyze import point_statistics, two_stage_bootstrap  # noqa: E402
from x81_build_arms import admit_prompt  # noqa: E402
from x81_common import replace_span, stable_seed  # noqa: E402
from x81_judge_semantics import exact_decision  # noqa: E402


def candidate(rank: int, surface: str, probability: float, decision: str = "SAME") -> dict:
    return {
        "rank": rank,
        "token_id": rank,
        "decoded": surface,
        "surface": surface,
        "probability": probability,
        "lexical_candidate": surface == "_" or surface.isalpha(),
        "is_original_surface": surface == "bomb",
        "is_blank": surface == "_",
        "decision": decision,
    }


def neutral_feature(word: str) -> dict:
    return {
        "word": word,
        "document_frequency": 12,
        "full_coef": 0.0,
        "fold_coefs": [0.0] * 5,
        "univariate_delta_r": 0.01,
    }


def admitted_row(top10_mass: float = 0.96) -> dict:
    candidates = [
        candidate(1, "item", 0.40),
        candidate(2, "_", 0.30),
        candidate(3, "object", 0.26),
    ]
    candidates.extend(candidate(rank, f"bad{rank}", 0.0, "CHANGED") for rank in range(4, 11))
    return {
        "target_word": "bomb",
        "top10_mass": top10_mass,
        "candidates": candidates,
    }


def test_admission_uses_raw_same_mass_and_selects_highest_neutral() -> None:
    features = {word: neutral_feature(word) for word in ("item", "object")}
    accepted = admit_prompt(admitted_row(), {"bomb"}, features)
    assert accepted["admitted"]
    assert accepted["p_same"] == 0.96
    assert accepted["selected_c"]["surface"] == "item"

    # Same-judged raw mass below the 0.90 threshold is rejected; the retained
    # SAME mass is never renormalized to one.
    rejected = admitted_row()
    rejected["candidates"][2]["decision"] = "CHANGED"  # drop object (0.26) -> p_same 0.70
    result = admit_prompt(rejected, {"bomb"}, features)
    assert not result["admitted"]
    assert "same_mass_below_0.9" in result["rejection_reasons"]


def test_blank_no_longer_gates_admission() -> None:
    # Blank present-and-SAME was dropped as an admission gate: a prompt whose
    # blank candidate is CHANGED is still admitted when p_same and the neutral
    # replacement rule hold.
    row = admitted_row()
    row["candidates"][0]["probability"] = 0.60  # item (SAME, neutral)
    row["candidates"][1]["probability"] = 0.05  # blank "_"
    row["candidates"][2]["probability"] = 0.35  # object (SAME, neutral)
    row["candidates"][1]["decision"] = "CHANGED"  # blank "_" now CHANGED
    result = admit_prompt(row, {"bomb"}, {"item": neutral_feature("item")})
    assert result["admitted"]  # p_same = 0.60 + 0.35 = 0.95 >= 0.90
    assert not result["blank_same"]
    assert result["selected_c"]["surface"] == "item"


def test_deletion_normalizes_only_the_edit_seam() -> None:
    text = "Keep  two spaces around bomb  here."
    start = text.index("bomb")
    result = replace_span(text, start, start + len("bomb"), "")
    assert result == "Keep  two spaces around  here."
    assert result.startswith("Keep  two")


def test_semantic_parser_is_exact_and_conservative() -> None:
    assert exact_decision(" SAME \n") == "SAME"
    assert exact_decision("changed") == "CHANGED"
    assert exact_decision("SAME because it is equivalent") == "CHANGED"


def test_per_prompt_seeds_are_stable_and_condition_specific() -> None:
    assert stable_seed(17, "A") == stable_seed(17, "A")
    assert stable_seed(17, "A") != stable_seed(17, "B")
    assert stable_seed(17, "A") != stable_seed(18, "A")


def test_two_stage_bootstrap_and_endpoint_signs() -> None:
    labels = np.zeros((3, 4, 10), dtype=np.int8)
    labels[:, 0, :] = 1
    labels[:, 1, :5] = 1
    labels[:, 2, :7] = 1
    labels[:, 3, :2] = 1
    points = point_statistics(labels)
    assert np.isclose(points["delta_C"], 0.8)
    assert np.isclose(points["delta_A"], 0.5)
    assert np.isclose(points["mask_effect"], 0.2)
    assert np.isclose(points["replacement_vs_blank"], -0.3)
    first = two_stage_bootstrap(labels, 20, 123)
    second = two_stage_bootstrap(labels, 20, 123)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
