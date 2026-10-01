"""Contract tests for x85's crossed design and vector decomposition."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x85_common import (  # noqa: E402
    BEHAVIOR_PROMPT_DEGREE,
    WORD_PAIRS,
    assign_prompt_folds,
    assign_word_folds,
    behavior_edges,
    locate_positions,
    select_prompts,
    two_way_residual,
    vector_anova,
)
from x85_analyze import sign_flip_p  # noqa: E402


def synthetic_units():
    rates = [0.0] * 40 + [0.1, 0.2, 0.3, 0.4] * 10 + [0.5, 0.6, 0.7, 0.8] * 10
    return [
        {"id": index, "historical_refusal_rate": rate, "prompts": {"O": f"prompt {index}"}}
        for index, rate in enumerate(rates)
    ]


def test_word_pairs_are_balanced_and_unique():
    categories = Counter(category for category, _, _ in WORD_PAIRS)
    harms = [harm for _, harm, _ in WORD_PAIRS]
    neutrals = [neutral for _, _, neutral in WORD_PAIRS]
    assert len(WORD_PAIRS) == 24
    assert set(categories.values()) == {3} and len(categories) == 8
    assert len(set(harms)) == len(harms) and len(set(neutrals)) == len(neutrals)
    assert not set(harms) & set(neutrals)


def test_prompt_selection_and_folds_are_stratum_balanced():
    selected = select_prompts(synthetic_units(), 85)
    assert len(selected) == 96 and len({row["id"] for row in selected}) == 96
    folds = assign_prompt_folds(selected, 86)
    assert Counter(folds.values()) == Counter({0: 24, 1: 24, 2: 24, 3: 24})
    for fold in range(4):
        rates = [row["historical_refusal_rate"] for row in selected if folds[row["id"]] == fold]
        assert sum(rate == 0 for rate in rates) == 8
        assert sum(0.1 <= rate <= 0.4 for rate in rates) == 8
        assert sum(0.5 <= rate <= 0.8 for rate in rates) == 8


def test_word_folds_and_behavior_graph_are_regular():
    folds = assign_word_folds(87)
    assert Counter(folds.values()) == Counter({0: 6, 1: 6, 2: 6, 3: 6})
    for category in {row[0] for row in WORD_PAIRS}:
        category_folds = [folds[harm] for cat, harm, _ in WORD_PAIRS if cat == category]
        assert len(set(category_folds)) == 3
    edges = behavior_edges(range(96), 88)
    assert len(edges) == 96 * BEHAVIOR_PROMPT_DEGREE
    assert set(Counter(prompt for prompt, _ in edges).values()) == {6}
    assert set(Counter(word for _, word in edges).values()) == {24}


def test_vector_anova_reconstructs_and_has_zero_margins():
    rng = np.random.default_rng(0)
    deltas = rng.normal(size=(7, 5, 11))
    parts = vector_anova(deltas)
    reconstructed = (
        parts["grand"]
        + parts["prompt"][:, None, :]
        + parts["word"][None, :, :]
        + parts["interaction"]
    )
    assert np.allclose(reconstructed, deltas)
    assert np.allclose(parts["prompt"].mean(axis=0), 0)
    assert np.allclose(parts["word"].mean(axis=0), 0)
    assert np.allclose(parts["interaction"].mean(axis=0), 0)
    assert np.allclose(parts["interaction"].mean(axis=1), 0)


def test_two_way_residual_removes_additive_prompt_and_word_effects():
    edges = sorted(behavior_edges(range(96), 88))
    p = np.asarray([prompt for prompt, _ in edges], dtype=int)
    word_order = {row[1]: index for index, row in enumerate(WORD_PAIRS)}
    w = np.asarray([word_order[word] for _, word in edges], dtype=int)
    values = 3.0 + 0.2 * p - 0.7 * w
    residual = two_way_residual(values, p, w)
    assert np.max(np.abs(residual)) < 1e-10


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert text == "Xalpha bombYZabcd"
        return {
            "input_ids": [1, 2, 3, 99, 4, 5, 6, 7],
            "offset_mapping": [(0, 1), (1, 6), (6, 11), (0, 0), (12, 13), (13, 14), (14, 15), (15, 19)],
        }

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 99


def test_position_locator_uses_inserted_span_and_fixed_boundary_tokens():
    ids, positions = locate_positions(
        FakeTokenizer(), "Xalpha bombYZabcd", "alpha bomb", (6, 10)
    )
    assert ids[-2:] == [6, 7]
    assert positions == {
        "instruction_last": 2,
        "user_end": 3,
        "tbg_minus1": 6,
        "tbg": 7,
    }


def test_sign_flip_detects_a_consistent_shared_shift():
    values = {
        "tbg_minus1": np.ones(128),
        "tbg": np.ones(128) * 0.8,
    }
    result = sign_flip_p(values, n_randomization=999, seed=0)
    assert result["tbg_minus1"] <= 0.01 and result["tbg"] <= 0.01
