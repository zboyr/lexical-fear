from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x81_analyze_de import CONTRASTS, point_contrasts  # noqa: E402
from x81_build_arms_de import D_NOTE, build_prompts, word_position  # noqa: E402


def test_word_position_start_middle_end():
    text = "harm the assistant now."
    assert word_position(text, 0, 4) == "start"          # "harm"
    assert word_position(text, 9, 18) == "middle"         # "assistant"
    tail = "do it without help"
    assert word_position(tail, tail.index("help"), len(tail)) == "end"


def test_middle_arms_keep_word_in_place():
    original = "As an AI assistant, do the task."
    start, end = original.index("assistant"), original.index("assistant") + len("assistant")
    prompts = build_prompts(original, "assistant", start, end, "middle")
    assert prompts["P"] == original
    assert prompts["D"] == f"{D_NOTE}\n\n{original}"
    assert prompts["D"].endswith(original) and "assistant" in prompts["D"]
    # D never names the target word.
    assert "assistant" not in prompts["D"][: len(D_NOTE)]
    # E names the word once in the note and keeps the in-place occurrence.
    assert prompts["E"].startswith('The word "assistant" was accidentally inserted')
    assert prompts["E"].endswith(original)
    assert prompts["E"].count("assistant") == 2


def test_end_arm_moves_word_after_note_exactly_once():
    original = "Do it without any help"
    start, end = original.index("help"), len(original)
    prompts = build_prompts(original, "help", start, end, "end")
    # D keeps the word at the end in place.
    assert prompts["D"].rstrip().endswith("help")
    # E removes the inline occurrence and re-presents the word once, after the note.
    assert prompts["E"].count("help") == 1
    assert prompts["E"].rstrip().endswith("Ignore it: help")
    assert "A word will appear below" in prompts["E"]


def test_contrast_signs_positive_means_refusal_removed():
    # conditions order O, P, D, E ; two prompts, ten responses.
    labels = np.zeros((2, 4, 10), dtype=np.int8)
    labels[:, 1, :] = 1          # P refuses always
    labels[:, 2, :5] = 1         # D refuses half
    labels[:, 3, :] = 0          # E never refuses
    points = point_contrasts(labels)
    # delta_D = r_P - r_D = 1.0 - 0.5 = +0.5 (D removed refusal vs P)
    assert abs(points["delta_D"] - 0.5) < 1e-9
    # delta_E = r_P - r_E = 1.0 - 0.0 = +1.0
    assert abs(points["delta_E"] - 1.0) < 1e-9
    # d_vs_e = r_D - r_E = 0.5
    assert abs(points["d_vs_e"] - 0.5) < 1e-9


def test_contrasts_are_wired_over_expected_indices():
    names = {name for name, _, _ in CONTRASTS}
    assert {"delta_D", "delta_E", "d_vs_e", "run_era"} <= names
