"""Contract tests for the x83 insertion rules, surface variants, and paired bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x83_analyze import holm, paired_bootstrap  # noqa: E402
from x83_common import insert_word, lexicon_words, middle_insertion_index, surface_variants  # noqa: E402


def test_insert_start_end_offsets():
    text = "make a plan for me."
    s, off = insert_word(text, "bomb", "S")
    assert s == "bomb make a plan for me." and s[off : off + 4] == "bomb"
    e, off = insert_word(text, "bomb", "E")
    assert e == "make a plan for me. bomb" and e[off : off + 4] == "bomb"


def test_insert_middle_at_nearest_whitespace():
    text = "aaaa bbbb cccc dddd"  # len 19, midpoint 9.5 -> whitespace at 9
    assert middle_insertion_index(text) == 9
    m, off = insert_word(text, "kill", "M")
    assert m == "aaaa bbbb kill cccc dddd" and m[off : off + 4] == "kill"
    assert m.replace(" kill", "", 1) == text  # only the word and one space were added


def test_insert_middle_no_whitespace_falls_back_to_end():
    m, off = insert_word("abcdef", "hack", "M")
    assert m == "abcdef hack" and off == 7


def test_lexicon_words_unique_lowercase():
    pairs = lexicon_words()
    words = [w for w, _ in pairs]
    assert len(words) == len(set(words)) and all(w == w.lower() and w.isalpha() for w in words)
    assert len(words) >= 50


def test_surface_variants():
    v = surface_variants("meth")
    assert v[:3] == ["meth", "Meth", "METH"] and " meth" in v and len(v) == 6


def test_paired_bootstrap_null_and_shift():
    rng = np.random.default_rng(0)
    base = rng.binomial(1, 0.4, size=(200, 10)).astype(np.int8)
    same = paired_bootstrap(base, base, 400, 1)
    assert abs(same.mean()) < 0.02  # response resampling is independent per condition, centred at 0
    treated = np.clip(base + rng.binomial(1, 0.5, size=(200, 10)), 0, 1).astype(np.int8)
    shift = paired_bootstrap(base, treated, 400, 2)
    point = float(treated.mean() - base.mean())
    assert abs(shift.mean() - point) < 0.02 and np.quantile(shift, 0.025) > 0


def test_holm_monotone():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
