from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from edit_rank1 import RankOneDownProjection


def test_rank_one_delta_matches_outer_product() -> None:
    torch.manual_seed(1)
    base = nn.Linear(5, 3, bias=False)
    q = torch.tensor([1.0, 2.0, -1.0])
    wrapped = RankOneDownProjection(base, q, scale=0.7)
    wrapped.right.data.copy_(torch.tensor([0.5, -0.2, 0.1, 0.3, -0.4]))
    x = torch.randn(2, 4, 5)
    expected = base(x) + 0.7 * torch.einsum(
        "...i,oi->...o", x, torch.outer(wrapped.q_unit, wrapped.right)
    )
    torch.testing.assert_close(wrapped(x), expected)


def test_zero_right_factor_is_identity() -> None:
    base = nn.Linear(4, 2, bias=False)
    wrapped = RankOneDownProjection(base, torch.tensor([3.0, 4.0]))
    x = torch.randn(3, 4)
    torch.testing.assert_close(wrapped(x), base(x))

