from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x75_analyze import holm_adjust, symmetric_effect  # noqa: E402
from x75_dose_kl import calibrate_target, reference_anchors  # noqa: E402
from x75_train_weighted_probe import fit_logistic  # noqa: E402
from x75_weighting import (  # noqa: E402
    binary_entropy,
    normalize_weights,
    raw_weights,
    subgroup_masks,
    weighted_mean_std,
)


def test_x75_weight_formulas_and_endpoints() -> None:
    rate = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    entropy = binary_entropy(rate)
    assert entropy[0] == 0 and entropy[-1] == 0 and entropy[2] == 1
    boundary = raw_weights(rate, "boundary")
    safe = raw_weights(rate, "safe_boundary")
    refusal = raw_weights(rate, "refusal_weighted")
    torch.testing.assert_close(boundary, entropy)
    torch.testing.assert_close(safe, (1 - rate) * entropy)
    torch.testing.assert_close(refusal, rate)
    assert safe[0] == 0 and safe[-1] == 0
    assert refusal[0] == 0 and refusal[-1] == 1
    normalized = normalize_weights(boundary)
    torch.testing.assert_close(normalized.mean(), torch.tensor(1.0))
    assert normalized[0] == 0 and normalized[-1] == 0


def test_zero_weight_rows_do_not_affect_standardization() -> None:
    states = torch.tensor([[1000.0, -1000.0], [2.0, 4.0], [-900.0, 800.0]])
    weight = torch.tensor([0.0, 1.0, 0.0])
    mean, std = weighted_mean_std(states, weight)
    torch.testing.assert_close(mean, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(std, torch.tensor([1e-4, 1e-4]))


def test_x75_subgroups_partition_rates() -> None:
    rates = np.asarray([0.0, 0.1, 0.5, 0.6, 0.9, 1.0])
    masks = subgroup_masks(rates)
    assert masks["mixed"].sum() == 4
    assert masks["low_refusal_mixed"].sum() == 2
    assert masks["high_refusal_mixed"].sum() == 2
    assert masks["stable_comply"].sum() == 1
    assert masks["stable_refuse"].sum() == 1


def test_kl_calibration_matches_quadratic_curve() -> None:
    result = calibrate_target(
        lambda dose: 0.01 * dose * dose,
        reference_dose=1.0,
        target_kl=0.0225,
        absolute_tolerance=1e-8,
        relative_tolerance=0.0,
        max_iterations=30,
        max_abs_dose=8.0,
        bracket_growth=2.0,
        monotonic_slack=1e-12,
    )
    assert result["status"] == "matched"
    assert abs(result["actual_dose"] - 1.5) < 1e-5
    assert abs(result["achieved_kl"] - 0.0225) < 1e-8


def test_kl_calibration_fails_nonmonotone_bracket() -> None:
    result = calibrate_target(
        lambda dose: 0.01 if abs(dose) <= 1 else 0.005,
        reference_dose=-1.0,
        target_kl=0.03,
        absolute_tolerance=1e-6,
        relative_tolerance=0.0,
        max_iterations=5,
        max_abs_dose=4.0,
        bracket_growth=2.0,
        monotonic_slack=0.0,
    )
    assert result["status"] == "non_monotone_bracket"


def test_reference_anchors_and_primary_helpers() -> None:
    obj = {"measurements": [{"dose": -1.0, "kl_full_mean": 0.2}, {"dose": 1.0, "kl_full_mean": 0.3}]}
    anchors = reference_anchors(obj, (-1.0, 1.0))
    assert anchors == [
        {"reference_dose": -1.0, "target_kl": 0.2},
        {"reference_dose": 1.0, "target_kl": 0.3},
    ]
    np.testing.assert_allclose(
        symmetric_effect(np.asarray([0.1, 0.2]), np.asarray([0.5, 0.8])),
        np.asarray([0.2, 0.3]),
    )
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_weighted_probe_fit_ignores_zero_weight_rows() -> None:
    x = torch.tensor([[-100.0], [-1.0], [1.0], [100.0]])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weight = torch.tensor([0.0, 2.0, 2.0, 0.0])
    probe_w, probe_b = fit_logistic(x, target, weight, l2=0.1, max_iter=50)
    assert float(probe_w) > 0
    assert abs(float(probe_b)) < 1e-3
