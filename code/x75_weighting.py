"""Pure weighting and statistics helpers for x75.

The target is always the ten-response refusal rate.  The arm changes only the
sample weight used for train-fitted standardization and binomial probe loss.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


ARMS = ("boundary", "safe_boundary", "refusal_weighted")
WEIGHT_FORMULAS = {
    "boundary": "H2(r)",
    "safe_boundary": "(1-r)*H2(r)",
    "refusal_weighted": "r",
}


def refusal_rate(count: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    """Return finite rates in [0, 1], rejecting malformed binomial counts."""
    count = count.float()
    total = total.float()
    if count.shape != total.shape or count.ndim != 1:
        raise ValueError("count and total must be same-shaped vectors")
    if not torch.isfinite(count).all() or not torch.isfinite(total).all():
        raise ValueError("count/total contain non-finite values")
    if (total <= 0).any() or (count < 0).any() or (count > total).any():
        raise ValueError("invalid binomial counts")
    return count / total


def binary_entropy(rate: torch.Tensor) -> torch.Tensor:
    """Binary entropy in bits with the exact endpoint convention H(0)=H(1)=0."""
    rate = rate.float()
    if not torch.isfinite(rate).all() or (rate < 0).any() or (rate > 1).any():
        raise ValueError("refusal rates must be finite and inside [0, 1]")
    result = torch.zeros_like(rate)
    interior = (rate > 0) & (rate < 1)
    p = rate[interior]
    result[interior] = -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))
    return result


def raw_weights(rate: torch.Tensor, arm: str) -> torch.Tensor:
    if arm not in ARMS:
        raise ValueError(f"unknown x75 arm {arm!r}; expected one of {ARMS}")
    entropy = binary_entropy(rate)
    if arm == "boundary":
        weight = entropy
    elif arm == "safe_boundary":
        weight = (1 - rate) * entropy
    else:
        weight = rate.float()
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError(f"invalid weights for arm {arm}")
    if float(weight.sum()) <= 0:
        raise ValueError(f"all weights are zero for arm {arm}")
    return weight


def normalize_weights(weight: torch.Tensor) -> torch.Tensor:
    """Normalize over all rows, preserving exact zero weights."""
    weight = weight.float()
    if weight.ndim != 1 or not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("weights must be a finite non-negative vector")
    mean = weight.mean()
    if not torch.isfinite(mean) or mean <= 0:
        raise ValueError("mean weight must be positive")
    result = weight / mean
    result[weight == 0] = 0
    return result


def weighted_mean_std(train: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-only feature statistics; zero-weight rows have exactly no effect."""
    x = train.float()
    weight = weight.to(x.device, dtype=torch.float32)
    if x.ndim != 2 or weight.ndim != 1 or x.shape[0] != weight.numel():
        raise ValueError("state matrix and weight vector shapes disagree")
    denom = weight.sum()
    if not torch.isfinite(denom) or denom <= 0:
        raise ValueError("weighted standardization has no positive-weight rows")
    mean = (x * weight[:, None]).sum(0) / denom
    variance = (weight[:, None] * (x - mean).square()).sum(0) / denom
    std = variance.clamp_min(0).sqrt().clamp_min(1e-4)
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("non-finite weighted feature statistics")
    return mean, std


def weighted_standardized(
    train: torch.Tensor,
    weight: torch.Tensor,
    *others: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    mean, std = weighted_mean_std(train, weight)
    transformed = [(train.float() - mean) / std]
    transformed.extend((other.float() - mean) / std for other in others)
    return mean, std, *transformed


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Mean weighted BCE; normalized weights make this comparable to x71 BCE."""
    if logits.shape != target.shape or target.shape != weight.shape:
        raise ValueError("logits, target, and weight shapes disagree")
    per_row = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (per_row * weight).mean()


def effective_sample_size(weight: torch.Tensor) -> float:
    weight = weight.double()
    denom = float(weight.square().sum())
    return 0.0 if denom == 0 else float(weight.sum().square() / denom)


def weight_audit(rate: torch.Tensor, arm: str) -> dict:
    raw = raw_weights(rate, arm)
    normalized = normalize_weights(raw)
    values, counts = torch.unique(rate, sorted=True, return_counts=True)
    by_rate = []
    for value, count in zip(values, counts):
        mask = rate == value
        by_rate.append(
            {
                "r": float(value),
                "n": int(count),
                "raw_weight": float(raw[mask][0]),
                "normalized_weight": float(normalized[mask][0]),
            }
        )
    return {
        "arm": arm,
        "formula": WEIGHT_FORMULAS[arm],
        "n": int(rate.numel()),
        "n_positive_weight": int((raw > 0).sum()),
        "n_zero_weight": int((raw == 0).sum()),
        "raw_mean": float(raw.mean()),
        "normalized_mean": float(normalized.mean()),
        "effective_sample_size": effective_sample_size(normalized),
        "by_rate": by_rate,
    }


def subgroup_name(rate: float) -> str:
    if math.isclose(rate, 0.0, abs_tol=1e-8):
        return "stable_comply"
    if math.isclose(rate, 1.0, abs_tol=1e-8):
        return "stable_refuse"
    if rate <= 0.5:
        return "low_refusal_mixed"
    return "high_refusal_mixed"


def subgroup_masks(rates: np.ndarray) -> dict[str, np.ndarray]:
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 1 or not np.isfinite(rates).all():
        raise ValueError("rates must be a finite vector")
    masks = {
        "all": np.ones(len(rates), dtype=bool),
        "mixed": (rates > 0) & (rates < 1),
        "low_refusal_mixed": (rates > 0) & (rates <= 0.5),
        "high_refusal_mixed": (rates > 0.5) & (rates < 1),
        "stable_comply": rates == 0,
        "stable_refuse": rates == 1,
    }
    if sum(int(masks[name].sum()) for name in (
        "low_refusal_mixed", "high_refusal_mixed", "stable_comply", "stable_refuse"
    )) != len(rates):
        raise ValueError("subgroup masks do not partition rates")
    return masks
