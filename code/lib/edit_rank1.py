"""Fixed-left-factor rank-one edit for a decoder MLP down projection."""

from __future__ import annotations

import torch
from torch import nn

from edit_common import decoder_layers


class RankOneDownProjection(nn.Module):
    """Wrap ``down_proj`` with delta W = scale * q_unit * right^T."""

    def __init__(self, base: nn.Module, q_unit: torch.Tensor, scale: float = 1.0):
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError("down projection must expose in_features/out_features")
        q = q_unit.detach().float().flatten()
        if q.numel() != int(base.out_features):
            raise ValueError(
                f"q has {q.numel()} elements but down_proj writes {base.out_features}"
            )
        q = q / q.norm().clamp_min(1e-12)
        base_device = base.weight.device
        q = q.to(base_device)
        self.base = base
        self.register_buffer("q_unit", q, persistent=True)
        self.right = nn.Parameter(
            torch.zeros(int(base.in_features), dtype=torch.float32, device=base_device)
        )
        self.scale = float(scale)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = self.base(hidden)
        gate = torch.matmul(hidden.float(), self.right)
        delta = gate.unsqueeze(-1) * self.q_unit
        return output + (self.scale * delta).to(output.dtype)


def install_rank_one_edit(
    model: nn.Module,
    block_index: int,
    q_unit: torch.Tensor,
    scale: float = 1.0,
) -> RankOneDownProjection:
    layers = decoder_layers(model)
    if block_index < 0 or block_index >= len(layers):
        raise IndexError(f"block index {block_index} outside [0, {len(layers)})")
    mlp = layers[block_index].mlp
    wrapper = RankOneDownProjection(mlp.down_proj, q_unit=q_unit, scale=scale)
    mlp.down_proj = wrapper
    return wrapper


def adapter_artifact(
    wrapper: RankOneDownProjection,
    *,
    model_id: str,
    probe_path: str,
    state_index: int,
    block_index: int,
    metadata: dict,
) -> dict:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "probe_path": probe_path,
        "state_index": int(state_index),
        "block_index": int(block_index),
        "q_unit": wrapper.q_unit.detach().cpu(),
        "right": wrapper.right.detach().cpu(),
        "right_norm": float(wrapper.right.detach().float().norm().cpu()),
        "metadata": metadata,
    }
