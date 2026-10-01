"""Train only the right factor of a probe-constrained rank-one weight edit."""

from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

from edit_common import (  # noqa: E402
    atomic_json_dump,
    file_record,
    format_user_chat,
    load_causal_lm,
    load_config,
    set_seed,
)
from edit_rank1 import adapter_artifact, install_rank_one_edit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--prompt-splits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument(
        "--model-source",
        help="exact local snapshot to load; artifact model_id remains the canonical config model_id",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def cycle(loader: DataLoader):
    while True:
        yield from loader


def collate_rows(rows: list[dict]) -> list[str]:
    return [str(row["prompt"]) for row in rows]


def tokenize_prompts(tokenizer, prompts: list[str], date_string: str, device: torch.device):
    chats = [format_user_chat(tokenizer, prompt, date_string) for prompt in prompts]
    encoded = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def last_token_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    positions = attention_mask.long().sum(-1) - 1
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, positions]


def probe_logits(hidden: torch.Tensor, probe: dict) -> torch.Tensor:
    mean = probe["mean"].to(hidden.device, dtype=torch.float32)
    std = probe["std"].to(hidden.device, dtype=torch.float32)
    weight = probe["w_standardized"].to(hidden.device, dtype=torch.float32)
    bias = probe["bias"].to(hidden.device, dtype=torch.float32)
    return ((hidden.float() - mean) / std) @ weight + bias


def select_last_positions(logits: torch.Tensor, mask: torch.Tensor, last_n: int) -> torch.Tensor:
    seq = logits.shape[1]
    pos = torch.arange(seq, device=logits.device).unsqueeze(0)
    lengths = mask.long().sum(-1, keepdim=True)
    keep = mask.bool() & (pos >= (lengths - last_n).clamp_min(0))
    return logits[keep]


@torch.no_grad()
def validation_metrics(
    model,
    wrapper,
    tokenizer,
    probe,
    harmful_rows: list[dict],
    benign_rows: list[dict],
    date_string: str,
    batch_size: int,
    last_n: int,
    device: torch.device,
    max_batches: int = 16,
) -> dict:
    model.eval()
    harmful_losses = []
    for start in range(0, min(len(harmful_rows), max_batches * batch_size), batch_size):
        batch = harmful_rows[start : start + batch_size]
        tokens = tokenize_prompts(tokenizer, collate_rows(batch), date_string, device)
        output = model.model(
            **tokens, output_hidden_states=True, return_dict=True, use_cache=False
        )
        hidden = last_token_hidden(output.hidden_states[probe["state_index"]], tokens["attention_mask"])
        harmful_losses.append(F.softplus(-probe_logits(hidden, probe)).mean().item())

    kl_values = []
    for start in range(0, min(len(benign_rows), max_batches * batch_size), batch_size):
        batch = benign_rows[start : start + batch_size]
        tokens = tokenize_prompts(tokenizer, collate_rows(batch), date_string, device)
        wrapper.scale = 0.0
        base = model(**tokens, return_dict=True, use_cache=False).logits
        wrapper.scale = 1.0
        edited = model(**tokens, return_dict=True, use_cache=False).logits
        base = select_last_positions(base, tokens["attention_mask"], last_n).float()
        edited = select_last_positions(edited, tokens["attention_mask"], last_n).float()
        base_logp = F.log_softmax(base, dim=-1)
        edited_logp = F.log_softmax(edited, dim=-1)
        kl_values.append((base_logp.exp() * (base_logp - edited_logp)).sum(-1).mean().item())
    wrapper.scale = 1.0
    return {
        "probe_target_loss": float(sum(harmful_losses) / max(1, len(harmful_losses))),
        "benign_prompt_kl": float(sum(kl_values) / max(1, len(kl_values))),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    edit_cfg = config["edit"]
    set_seed(int(config["split"]["validation_seed"]))
    device = torch.device(args.device)
    prompt_splits = json.loads(Path(args.prompt_splits).read_text())
    probe = torch.load(args.probe, map_location="cpu", weights_only=False)

    model_source = args.model_source or config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = load_causal_lm(model_source)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapper = install_rank_one_edit(
        model,
        block_index=int(probe["block_index"]),
        q_unit=probe["q_unit"],
        scale=1.0,
    )
    wrapper.right.requires_grad_(True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters()
                 if parameter.requires_grad]
    if len(trainable) != 1 or trainable[0][1] is not wrapper.right:
        raise RuntimeError(
            "rank-one training must expose only wrapper.right as trainable; "
            f"found {[name for name, _ in trainable]}"
        )
    trainable_parameter_names = [trainable[0][0]]
    optimizer = torch.optim.AdamW(
        [wrapper.right],
        lr=float(edit_cfg["learning_rate"]),
        weight_decay=float(edit_cfg["weight_decay"]),
    )

    batch_size = int(edit_cfg["batch_size"])
    harmful_loader = DataLoader(
        prompt_splits["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_rows,
    )
    benign_loader = DataLoader(
        prompt_splits["benign_retain"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_rows,
    )
    harmful_iter, benign_iter = cycle(harmful_loader), cycle(benign_loader)
    max_steps = int(args.max_steps or edit_cfg["max_steps"])
    last_n = int(edit_cfg["retain_last_n"])
    kl_weight = float(edit_cfg["kl_weight"])
    validate_every = int(edit_cfg["validate_every"])
    grad_clip = float(edit_cfg["gradient_clip"])
    date_string = config["probe_chat_date"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.with_suffix(".jsonl")

    best_feasible = None
    best_any = None
    logs = []
    gradient_parameter_names = None
    # Keep the frozen base model in inference mode. Gradients still flow through
    # its operations into the trainable rank-one factor.
    model.eval()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        harmful = next(harmful_iter)
        benign = next(benign_iter)

        harmful_tokens = tokenize_prompts(tokenizer, harmful, date_string, device)
        harmful_output = model.model(
            **harmful_tokens,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = last_token_hidden(
            harmful_output.hidden_states[int(probe["state_index"])],
            harmful_tokens["attention_mask"],
        )
        target_loss = F.softplus(-probe_logits(hidden, probe)).mean()

        benign_tokens = tokenize_prompts(tokenizer, benign, date_string, device)
        wrapper.scale = 0.0
        with torch.no_grad():
            base_logits = model(**benign_tokens, return_dict=True, use_cache=False).logits
        wrapper.scale = 1.0
        edited_logits = model(**benign_tokens, return_dict=True, use_cache=False).logits
        base_selected = select_last_positions(
            base_logits, benign_tokens["attention_mask"], last_n
        ).float()
        edited_selected = select_last_positions(
            edited_logits, benign_tokens["attention_mask"], last_n
        ).float()
        base_logp = F.log_softmax(base_selected, dim=-1)
        edited_logp = F.log_softmax(edited_selected, dim=-1)
        kl = (base_logp.exp() * (base_logp - edited_logp)).sum(-1).mean()
        loss = target_loss + kl_weight * kl
        if not all(torch.isfinite(value) for value in (target_loss, kl, loss)):
            raise RuntimeError(f"non-finite training objective at step {step}")
        loss.backward()
        if gradient_parameter_names is None:
            gradient_parameter_names = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None
            ]
            if gradient_parameter_names != trainable_parameter_names:
                raise RuntimeError(
                    "gradient-bearing parameters differ from the sole trainable factor: "
                    f"{gradient_parameter_names} != {trainable_parameter_names}"
                )
        torch.nn.utils.clip_grad_norm_([wrapper.right], grad_clip)
        optimizer.step()
        if not torch.isfinite(wrapper.right).all():
            raise RuntimeError(f"non-finite right factor at step {step}")

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "probe_target_loss": float(target_loss.detach().cpu()),
            "benign_prompt_kl": float(kl.detach().cpu()),
            "right_norm": float(wrapper.right.detach().float().norm().cpu()),
        }
        logs.append(row)
        with open(log_path, "a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

        if step % validate_every == 0 or step == max_steps:
            val = validation_metrics(
                model,
                wrapper,
                tokenizer,
                probe,
                prompt_splits["validation"],
                prompt_splits["benign_validation"],
                date_string,
                batch_size,
                last_n,
                device,
            )
            candidate = {
                "step": step,
                **val,
                "right": wrapper.right.detach().cpu().clone(),
            }
            if best_any is None or val["probe_target_loss"] < best_any["probe_target_loss"]:
                best_any = candidate
            if val["benign_prompt_kl"] <= float(edit_cfg["kl_budget"]):
                if best_feasible is None or val["probe_target_loss"] < best_feasible["probe_target_loss"]:
                    best_feasible = candidate
            print(json.dumps({"step": step, **val}), flush=True)
            model.eval()

    chosen = best_feasible or best_any
    if chosen is None or not torch.isfinite(chosen["right"]).all():
        raise RuntimeError("training did not produce a finite checkpoint")
    wrapper.right.data.copy_(chosen["right"].to(wrapper.right.device))
    artifact = adapter_artifact(
        wrapper,
        model_id=config["model_id"],
        probe_path=str(Path(args.probe).resolve()),
        state_index=int(probe["state_index"]),
        block_index=int(probe["block_index"]),
        metadata={
            "selected_step": int(chosen["step"]),
            "validation_probe_target_loss": float(chosen["probe_target_loss"]),
            "validation_benign_prompt_kl": float(chosen["benign_prompt_kl"]),
            "kl_budget": float(edit_cfg["kl_budget"]),
            "probe_kind": probe["kind"],
            "probe_arm": probe.get("arm"),
            "probe_chat_date": date_string,
            "model_source": str(model_source),
        },
    )
    torch.save(artifact, out_path)
    summary = {
        "schema_version": 1,
        "adapter": file_record(out_path),
        "probe": file_record(args.probe),
        "prompt_splits": file_record(args.prompt_splits),
        "selected": {key: value for key, value in chosen.items() if key != "right"},
        "max_steps": max_steps,
        "n_train_harmful": len(prompt_splits["train"]),
        "n_benign_retain": len(prompt_splits["benign_retain"]),
        "trainable_parameter_names": trainable_parameter_names,
        "gradient_parameter_names": gradient_parameter_names,
        "model_generation_performed": False,
        "model_id": config["model_id"],
        "model_source": str(model_source),
    }
    atomic_json_dump(summary, out_path.with_suffix(".json"))
    print(json.dumps(summary["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
