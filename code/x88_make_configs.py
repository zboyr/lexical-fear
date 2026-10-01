"""Emit per-model x75-chain configs for x88 from the x87 qwen template."""

from __future__ import annotations

import argparse
import copy

from x88_common import DATA, MODELS, atomic_json, read_json, tag_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template", default=str(DATA / "runs" / "x87" / "x87_x75_qwen_config.json")
    )
    parser.add_argument("--date", default="25 Aug 2026")
    args = parser.parse_args()
    template = read_json(args.template)
    for tag, entry in MODELS.items():
        config = copy.deepcopy(template)
        config["experiment_name"] = f"x88_{tag}_x75_refusal_weighted_port"
        config["model_id"] = entry["model_id"]
        config["probe_chat_date"] = args.date
        config["generation_chat_date"] = args.date
        out = tag_dir(tag) / "x88_x75_config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(out, config)
        print(out)


if __name__ == "__main__":
    main()
