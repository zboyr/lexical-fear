"""x81 Stage 3b: freeze the outcome-blind common O/A/B/C cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from x81_common import (
    BLANK,
    MASK,
    SAME_MASS_THRESHOLD,
    RUN,
    atomic_json,
    file_record,
    match_case,
    read_json,
    replace_span,
    sha256_text,
    simple_lemma,
)

MIN_PROMPTS = 100
MIN_TARGET_WORDS = 5
MAX_TARGET_SHARE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", default=str(RUN / "x81_lexicon.json"))
    parser.add_argument("--semantics", default=str(RUN / "x81_semantics.json"))
    parser.add_argument("--out", default=str(RUN / "x81_arms.json"))
    return parser.parse_args()


def neutral_reason(
    candidate: dict,
    target_word: str,
    sensitive_words: set[str],
    features: dict[str, dict],
) -> tuple[bool, str, dict | None]:
    if candidate.get("decision") != "SAME":
        return False, "not_same", None
    word = str(candidate.get("surface", "")).casefold()
    if word == BLANK or not candidate.get("lexical_candidate"):
        return False, "not_a_replacement_word", None
    target = target_word.casefold()
    if word == target or simple_lemma(word) == simple_lemma(target):
        return False, "target_or_inflection", None
    if word in sensitive_words:
        return False, "sensitive_word", features.get(word)
    feature = features.get(word)
    if feature is None:
        return False, "outside_discovery_vocabulary", None
    if int(feature["document_frequency"]) < 10:
        return False, "discovery_df_below_10", feature
    coefficients = [float(feature["full_coef"]), *map(float, feature["fold_coefs"])]
    if any(value != 0.0 for value in coefficients):
        return False, "nonzero_discovery_coefficient", feature
    return True, "eligible", feature


def admit_prompt(
    row: dict,
    sensitive_words: set[str],
    features: dict[str, dict],
) -> dict:
    # Admission is driven solely by the same-judged fill mass over the top-30
    # candidates plus the existence of a discovery-neutral SAME replacement.
    # The top-K cumulative-mass gate and the blank-in-top-K gate were removed
    # after the x81x_top30 exploration; blank_same is still recorded for the
    # diagnostics but no longer gates admission.
    candidates = list(row["candidates"])
    p_same = sum(float(item["probability"]) for item in candidates if item["decision"] == "SAME")
    blank_rows = [item for item in candidates if item["surface"] == BLANK]
    blank_same = any(item["decision"] == "SAME" for item in blank_rows)
    neutral = []
    annotated = []
    for item in candidates:
        eligible, reason, feature = neutral_reason(
            item, row["target_word"], sensitive_words, features
        )
        record = {
            **item,
            "neutral_eligible": eligible,
            "neutral_reason": reason,
            "discovery_feature": feature,
        }
        annotated.append(record)
        if eligible:
            neutral.append(record)
    neutral.sort(key=lambda item: (-float(item["probability"]), int(item["rank"])))
    reasons = []
    if p_same < SAME_MASS_THRESHOLD:
        reasons.append(f"same_mass_below_{SAME_MASS_THRESHOLD}")
    if not neutral:
        reasons.append("no_same_discovery_neutral_replacement")
    return {
        "admitted": not reasons,
        "rejection_reasons": reasons,
        "p_same": p_same,
        "blank_same": blank_same,
        "selected_c": neutral[0] if neutral else None,
        "candidates": annotated,
    }


def main() -> None:
    args = parse_args()
    lexicon_path = Path(args.lexicon)
    semantics_path = Path(args.semantics)
    if not lexicon_path.exists() or not semantics_path.exists():
        raise SystemExit("FATAL: run x81 lexicon and semantic stages first")
    lexicon = read_json(lexicon_path)
    semantics = read_json(semantics_path)
    if lexicon.get("uses_evaluation_refusal_labels") is not False:
        raise SystemExit("FATAL: lexicon manifest has invalid outcome-blinding flag")
    if semantics.get("uses_evaluation_refusal_labels") is not False:
        raise SystemExit("FATAL: semantic manifest has invalid outcome-blinding flag")
    if lexicon["model_id"] != semantics["model_id"]:
        raise SystemExit("FATAL: model IDs differ across upstream manifests")
    if (
        lexicon.get("model_source") != semantics.get("model_source")
        or lexicon.get("model_manifest_sha256") != semantics.get("model_manifest_sha256")
    ):
        raise SystemExit("FATAL: model snapshots differ across upstream manifests")

    features = {row["word"]: row for row in lexicon["features"]}
    sensitive_words = {row["word"] for row in lexicon["sensitive_words"]}
    targets = {int(row["id"]): row for row in lexicon["targets"]}
    diagnostics = []
    arms = []
    for semantic in semantics["rows"]:
        prompt_id = int(semantic["id"])
        target = targets.get(prompt_id)
        if target is None:
            raise SystemExit(f"FATAL: semantic row has unknown target {prompt_id}")
        for key in ("prompt", "target_word", "start", "end"):
            if semantic[key] != target[key]:
                raise SystemExit(f"FATAL: semantic/lexicon target mismatch {prompt_id}:{key}")
        result = admit_prompt(semantic, sensitive_words, features)
        diagnostics.append(
            {
                "id": prompt_id,
                "target_word": target["target_word"],
                "target_word_lower": target["target_word_lower"],
                "topk_mass": semantic["topk_mass"],
                "entropy_nats": semantic["entropy_nats"],
                **result,
            }
        )
        if not result["admitted"]:
            continue
        selected = result["selected_c"]
        original = target["prompt"]
        arm_a = replace_span(original, int(target["start"]), int(target["end"]), "")
        arm_b = replace_span(original, int(target["start"]), int(target["end"]), MASK)
        replacement = match_case(selected["surface"], target["target_word"])
        arm_c = replace_span(original, int(target["start"]), int(target["end"]), replacement)
        arm_strings = {"O": original, "A": arm_a, "B": arm_b, "C": arm_c}
        arms.append(
            {
                "id": prompt_id,
                "target_word": target["target_word"],
                "target_word_lower": target["target_word_lower"],
                "start": target["start"],
                "end": target["end"],
                "other_sensitive_words": target["other_sensitive_words"],
                "topk_mass": semantic["topk_mass"],
                "p_same": result["p_same"],
                "fill_entropy_nats": semantic["entropy_nats"],
                "original_rank": next(
                    (item["rank"] for item in semantic["candidates"] if item["is_original_surface"]),
                    None,
                ),
                "original_probability": next(
                    (
                        item["probability"]
                        for item in semantic["candidates"]
                        if item["is_original_surface"]
                    ),
                    None,
                ),
                "blank_probability": next(
                    (item["probability"] for item in semantic["candidates"] if item["is_blank"]),
                    None,
                ),
                "c_replacement": replacement,
                "c_candidate": selected,
                "prompts": arm_strings,
                "prompt_sha256": {key: sha256_text(value) for key, value in arm_strings.items()},
            }
        )

    target_counts = Counter(row["target_word_lower"] for row in arms)
    max_share = max(target_counts.values(), default=0) / max(1, len(arms))
    support_checks = {
        "at_least_100_prompts": len(arms) >= MIN_PROMPTS,
        "at_least_5_target_words": len(target_counts) >= MIN_TARGET_WORDS,
        "max_target_share_at_most_0.50": max_share <= MAX_TARGET_SHARE,
    }
    arms_sha = sha256_text(
        "\n".join(
            f"{row['id']}\t{row['prompt_sha256']['O']}\t{row['prompt_sha256']['A']}\t"
            f"{row['prompt_sha256']['B']}\t{row['prompt_sha256']['C']}"
            for row in sorted(arms, key=lambda value: value["id"])
        )
    )
    output = {
        "experiment": "x81",
        "stage": "3b",
        "model_id": lexicon["model_id"],
        "model_source": lexicon["model_source"],
        "model_manifest_sha256": lexicon["model_manifest_sha256"],
        "uses_evaluation_refusal_labels": False,
        "status": "pass" if all(support_checks.values()) else "support_gate_failed",
        "config": {
            "same_mass_threshold": SAME_MASS_THRESHOLD,
            "top_k_fills": len(next(iter(semantics["rows"]))["candidates"]) if semantics["rows"] else None,
            "gates": "p_same over top-30 >= same_mass_threshold AND a SAME discovery-neutral C exists",
            "removed_gates": ["topk_cumulative_mass", "blank_in_topk_and_same"],
            "min_prompts": MIN_PROMPTS,
            "min_target_words": MIN_TARGET_WORDS,
            "max_target_share": MAX_TARGET_SHARE,
        },
        "inputs": {"lexicon": file_record(lexicon_path), "semantics": file_record(semantics_path)},
        "counts": {
            "evaluated": len(diagnostics),
            "admitted": len(arms),
            "target_words": len(target_counts),
            "max_target_share": max_share,
            "target_word_counts": dict(target_counts.most_common()),
        },
        "support_checks": support_checks,
        "arms_sha256": arms_sha,
        "arms": arms,
        "diagnostics": diagnostics,
    }
    atomic_json(args.out, output)
    print(f"x81 arms: admitted={len(arms)}/{len(diagnostics)} status={output['status']} -> {args.out}")
    if output["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
