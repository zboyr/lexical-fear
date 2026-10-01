"""x85 Stage 4: crossed vector ANOVA, low-rank geometry, and behavior linkage."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.utils.extmath import randomized_svd

from x85_common import (
    FEATURES,
    POSITIONS,
    PRIMARY_STATES,
    RESULTS,
    RUN,
    atomic_json,
    cohort_path,
    file_record,
    read_json,
    sha256_file,
    two_way_residual,
    vector_anova,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default=str(cohort_path()))
    parser.add_argument("--features-glob", default=str(FEATURES / "x85_hidden_shard_*.pt"))
    parser.add_argument("--run-dir", default=str(RUN), help="generation/judgment directory")
    parser.add_argument("--states", default=",".join(str(value) for value in PRIMARY_STATES))
    parser.add_argument("--positions", default="tbg_minus1,tbg")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--randomization", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=85)
    parser.add_argument("--out", default=str(RESULTS / "x85_crossed_keyword_latent.json"))
    parser.add_argument("--directions-out", default=str(RESULTS / "x85_crossed_keyword_directions.npz"))
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("states must be a nonempty unique comma-separated list")
    return result


def parse_position_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or not set(result).issubset(POSITIONS):
        raise ValueError(f"positions must be unique members of {POSITIONS}")
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom == 0 else float(np.dot(a, b) / denom)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("cannot normalize a zero direction")
    return vector / norm


def finite_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return {"n": int(keep.sum()), "rho": None, "p": None}
    result = stats.spearmanr(x[keep], y[keep])
    return {"n": int(keep.sum()), "rho": float(result.statistic), "p": float(result.pvalue)}


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def two_way_bootstrap_stat(
    values: np.ndarray,
    prompt_index: np.ndarray,
    word_index: np.ndarray,
    n_bootstrap: int,
    seed: int,
    statistic: str = "mean",
    other: np.ndarray | None = None,
) -> np.ndarray:
    """Two-way cluster bootstrap for a scalar mean or Spearman correlation."""
    rng = np.random.default_rng(seed)
    prompts = np.unique(prompt_index)
    words = np.unique(word_index)
    out = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        prompt_counts = Counter(rng.choice(prompts, size=len(prompts), replace=True).tolist())
        word_counts = Counter(rng.choice(words, size=len(words), replace=True).tolist())
        weights = np.asarray(
            [prompt_counts[int(p)] * word_counts[int(w)] for p, w in zip(prompt_index, word_index)],
            dtype=int,
        )
        keep = weights > 0
        if statistic == "mean":
            out[bootstrap_index] = float(np.average(values[keep], weights=weights[keep]))
        elif statistic == "spearman":
            if other is None:
                raise ValueError("Spearman bootstrap requires other")
            repeated = np.repeat(np.flatnonzero(keep), weights[keep])
            result = finite_spearman(values[repeated], other[repeated])
            out[bootstrap_index] = np.nan if result["rho"] is None else result["rho"]
        else:
            raise ValueError(f"unknown bootstrap statistic {statistic}")
    return out


def behavior_two_way_bootstrap_spearman(
    score: np.ndarray,
    prompt_index: np.ndarray,
    word_index: np.ndarray,
    harm_count: np.ndarray,
    neutral_count: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Crossed cluster bootstrap plus within-cell binomial response resampling."""
    rng = np.random.default_rng(seed)
    prompts = np.unique(prompt_index)
    words = np.unique(word_index)
    out = np.empty(n_bootstrap, dtype=float)
    for bootstrap_index in range(n_bootstrap):
        prompt_counts = Counter(rng.choice(prompts, size=len(prompts), replace=True).tolist())
        word_counts = Counter(rng.choice(words, size=len(words), replace=True).tolist())
        weights = np.asarray(
            [prompt_counts[int(p)] * word_counts[int(w)] for p, w in zip(prompt_index, word_index)],
            dtype=int,
        )
        sampled_delta = (
            rng.binomial(10, harm_count / 10.0) - rng.binomial(10, neutral_count / 10.0)
        ) / 10.0
        sampled_residual = two_way_residual(sampled_delta, prompt_index, word_index)
        repeated = np.repeat(np.flatnonzero(weights > 0), weights[weights > 0])
        result = finite_spearman(score[repeated], sampled_residual[repeated])
        out[bootstrap_index] = np.nan if result["rho"] is None else result["rho"]
    return out


def sign_flip_p(values_by_position: dict[str, np.ndarray], n_randomization: int, seed: int) -> dict[str, float]:
    """Paired sign-flip max statistic across the requested positions."""
    names = list(values_by_position)
    matrix = np.stack([values_by_position[name] for name in names])
    observed = np.abs(matrix.mean(axis=1))
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(names), dtype=int)
    for _ in range(n_randomization):
        signs = rng.choice((-1.0, 1.0), size=matrix.shape[1])
        null_max = float(np.abs((matrix * signs).mean(axis=1)).max())
        exceed += null_max >= observed
    return {name: float((exceed[index] + 1) / (n_randomization + 1)) for index, name in enumerate(names)}


def load_selected_features(
    paths: list[Path], manifest: dict, states: list[int], positions: list[str]
) -> tuple[np.ndarray, dict]:
    n_cells = len(manifest["cells"])
    output = None
    seen = np.zeros(n_cells, dtype=bool)
    metadata = None
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("experiment") != "x85" or artifact.get("cohort_sha256") != manifest["cohort_sha256"]:
            raise SystemExit(f"FATAL: invalid feature shard {path}")
        position_lookup = {name: index for index, name in enumerate(artifact["position_names"])}
        if not set(positions).issubset(position_lookup):
            raise SystemExit(f"FATAL: shard {path} lacks a requested position")
        if max(states) >= int(artifact["num_states"]):
            raise SystemExit(f"FATAL: requested state absent from {path}")
        hidden_dim = int(artifact["hidden_dim"])
        if output is None:
            output = np.empty((n_cells, len(positions), len(states), hidden_dim), dtype=np.float32)
            metadata = {"num_states": int(artifact["num_states"]), "hidden_dim": hidden_dim}
        elif hidden_dim != output.shape[-1]:
            raise SystemExit("FATAL: hidden dimension differs across shards")
        tensor = artifact["states"]
        if tensor.shape[0] != len(artifact["cells"]):
            raise SystemExit(f"FATAL: cell/state row mismatch in {path}")
        for shard_row, record in enumerate(artifact["cells"]):
            cell_index = int(record["cell_index"])
            expected = manifest["cells"][cell_index]
            if (
                seen[cell_index]
                or record["cell_id"] != expected["cell_id"]
                or record["prompt_sha256"] != expected["prompt_sha256"]
            ):
                raise SystemExit(f"FATAL: duplicate or mismatched feature cell {cell_index}")
            selected = tensor[
                shard_row,
                torch.tensor([position_lookup[name] for name in positions]),
            ][:, torch.tensor(states)].float().numpy()
            output[cell_index] = selected
            seen[cell_index] = True
    if output is None or not seen.all():
        missing = np.flatnonzero(~seen)[:10].tolist()
        raise SystemExit(f"FATAL: feature shards lack cells {missing}")
    return output, metadata


def cell_index_grids(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    n_prompts, n_words = len(manifest["prompts"]), len(manifest["words"])
    harm = np.full((n_prompts, n_words), -1, dtype=int)
    neutral = np.full_like(harm, -1)
    for cell in manifest["cells"]:
        if cell["condition"] not in ("H", "N"):
            continue
        target = harm if cell["condition"] == "H" else neutral
        p, w = int(cell["prompt_index"]), int(cell["word_index"])
        if target[p, w] != -1:
            raise SystemExit(f"FATAL: duplicate {cell['condition']} cell at {p}/{w}")
        target[p, w] = int(cell["cell_index"])
    if (harm < 0).any() or (neutral < 0).any():
        raise SystemExit("FATAL: dense H/N grid is incomplete")
    return harm, neutral


def component_report(deltas: np.ndarray, rank: int, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    parts = vector_anova(deltas)
    p_count, w_count, _ = deltas.shape
    grand = parts["grand"]
    q = unit(grand)
    cell_projection = deltas @ q
    prompt_means = deltas.mean(axis=1)
    word_means = deltas.mean(axis=0)
    energies = {
        "global": p_count * w_count * float(np.square(parts["grand"]).sum()),
        "prompt": w_count * float(np.square(parts["prompt"]).sum()),
        "word": p_count * float(np.square(parts["word"]).sum()),
        "interaction": float(np.square(parts["interaction"]).sum()),
    }
    total = float(np.square(deltas).sum())
    if not np.isclose(sum(energies.values()), total, rtol=2e-4, atol=1e-3):
        raise AssertionError("vector ANOVA energy does not reconstruct total energy")
    word_rank = min(rank, min(word_means.shape))
    _, word_s, word_vt = randomized_svd(
        word_means, n_components=word_rank, n_iter=5, random_state=seed
    )
    interaction_matrix = parts["interaction"].reshape(p_count * w_count, -1)
    interaction_rank = min(rank, min(interaction_matrix.shape))
    _, interaction_s, interaction_vt = randomized_svd(
        interaction_matrix, n_components=interaction_rank, n_iter=5, random_state=seed + 1
    )
    word_energy = float(np.square(word_means).sum())
    interaction_energy = float(np.square(interaction_matrix).sum())
    report = {
        "grand_norm": float(np.linalg.norm(grand)),
        "positive_cell_fraction": float((cell_projection > 0).mean()),
        "positive_prompt_mean_fraction": float(((prompt_means @ q) > 0).mean()),
        "positive_word_mean_fraction": float(((word_means @ q) > 0).mean()),
        "energy": {
            **energies,
            "total": total,
            "fractions": {name: value / total if total else None for name, value in energies.items()},
        },
        "word_mean_singular_values": word_s.tolist(),
        "word_mean_rank_energy": (
            np.cumsum(np.square(word_s)) / word_energy if word_energy else np.zeros_like(word_s)
        ).tolist(),
        "interaction_singular_values": interaction_s.tolist(),
        "interaction_rank_energy": (
            np.cumsum(np.square(interaction_s)) / interaction_energy
            if interaction_energy
            else np.zeros_like(interaction_s)
        ).tolist(),
        "word_mean_cosine_to_global": [cosine(row, grand) for row in word_means],
        "prompt_mean_cosine_to_global": [cosine(row, grand) for row in prompt_means],
    }
    arrays = {
        "global": grand,
        "global_unit": q,
        "prompt_effect": parts["prompt"],
        "word_effect": parts["word"],
        "word_basis": word_vt,
        "interaction_basis": interaction_vt,
        "cell_projection": cell_projection,
    }
    return report, arrays


def cross_validation_report(
    deltas: np.ndarray, prompt_folds: np.ndarray, word_folds: np.ndarray
) -> tuple[list[dict], np.ndarray]:
    scores = np.full(deltas.shape[:2], np.nan, dtype=float)
    reports = []
    for fold in range(4):
        p_train, p_test = prompt_folds != fold, prompt_folds == fold
        w_train, w_test = word_folds != fold, word_folds == fold
        train_mean = deltas[np.ix_(p_train, w_train)].mean(axis=(0, 1))
        q = unit(train_mean)
        quadrants = {}
        for name, pmask, wmask in (
            ("seen_prompt_seen_word", p_train, w_train),
            ("new_prompt_seen_word", p_test, w_train),
            ("seen_prompt_new_word", p_train, w_test),
            ("new_prompt_new_word", p_test, w_test),
        ):
            values = deltas[np.ix_(pmask, wmask)] @ q
            quadrants[name] = {
                "n_cells": int(values.size),
                "mean_projection": float(values.mean()),
                "positive_fraction": float((values > 0).mean()),
                "mean_direction_cosine": cosine(
                    deltas[np.ix_(pmask, wmask)].mean(axis=(0, 1)), train_mean
                ),
            }
        held = deltas[np.ix_(p_test, w_test)] @ q
        scores[np.ix_(p_test, w_test)] = held
        reports.append({"fold": fold, "quadrants": quadrants})
    return reports, scores


def load_refusal_rates(condition: str, manifest: dict, run_dir: Path) -> dict[int, dict] | None:
    path = run_dir / f"x85_refusal_{condition}.jsonl"
    generation_path = run_dir / f"x85_generations_{condition}.json"
    if not path.exists() or not generation_path.exists():
        return None
    generations = read_json(generation_path)
    if generations.get("cohort_sha256") != manifest["cohort_sha256"]:
        raise SystemExit(f"FATAL: {condition} generations belong to another cohort")
    expected = {int(row["id"]): row for row in generations["rows"]}
    rates = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("failed"):
                continue
            behavior_id = int(row["id"])
            source = expected.get(behavior_id)
            if source is None:
                continue
            if (
                row.get("cell_id") != source["cell_id"]
                or row.get("response_sha256") != source["response_sha256"]
                or len(row.get("refusal", [])) != 10
            ):
                raise SystemExit(f"FATAL: invalid {condition} judgment row {behavior_id}")
            if behavior_id in rates:
                raise SystemExit(f"FATAL: duplicate {condition} judgment row {behavior_id}")
            count = int(sum(row["refusal"]))
            rates[behavior_id] = {"count": count, "rate": count / 10.0}
    if set(rates) != set(expected):
        return None
    return rates


def behavior_arrays(manifest: dict, run_dir: Path) -> dict | None:
    rates = {
        condition: load_refusal_rates(condition, manifest, run_dir)
        for condition in ("O", "H", "N")
    }
    if any(value is None for value in rates.values()):
        return None
    original_by_prompt = {}
    cells_by_key = {}
    for cell in manifest["cells"]:
        if not cell["behavior"]:
            continue
        if cell["condition"] == "O":
            original_by_prompt[int(cell["prompt_index"])] = rates["O"][int(cell["behavior_id"])]
        else:
            cells_by_key[(cell["condition"], int(cell["prompt_index"]), int(cell["word_index"]))] = cell
    rows = []
    for condition, prompt_index, word_index in sorted(cells_by_key):
        if condition != "H":
            continue
        harm = cells_by_key[("H", prompt_index, word_index)]
        neutral = cells_by_key[("N", prompt_index, word_index)]
        h_result = rates["H"][int(harm["behavior_id"])]
        n_result = rates["N"][int(neutral["behavior_id"])]
        o_result = original_by_prompt[prompt_index]
        r_h, r_n, r_o = h_result["rate"], n_result["rate"], o_result["rate"]
        rows.append(
            {
                "prompt_index": prompt_index,
                "word_index": word_index,
                "r_O": r_o,
                "r_H": r_h,
                "r_N": r_n,
                "delta_H_minus_N": r_h - r_n,
                "delta_H_minus_O": r_h - r_o,
                "delta_N_minus_O": r_n - r_o,
                "count_H": h_result["count"],
                "count_N": n_result["count"],
                "count_O": o_result["count"],
            }
        )
    return {
        "rows": rows,
        "prompt_index": np.asarray([row["prompt_index"] for row in rows], dtype=int),
        "word_index": np.asarray([row["word_index"] for row in rows], dtype=int),
        "delta": np.asarray([row["delta_H_minus_N"] for row in rows], dtype=float),
        "count_H": np.asarray([row["count_H"] for row in rows], dtype=int),
        "count_N": np.asarray([row["count_N"] for row in rows], dtype=int),
    }


def main() -> None:
    args = parse_args()
    states = parse_int_list(args.states)
    positions = parse_position_list(args.positions)
    manifest = read_json(Path(args.cohort))
    if manifest.get("experiment") != "x85" or manifest.get("status") != "frozen":
        raise SystemExit("FATAL: invalid or unfrozen x85 cohort")
    feature_paths = [Path(value) for value in sorted(glob.glob(args.features_glob))]
    if not feature_paths:
        raise SystemExit(f"FATAL: no feature shards match {args.features_glob}")
    features, feature_metadata = load_selected_features(feature_paths, manifest, states, positions)
    harm_grid, neutral_grid = cell_index_grids(manifest)
    prompt_folds = np.asarray([row["prompt_fold"] for row in manifest["prompts"]], dtype=int)
    word_folds = np.asarray([row["word_fold"] for row in manifest["words"]], dtype=int)
    prompt_ids = np.asarray([row["id"] for row in manifest["prompts"]], dtype=int)
    word_indices = np.asarray([row["word_index"] for row in manifest["words"]], dtype=int)

    reports = {}
    arrays_for_output = {}
    deltas_by_key = {}
    crossfit_by_key = {}
    for position_offset, position in enumerate(positions):
        reports[position] = {}
        for state_offset, state in enumerate(states):
            deltas = (
                features[harm_grid, position_offset, state_offset]
                - features[neutral_grid, position_offset, state_offset]
            )
            key = f"{position}_s{state}"
            report, component_arrays = component_report(deltas, args.rank, args.seed + state)
            cross_validation, crossfit = cross_validation_report(deltas, prompt_folds, word_folds)
            finite = np.isfinite(crossfit)
            report["cross_validation"] = cross_validation
            report["double_heldout"] = {
                "n_cells": int(finite.sum()),
                "mean_projection": float(crossfit[finite].mean()),
                "positive_fraction": float((crossfit[finite] > 0).mean()),
            }
            reports[position][str(state)] = report
            deltas_by_key[key] = deltas
            crossfit_by_key[key] = crossfit
            arrays_for_output[f"global_{key}"] = component_arrays["global"]
            arrays_for_output[f"global_unit_{key}"] = component_arrays["global_unit"]
            arrays_for_output[f"word_basis_{key}"] = component_arrays["word_basis"]
            arrays_for_output[f"interaction_basis_{key}"] = component_arrays["interaction_basis"]

    position_comparisons = {}
    if "tbg_minus1" in positions and "tbg" in positions:
        for state in states:
            a = arrays_for_output[f"global_tbg_minus1_s{state}"]
            b = arrays_for_output[f"global_tbg_s{state}"]
            position_comparisons[str(state)] = cosine(a, b)

    # Primary cross-fitted score: average unit-direction projection over the
    # independently fixed primary states. Only fold-diagonal cells are finite.
    primary_states = [state for state in PRIMARY_STATES if state in states]
    primary_scores = {}
    primary_summary = {}
    if primary_states:
        for position in (name for name in ("tbg_minus1", "tbg") if name in positions):
            stack = np.stack([crossfit_by_key[f"{position}_s{state}"] for state in primary_states])
            finite = np.isfinite(stack).all(axis=0)
            score = np.full(stack.shape[1:], np.nan, dtype=float)
            score[finite] = stack[:, finite].mean(axis=0)
            pidx, widx = np.indices(score.shape)
            values = score[finite]
            pflat, wflat = pidx[finite], widx[finite]
            bootstrap = two_way_bootstrap_stat(
                values, pflat, wflat, args.bootstrap, args.seed + 100, statistic="mean"
            )
            word_means = np.asarray([values[wflat == word].mean() for word in np.unique(wflat)])
            fold_means = np.asarray(
                [values[prompt_folds[pflat] == fold].mean() for fold in range(4)]
            )
            primary_scores[position] = values
            primary_summary[position] = {
                "states": primary_states,
                "n_double_heldout_cells": int(len(values)),
                "mean_projection": float(values.mean()),
                "ci95_two_way_bootstrap": [
                    float(value) for value in np.nanquantile(bootstrap, (0.025, 0.975))
                ],
                "positive_cell_fraction": float((values > 0).mean()),
                "positive_word_mean_fraction": float((word_means > 0).mean()),
                "fold_mean_projections": fold_means.tolist(),
            }
        randomization_p = sign_flip_p(primary_scores, args.randomization, args.seed + 200)
        adjusted = holm([randomization_p[name] for name in primary_summary])
        for (name, summary), adj in zip(primary_summary.items(), adjusted):
            summary["sign_flip_max_stat_p"] = randomization_p[name]
            summary["holm_adjusted_p"] = adj
            summary["gate_pass"] = bool(
                min(summary["fold_mean_projections"]) > 0
                and summary["ci95_two_way_bootstrap"][0] > 0
                and summary["positive_word_mean_fraction"] >= 0.75
                and adj < 0.05
            )

    behavior = behavior_arrays(manifest, Path(args.run_dir))
    behavior_report = None
    behavior_gate = None
    if behavior is not None:
        pidx, widx, y = behavior["prompt_index"], behavior["word_index"], behavior["delta"]
        y_residual = two_way_residual(y, pidx, widx)
        behavior_report = {
            "n_edges": int(len(y)),
            "mean_r_H_minus_r_N": float(y.mean()),
            "mean_r_H_minus_r_O": float(np.mean([row["delta_H_minus_O"] for row in behavior["rows"]])),
            "mean_r_N_minus_r_O": float(np.mean([row["delta_N_minus_O"] for row in behavior["rows"]])),
            "per_position_state": {},
        }
        band_scores = {}
        for position in positions:
            behavior_report["per_position_state"][position] = {}
            state_scores = []
            for state in states:
                deltas = deltas_by_key[f"{position}_s{state}"]
                scores = np.empty(len(y), dtype=float)
                for row_index, (p, w) in enumerate(zip(pidx, widx)):
                    pf, wf = prompt_folds[p], word_folds[w]
                    train = deltas[np.ix_(prompt_folds != pf, word_folds != wf)]
                    scores[row_index] = float(deltas[p, w] @ unit(train.mean(axis=(0, 1))))
                residual = two_way_residual(scores, pidx, widx)
                behavior_report["per_position_state"][position][str(state)] = {
                    "raw": finite_spearman(scores, y),
                    "two_way_residualized": finite_spearman(residual, y_residual),
                }
                if state in primary_states:
                    state_scores.append(residual)
            if state_scores:
                band_scores[position] = np.mean(state_scores, axis=0)
        behavior_gate = {}
        for offset, (position, score) in enumerate(band_scores.items()):
            association = finite_spearman(score, y_residual)
            boot = behavior_two_way_bootstrap_spearman(
                score,
                pidx,
                widx,
                behavior["count_H"],
                behavior["count_N"],
                args.bootstrap,
                args.seed + 300 + offset,
            )
            fold_rhos = []
            for fold in range(4):
                keep = prompt_folds[pidx] == fold
                fold_rhos.append(finite_spearman(score[keep], y_residual[keep])["rho"])
            ci = [float(value) for value in np.nanquantile(boot, (0.025, 0.975))]
            finite_fold_rhos = [value for value in fold_rhos if value is not None]
            behavior_gate[position] = {
                "association": association,
                "ci95_two_way_bootstrap": ci,
                "prompt_fold_rhos": fold_rhos,
                "gate_pass": bool(
                    association["rho"] is not None
                    and len(finite_fold_rhos) == 4
                    and min(finite_fold_rhos) > 0
                    and ci[0] > 0
                ),
            }

    output = {
        "experiment": "x85",
        "status": "representation_complete_behavior_complete" if behavior is not None else "representation_complete_behavior_pending",
        "scope": "Qwen3.5-4B x83 snapshot; 96 prompts x 24 harm/neutral word pairs; append position",
        "config": {
            "states": states,
            "positions": positions,
            "primary_states_available": primary_states,
            "rank": args.rank,
            "bootstrap": args.bootstrap,
            "randomization": args.randomization,
            "seed": args.seed,
        },
        "inputs": {
            "cohort": file_record(Path(args.cohort)),
            "feature_shards": [
                {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in feature_paths
            ],
        },
        "feature_metadata": feature_metadata,
        "representation": reports,
        "tbg_minus1_vs_tbg_global_cosine": position_comparisons,
        "primary_representation_gate": primary_summary,
        "representation_gate_pass": bool(
            len(primary_summary) == 2 and all(row["gate_pass"] for row in primary_summary.values())
        ),
        "behavior": behavior_report,
        "behavior_association_gate": behavior_gate,
        "behavior_gate_pass": bool(
            behavior_gate is not None
            and len(behavior_gate) == 2
            and all(row["gate_pass"] for row in behavior_gate.values())
        ),
        "interpretation_rule": (
            "A representation-only pass is a shared harm-word shift. It is refusal-associated only "
            "if the separate behavior gate passes, and causal only after the optional intervention stage."
        ),
    }
    out_path, directions_path = Path(args.out), Path(args.directions_out)
    atomic_json(out_path, output)
    directions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directions_path,
        **arrays_for_output,
        prompt_ids=prompt_ids,
        word_indices=word_indices,
        prompt_folds=prompt_folds,
        word_folds=word_folds,
    )
    print(
        f"x85 analysis: representation_gate={output['representation_gate_pass']} "
        f"behavior_gate={output['behavior_gate_pass']} -> {out_path}"
    )


if __name__ == "__main__":
    main()
