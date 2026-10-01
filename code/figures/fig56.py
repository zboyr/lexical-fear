"""Figures 5 and 6 for the paper draft (Section 4, path B geometry).

fig5 (projection.pdf): held-out projection vs. refusal change on the
    576 behavior edges, both axes residualized on prompt + word fixed
    effects. Recomputed from the x85 feature shards and refusal judgments
    exactly as x85_analyze.py does for its association gate (position
    tbg_minus1, band = PRIMARY_STATES 29-31, leave-(prompt-fold, word-fold)
    -out direction per edge); the recomputed Spearman rho is asserted to
    match the frozen result JSON.
fig6 (heatmap.pdf): position x state heatmaps of (a) the word-identity
    energy fraction of the harm-minus-neutral shift and (b) the two-way
    residualized Spearman rho between held-out projection and refusal
    change, read from x85_crossed_keyword_latent_allstates.json.

Needs the 24 x85 hidden-state shards (data/runs/x85/features/). Run:
    python code/figures/fig56.py [--out-dir DIR]     (default: figures/)
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--out-dir", type=Path, default=REPO / "figures",
                 help="output directory (default: figures/ at the repository root)")
FIG = _ap.parse_args().out_dir
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "code"))

from x85_analyze import (  # noqa: E402
    behavior_arrays,
    cell_index_grids,
    load_selected_features,
    unit,
)
from x85_common import PRIMARY_STATES, read_json, two_way_residual  # noqa: E402

BLUE = "#0072B2"
INK = "#333333"
MUTED = "#888888"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
})

POSITION = "tbg_minus1"  # the position quoted in the text (rho = 0.65)


def fig5():
    manifest = read_json(REPO / "data/runs/x85/x85_cohort.json")
    paths = [Path(p) for p in sorted(glob.glob(str(REPO / "data/runs/x85/features/x85_hidden_shard_*.pt")))]
    features, _ = load_selected_features(paths, manifest, list(PRIMARY_STATES), [POSITION])
    harm_grid, neutral_grid = cell_index_grids(manifest)
    prompt_folds = np.asarray([r["prompt_fold"] for r in manifest["prompts"]], dtype=int)
    word_folds = np.asarray([r["word_fold"] for r in manifest["words"]], dtype=int)

    behavior = behavior_arrays(manifest, REPO / "data/runs/x85")
    if behavior is None:
        raise SystemExit("FATAL: behavior artifacts incomplete")
    pidx, widx, y = behavior["prompt_index"], behavior["word_index"], behavior["delta"]

    state_scores = []
    for state_offset in range(len(PRIMARY_STATES)):
        deltas = (
            features[harm_grid, 0, state_offset]
            - features[neutral_grid, 0, state_offset]
        )
        # leave-(prompt-fold, word-fold)-out unit directions, one per fold pair
        units = {
            (pf, wf): unit(deltas[np.ix_(prompt_folds != pf, word_folds != wf)].mean(axis=(0, 1)))
            for pf in range(4)
            for wf in range(4)
        }
        state_scores.append(np.asarray(
            [deltas[p, w] @ units[(prompt_folds[p], word_folds[w])] for p, w in zip(pidx, widx)]
        ))
    band = np.mean(state_scores, axis=0)
    x_res = two_way_residual(band, pidx, widx)
    y_res = two_way_residual(y, pidx, widx)
    rho = stats.spearmanr(x_res, y_res).statistic

    frozen = read_json(REPO / "data/results/x85_crossed_keyword_latent.json")
    frozen_rho = frozen["behavior_association_gate"][POSITION]["association"]["rho"]
    if abs(rho - frozen_rho) > 5e-3:
        raise SystemExit(f"FATAL: recomputed rho {rho:.4f} != frozen {frozen_rho:.4f}")

    fig, ax = plt.subplots(figsize=(2.0, 1.8))
    ax.axhline(0.0, ls="--", lw=0.7, color=MUTED)
    ax.axvline(0.0, ls="--", lw=0.7, color=MUTED)
    ax.scatter(x_res, y_res, s=7, color=BLUE, alpha=0.45, linewidths=0)
    ax.set_xlabel("held-out projection (residualized)")
    ax.set_ylabel(r"$\Delta$ refusal $r_H - r_N$ (residualized)")
    ax.annotate(rf"$\rho = {rho:.2f}$" + f"\n$n = {len(y)}$",
                xy=(0.03, 0.97), xycoords="axes fraction",
                ha="left", va="top", fontsize=7)
    fig.savefig(FIG / "projection.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"fig5: rho={rho:.4f} (frozen {frozen_rho:.4f}), n={len(y)}")


def fig6():
    allstates = read_json(REPO / "data/results/x85_crossed_keyword_latent_allstates.json")
    positions = ["instruction_last", "user_end", "tbg_minus1", "tbg"]
    labels = ["word token", "user end", "</think>", r'"\n\n"']
    states = sorted(int(s) for s in allstates["representation"][positions[0]])

    word_frac = np.full((len(positions), len(states)), np.nan)
    rho = np.full_like(word_frac, np.nan)
    for i, pos in enumerate(positions):
        for j, state in enumerate(states):
            rep = allstates["representation"][pos][str(state)]
            word_frac[i, j] = rep["energy"]["fractions"]["word"]
            rho[i, j] = allstates["behavior"]["per_position_state"][pos][str(state)][
                "two_way_residualized"]["rho"]

    fig, axes = plt.subplots(2, 1, figsize=(3.3, 1.9), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    panels = [
        (word_frac, "word-identity energy fraction", 0.0, 0.9),
        (rho, r"residualized $\rho$ (proj., $\Delta$ refusal)", 0.0, 0.7),
    ]
    extent = (states[0] - 0.5, states[-1] + 0.5, len(positions) - 0.5, -0.5)
    for ax, (grid, title, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(grid, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax,
                       extent=extent, interpolation="nearest")
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(labels)
        ax.grid(False)
        cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05)
        cbar.ax.tick_params(labelsize=6.5)
        cbar.outline.set_visible(False)
        ax.set_title(title, loc="left", fontsize=7.5, pad=2)
    axes[1].set_xlabel("residual-stream state (layer)")
    fig.savefig(FIG / "heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig6: states", states[0], "-", states[-1])


if __name__ == "__main__":
    fig5()
    fig6()
    print("wrote", FIG / "projection.pdf", "and", FIG / "heatmap.pdf")
