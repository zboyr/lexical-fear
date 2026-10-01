"""Eight-model transfer figure, drawn in the style of the dose-curves
figure: per condition, marker-and-line series over models with 95%
intervals. The Qwen family sizes are connected by lines; off-family
models stand alone. A directed arrow per model runs from its Toxic delta
to its Hard-1K delta, making the benefit-cost gap visible. Reads
data/results/x88_cross_model_deep_dose.json.
Output: <out-dir>/transfer-endpoints.pdf (default figures/).
Run: python code/figures/fig_transfer.py [--out-dir DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = REPO / "data" / "results" / "x88_cross_model_deep_dose.json"
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--out-dir", type=Path, default=REPO / "figures",
                 help="output directory (default: figures/ at the repository root)")
OUT_DIR = _ap.parse_args().out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "transfer-endpoints.pdf"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.edgecolor": "#333333",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "text.color": "#333333",
        "axes.labelcolor": "#333333",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.5,
    }
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#333333"
MUTED = "#888888"

MODELS = [
    ("qwen08b", "0.8B"),
    ("qwen2b", "2B"),
    ("qwen", "4B"),
    ("qwen9b", "9B"),
    ("qwen27b", "27B"),
    ("llama", "Llama"),
    ("gemma", "Gemma"),
    ("phi", "Phi"),
]
# index ranges of connected families (lines break between families)
FAMILIES = [range(0, 5), range(5, 6), range(6, 7), range(7, 8)]
CONDITIONS = [
    ("x86_m150", "boundary write $-1.5$"),
    ("x75_m3", "rank-one $-3$ (safety only)"),
    ("x75_m6", "rank-one $-6$"),
]
ENDPOINTS = [
    ("orbench_hard_refusal", "Hard-1K refusal", BLUE, "o", True),
    ("orbench_toxic_refusal", "Toxic refusal", ORANGE, "s", False),
    ("strongreject_score", "StrongREJECT", MUTED, "^", False),
    ("ifeval_prompt_strict", "IFEval strict", INK, "D", False),
]

models = json.loads(RESULTS.read_text())["models"]


def get(tag, cond, key):
    e = models[tag]["comparisons"][cond].get(key)
    if e is None:
        return None
    return (
        e["mean_delta"],
        e["mean_delta"] - e["two_sided_ci"][0],
        e["two_sided_ci"][1] - e["mean_delta"],
    )


fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), sharey=True)

xs = list(range(len(MODELS)))

for ax, (cond, title) in zip(axes, CONDITIONS):
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    # benefit-cost arrow: Toxic delta -> Hard-1K delta, per model
    for x, (tag, _) in zip(xs, MODELS):
        h = get(tag, cond, "orbench_hard_refusal")[0]
        t = get(tag, cond, "orbench_toxic_refusal")[0]
        ax.annotate(
            "",
            xy=(x, h),
            xytext=(x, t),
            arrowprops=dict(
                arrowstyle="-|>",
                color=MUTED,
                lw=0.9,
                shrinkA=2,
                shrinkB=2.5,
                mutation_scale=6,
            ),
            zorder=2,
        )
    for key, label, color, marker, filled in ENDPOINTS:
        if cond == "x75_m3" and key == "ifeval_prompt_strict":
            continue  # safety-endpoints-only condition
        vals = [get(tag, cond, key) for tag, _ in MODELS]
        for fam in FAMILIES:
            fx = [x for x in fam]
            ax.plot(
                fx,
                [vals[x][0] for x in fam],
                color=color,
                lw=1.0 if len(fam) > 1 else 0,
                zorder=3,
            )
        ax.errorbar(
            xs,
            [v[0] for v in vals],
            yerr=[[v[1] for v in vals], [v[2] for v in vals]],
            fmt=marker,
            color=color,
            ms=2.8,
            elinewidth=0.6,
            capsize=1.0,
            lw=0,
            markerfacecolor=color if filled else "white",
            markeredgewidth=0.8,
            zorder=4,
        )
    ax.set_title(title, pad=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in MODELS], rotation=55, ha="right")
    ax.grid(axis="y", zorder=0)
    ax.set_xlim(-0.6, 7.6)

axes[0].set_ylabel("change vs. base")
axes[0].legend(
    handles=[
        Line2D([], [], color=c, marker=m, ms=3,
               markerfacecolor=c if f else "white", lw=1.0, label=l)
        for _, l, c, m, f in ENDPOINTS
    ],
    loc="upper left",
    frameon=False,
    handlelength=1.3,
    labelspacing=0.25,
    borderaxespad=0.1,
)

fig.tight_layout(pad=0.4, w_pad=0.6)
fig.savefig(OUT)
print("wrote", OUT)
