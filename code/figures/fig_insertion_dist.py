"""Trial redesign of the insertion-effect figure: per position, the gate
admission rate plus the full distribution shift of per-prompt refusal
rates (control vs inserted), with means (dashed) and their gap as an
arrow. Reads data/runs/x83/x83_refusal_{O,S,M,E}.jsonl and
x83_detect.json. Output: <out-dir>/insertion-dist.pdf (default figures/).
Run: python code/figures/fig_insertion_dist.py [--out-dir DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RUNS = REPO / "data" / "runs" / "x83"
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--out-dir", type=Path, default=REPO / "figures",
                 help="output directory (default: figures/ at the repository root)")
OUT_DIR = _ap.parse_args().out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "insertion-dist.pdf"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
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
PALEBLUE = "#EAF3F9"
PALEGRAY = "#F4F4F4"


def load_r(cond):
    out = {}
    with open(RUNS / f"x83_refusal_{cond}.jsonl") as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = float(np.mean(row["refusal"]))
    return out


detect = json.load(open(RUNS / "x83_detect.json"))["rows"]
admitted = {p: {r["id"] for r in detect if r["positions"][p]["admitted"]}
            for p in "SME"}
n_total = len(detect)

r = {c: load_r(c) for c in "OSME"}

POSITIONS = [("S", "start"), ("M", "middle"), ("E", "end")]

fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9), sharey=True)

bins = np.linspace(-0.05, 1.05, 12)  # r takes 11 discrete values
centers = np.linspace(0, 1, 11)

for ax, (pos, name) in zip(axes, POSITIONS):
    ids = sorted(admitted[pos])
    ro = np.array([r["O"][i] for i in ids])
    rt = np.array([r[pos][i] for i in ids])
    ho, _ = np.histogram(ro, bins=bins)
    ht, _ = np.histogram(rt, bins=bins)
    ho = ho / len(ids)
    ht = ht / len(ids)
    w = 0.09
    ax.bar(centers, ho, width=w, color=PALEGRAY, edgecolor=MUTED,
           linewidth=0.7, zorder=2, label="control")
    ax.step(np.append(centers - w / 2 * 0 - 0.045, 1.045),
            np.append(ht, ht[-1]), where="post", color=BLUE, lw=1.2,
            zorder=4)
    ax.fill_between(np.append(centers - 0.045, 1.045),
                    np.append(ht, ht[-1]), step="post", color=BLUE,
                    alpha=0.15, zorder=3, label="+1 harm word")
    # means and their gap
    y_arrow = 0.56
    ax.annotate("", xy=(rt.mean(), y_arrow), xytext=(ro.mean(), y_arrow),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0,
                                mutation_scale=7), zorder=5)
    ax.plot([ro.mean()] * 2, [0, y_arrow], color=MUTED, lw=0.8, ls="--",
            zorder=5)
    ax.plot([rt.mean()] * 2, [0, y_arrow], color=BLUE, lw=0.8, ls="--",
            zorder=5)
    ax.text((ro.mean() + rt.mean()) / 2, y_arrow - 0.05,
            f"$+{rt.mean()-ro.mean():.2f}$", ha="center", va="top",
            fontsize=7, color=INK)
    # gate admission strip at the top
    ax.set_title(f"{name}: {len(ids)}/{n_total} pass gate", pad=10)
    frac = len(ids) / n_total
    ax.plot([0, 1], [0.70, 0.70], color="#DDDDDD", lw=3,
            solid_capstyle="butt", clip_on=False)
    ax.plot([0, frac], [0.70, 0.70], color=ORANGE, lw=3,
            solid_capstyle="butt", clip_on=False)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(0, 0.66)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("per-prompt refusal rate")
    ax.grid(axis="y", zorder=0)

axes[0].set_ylabel("fraction of prompts")
from matplotlib.lines import Line2D

h, l = axes[0].get_legend_handles_labels()
h.append(Line2D([], [], color=MUTED, ls="--", lw=0.8))
l.append("average")
axes[0].legend(h, l, loc="upper left", frameon=False, handlelength=1.2,
               borderaxespad=0.1, labelspacing=0.25)

fig.tight_layout(pad=0.4, w_pad=0.6)
fig.savefig(OUT)
print("wrote", OUT)
