"""Dose-extension figure: delta vs base on four external endpoints as
the dose moves in both directions, one panel per intervention family
(axes not comparable). Reads data/results/x87_external_safety_utility.json
and data/results/x89_positive_dose_extension.json (positive doses).
Output: <out-dir>/dose-curves.pdf (default figures/).
Run: python code/figures/fig_dose.py [--out-dir DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = REPO / "data" / "results" / "x87_external_safety_utility.json"
RESULTS_POS = REPO / "data" / "results" / "x89_positive_dose_extension.json"
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--out-dir", type=Path, default=REPO / "figures",
                 help="output directory (default: figures/ at the repository root)")
OUT_DIR = _ap.parse_args().out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "dose-curves.pdf"

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

ENDPOINTS = [
    ("orbench_hard_refusal", "Hard-1K refusal", BLUE, "o"),
    ("orbench_toxic_refusal", "Toxic refusal", ORANGE, "s"),
    ("strongreject_score", "StrongREJECT", MUTED, "^"),
    ("ifeval_prompt_strict", "IFEval strict", INK, "D"),
]

# condition name -> signed dose on that family's own axis; None = base (delta 0
# by construction, no resampling)
WRITE_DOSES = [
    ("x86_p150", 1.5),
    ("x86_p100", 1.0),
    ("x86_p050", 0.5),
    (None, 0.0),
    ("x86_full", -0.5),
    ("x86_m075", -0.75),
    ("x86_m100", -1.0),
    ("x86_m125", -1.25),
    ("x86_m150", -1.5),
    ("x86_m200", -2.0),
    ("x86_m300", -3.0),
]
EDIT_DOSES = [
    ("x75_p3", 3),
    ("x75_p2", 2),
    ("x75_p1", 1),
    (None, 0.0),
    ("x75_m1", -1),
    ("x75_m2", -2),
    ("x75_m3", -3),
    ("x75_m4", -4),
    ("x75_m5", -5),
    ("x75_m6", -6),
    ("x75_m7", -7),
]
CANDIDATE = {"x86_full", "x75_m3"}

comp = json.loads(RESULTS.read_text())["comparisons"]
comp.update(json.loads(RESULTS_POS.read_text())["comparisons"])

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)

for ax, doses, title, xlabel in [
    (axes[0], WRITE_DOSES, "boundary write", "signed dose $a$"),
    (axes[1], EDIT_DOSES, "rank-one edit", "signed scale $s$"),
]:
    # registered non-inferiority margin for the guard endpoints
    ax.axhspan(-0.02, 0.02, color=ORANGE, alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    for key, label, color, marker in ENDPOINTS:
        xs, ys, lo, hi = [], [], [], []
        for cond, dose in doses:
            xs.append(dose)
            if cond is None:
                ys.append(0.0)
                lo.append(0.0)
                hi.append(0.0)
            else:
                e = comp[cond][key]
                ys.append(e["mean_delta"])
                lo.append(e["mean_delta"] - e["two_sided_ci"][0])
                hi.append(e["two_sided_ci"][1] - e["mean_delta"])
        ax.errorbar(
            xs,
            ys,
            yerr=[lo, hi],
            color=color,
            marker=marker,
            ms=3.5,
            lw=1.2,
            elinewidth=0.7,
            capsize=1.5,
            label=label,
            markerfacecolor="white" if marker != "o" else color,
            markeredgewidth=0.8,
            zorder=3,
        )
    # candidate dose: filled marker overlay; base anchor at 0
    cand = [d for c, d in doses if c in CANDIDATE][0]
    for key, label, color, marker in ENDPOINTS:
        cond = [c for c, d in doses if d == cand][0]
        ax.plot(
            cand,
            comp[cond][key]["mean_delta"],
            marker=marker,
            ms=4.5,
            color=color,
            zorder=4,
        )
    ax.plot(0, 0, marker="o", ms=3.5, color=MUTED, zorder=5)
    ax.annotate(
        "base",
        (0, 0),
        textcoords="offset points",
        xytext=(0, 5),
        ha="center",
        fontsize=6,
        color=MUTED,
    )
    ax.invert_xaxis()
    ax.set_title(title, color=INK)
    ax.set_xlabel(xlabel)
    if doses is WRITE_DOSES:
        ax.set_xticks([1.5, 1.0, 0.5, 0, -0.5, -1.0, -1.5, -2.0, -3.0])
        ax.set_xticklabels(
            ["+1.5", "+1", "+.5", "0", "−.5", "−1", "−1.5", "−2", "−3"]
        )
    else:
        ax.set_xticks([3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7])
        ax.set_xticklabels(
            ["+3", "+2", "+1", "0", "−1", "−2", "−3", "−4", "−5", "−6", "−7"]
        )
    ax.grid(axis="y", zorder=0)

axes[0].set_ylabel("change vs. base")
axes[0].text(
    -3.0,
    -0.019,
    "±0.02 margin",
    fontsize=6,
    color=MUTED,
    ha="right",
    va="bottom",
)
axes[0].legend(loc="lower left", frameon=False, handlelength=1.6)

fig.tight_layout(pad=0.4)
fig.savefig(OUT)
print("wrote", OUT)
