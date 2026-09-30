"""Calibration schematic for panel (b) of fig:overview: per-gene calibration.

Two hypothetical genes, A and B, are scored against nine reference query
diseases and one new query disease. The reference scores are fixed lists, so
the figure is deterministic. Each gene's mean and standard deviation are
computed from its list (population SD, as in calibrate_scores.py), and the new
query is not part of the list. The upper panel shows raw scores, on which the
query for Gene B outranks the query for Gene A. The lower panel subtracts each
gene's own mean and divides by its own SD, and the order reverses. Nothing here
is experimental data.

    python code/figures/make_calibration_fig.py [output-stem]

Writes <stem>.pdf and <stem>.png; the default stem is
paper/fig/fig_calibration.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FONT = "cm"
DEFAULT_STEM = str(Path(__file__).resolve().parents[2] / "paper" / "fig" / "fig_calibration")

GENES = ["Gene A", "Gene B"]
COLORS = {"Gene A": "#0072b2", "Gene B": "#d55e00"}
ROW_Y = {"Gene A": 1.0, "Gene B": 0.0}
REFERENCE = {
    "Gene A": [0.07, 0.09, 0.12, 0.14, 0.15, 0.16, 0.18, 0.21, 0.23],
    "Gene B": [0.21, 0.27, 0.36, 0.42, 0.45, 0.48, 0.54, 0.63, 0.69],
}
QUERY = {"Gene A": 0.30, "Gene B": 0.50}

FIG_W, FIG_H = 3.08, 3.40
AX_LEFT, AX_W, AX_H = 0.66, 2.00, 0.80
AX_BOTTOMS = [2.36, 1.10]
LEGEND_Y = 0.28
NOTE_Y = 0.10


def configure_fonts(style):
    if style == "cm":
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
        })
    else:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
        })
    plt.rcParams.update({
        "font.size": 7,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })


def baseline(reference):
    ref = np.asarray(reference, dtype=float)
    return float(ref.mean()), float(ref.std())


def standardize(values, mean, sd):
    return (np.asarray(values, dtype=float) - mean) / sd


def ranks(values):
    order = sorted(values, key=values.get, reverse=True)
    return {gene: i + 1 for i, gene in enumerate(order)}


def ordinal(k):
    return {1: "1st", 2: "2nd"}[k]


def draw_row(ax, y, color, reference, mean, sd, query, label):
    ax.plot([mean - sd, mean + sd], [y, y], color=color, lw=1.2,
            solid_capstyle="butt", zorder=1)
    ax.plot([mean, mean], [y - 0.18, y + 0.18], color=color, lw=1.2, zorder=1)
    ax.plot(reference, [y] * len(reference), ls="", marker="o", ms=3.2,
            mfc="white", mec=color, mew=0.8, zorder=2)
    ax.plot([query], [y], ls="", marker="*", ms=8.5, mfc=color, mec="black",
            mew=0.5, zorder=3)
    ax.annotate(label, (query, y), xytext=(0, 5.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color=color, zorder=4)


def draw_panel(ax, title, xlabel, xlim, xticks, points, means, sds, queries,
               labels, ylabels, rank):
    for gene in GENES:
        y = ROW_Y[gene]
        draw_row(ax, y, COLORS[gene], points[gene], means[gene], sds[gene],
                 queries[gene], labels[gene])
        ax.text(1.05, y, r"$\mathbf{%s}$" % ordinal(rank[gene]),
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7.5, color=COLORS[gene])
    ax.text(1.05, 1.0, "rank", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color="0.45")
    ax.set_title(title, loc="left", fontsize=8, pad=4)
    ax.set_xlabel(xlabel, fontsize=7, labelpad=2)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.6, 1.85)
    ax.set_yticks([ROW_Y[g] for g in GENES])
    ax.set_yticklabels([ylabels[g] for g in GENES], fontsize=7)
    for tick, gene in zip(ax.get_yticklabels(), GENES):
        tick.set_color(COLORS[gene])
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", labelsize=6.5, length=2.5, pad=2)
    ax.spines[["top", "right", "left"]].set_visible(False)


def main(stem):
    configure_fonts(FONT)

    means, sds, zref, zq = {}, {}, {}, {}
    for gene in GENES:
        means[gene], sds[gene] = baseline(REFERENCE[gene])
        zref[gene] = standardize(REFERENCE[gene], means[gene], sds[gene])
        zq[gene] = float(standardize(QUERY[gene], means[gene], sds[gene]))

    raw_rank = ranks(QUERY)
    z_rank = ranks(zq)
    assert raw_rank["Gene B"] == 1 and z_rank["Gene A"] == 1

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = [fig.add_axes([AX_LEFT / FIG_W, b / FIG_H, AX_W / FIG_W, AX_H / FIG_H])
            for b in AX_BOTTOMS]

    draw_panel(
        axes[0], "Raw scores", "score (higher is better)", (0.0, 0.8),
        [0.0, 0.2, 0.4, 0.6, 0.8], REFERENCE, means, sds, QUERY,
        {g: f"{QUERY[g]:.2f}" for g in GENES},
        {g: f"{g}\n" + r"$%.2f \pm %.2f$" % (means[g], sds[g]) for g in GENES},
        raw_rank)
    draw_panel(
        axes[1], "Calibrated scores", r"$z$ = (score $-$ mean) / SD", (-2.0, 3.6),
        [-2, -1, 0, 1, 2, 3], zref, {g: 0.0 for g in GENES},
        {g: 1.0 for g in GENES}, zq,
        {g: f"{zq[g]:.2f}" for g in GENES},
        {g: f"{g}\n" + r"$0 \pm 1$" for g in GENES},
        z_rank)

    n_ref = len(REFERENCE["Gene A"])
    handles = [
        Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec="0.3", mew=0.8),
        Line2D([], [], color="0.3", lw=1.2, marker="|", ms=6, mew=1.2),
        Line2D([], [], ls="", marker="*", ms=8, mfc="0.3", mec="black", mew=0.5),
    ]
    labels = [
        f"{n_ref} reference diseases (new query excluded)",
        r"reference mean $\pm$ 1 SD",
        "new query disease",
    ]
    fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=6.5,
               ncol=1, handlelength=1.6, handletextpad=0.6, labelspacing=0.45,
               borderpad=0.0, bbox_to_anchor=(0.5, LEGEND_Y / FIG_H))
    fig.text(0.5, NOTE_Y / FIG_H, "Hypothetical example. Higher score is better.",
             ha="center", va="center", fontsize=6.5, color="0.35")

    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    for gene in GENES:
        print(f"{gene}: mean {means[gene]:.4f}  sd {sds[gene]:.4f}  "
              f"query {QUERY[gene]:.2f}  z {zq[gene]:.4f}  "
              f"raw rank {raw_rank[gene]}  calibrated rank {z_rank[gene]}")
    print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM)
