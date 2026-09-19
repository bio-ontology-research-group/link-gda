"""Figure: per-gene calibration, illustrative example (panel b of fig:overview).

Two genes with different score scales are shown before and after per-gene
calibration. The upper block scores each gene on its own arbitrary scale; the
lower block expresses the same query scores in standard deviations from that
gene's reference panel (the panel excludes the new query disease, so
calibration uses no ground-truth association). The raw scores rank Gene 2
above Gene 1; after calibration, Gene 1's score sits three SDs above its own
mean against Gene 2's third of an SD, so the rank reverses.

The layout is a tall, narrow column so the figure can sit beside the training
graph in a single row of fig:overview.

The numbers are embedded (illustrative, not experimental data).

    python code/make_calibration_fig.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

GENE_COLORS = {"Gene 1": "#0072b2", "Gene 2": "#d55e00"}
GENE_MARKERS = {"Gene 1": "o", "Gene 2": "D"}
# name -> (reference mean, reference SD, query score)
GENES = {
    "Gene 1": (0.15, 0.05, 0.30),
    "Gene 2": (0.45, 0.15, 0.50),
}
BLOCKS = [("Raw scores", "Score (arbitrary units)",
           "Gene 2 ranks above Gene 1"),
          ("Calibrated scores", "Standard deviations from the gene's mean",
           "Gene 1 ranks above Gene 2")]


def zscore(mean, sd, query):
    return (query - mean) / sd


def draw_cell(ax, gene, block):
    mean, sd, query = GENES[gene]
    color = GENE_COLORS[gene]
    if block == 0:
        band = (mean - sd, mean + sd)
        m, q = mean, query
        qlabel = f"s = {query:.2f}"
        blabel = f"Mean {mean:.2f}; SD {sd:.2f}"
    else:
        band = (-1.0, 1.0)
        m, q = 0.0, zscore(mean, sd, query)
        qlabel = f"z = {q:.2f}"
        blabel = f"{q:.2f} SDs from its mean"
    ax.axvspan(band[0], band[1], color=color, alpha=0.15, lw=0)
    ax.axvline(m, ymin=0, ymax=0.46, color=color, lw=1.8)
    ax.plot(q, 0.34, marker=GENE_MARKERS[gene], ms=5.5, mfc=color, mec=color,
            zorder=3)
    ax.annotate(qlabel, (q, 0.34), xytext=(0, 4.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=7, color=color, zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.3))
    ax.text(0.02, 0.98, blabel, transform=ax.transAxes, fontsize=6.5,
            color="0.3", va="top", bbox=dict(fc="white", ec="none", pad=0.3))
    ax.set_xlim((-1.3, 3.5) if block else (0.0, 0.9))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)


def main():
    w, h = 3.22, 3.58
    left, axw = 0.55, 2.61
    axh = 0.44
    bottoms = [2.90, 2.39, 1.30, 0.79]
    verdict_y = [2.09, 0.49]

    fig = plt.figure(figsize=(w, h))
    axes = [fig.add_axes([left / w, b / h, axw / w, axh / h]) for b in bottoms]

    for block, (title, xlabel, verdict) in enumerate(BLOCKS):
        top, bot = axes[2 * block], axes[2 * block + 1]
        for ax, gene in zip((top, bot), GENES):
            draw_cell(ax, gene, block)
            ax.set_ylabel(gene, fontsize=7.5, labelpad=3)
            ax.tick_params(axis="x", labelsize=7, pad=2)
        top.set_xlim(bot.get_xlim())
        top.set_xticklabels([])
        top.set_title(title, fontsize=8.5, loc="left", pad=3)
        bot.set_xlabel(xlabel, fontsize=7.5, labelpad=2)
        fig.text((left + axw / 2) / w, verdict_y[block] / h, verdict,
                 ha="center", va="top", fontsize=8, fontweight="bold")

    fig.text(1 - 0.06 / w, 1 - 0.005 / h, "Illustrative example", ha="right",
             va="top", fontsize=7, color="0.3")

    handles = [Patch(fc="#000000", alpha=0.15,
                     label="Reference mean plus/minus one SD "
                           "(excludes the query disease)"),
               (Line2D([], [], marker="o", ls="", ms=5, mfc="#000000",
                       mec="#000000"),
                Line2D([], [], marker="D", ls="", ms=5, mfc="#000000",
                       mec="#000000"))]
    labels = [handles[0].get_label(), "Score for the new query disease"]
    fig.legend(handles=handles, labels=labels, fontsize=6.0, frameon=False,
               loc="lower center", bbox_to_anchor=(0.52, 0.005), ncol=1,
               handlelength=1.8, handletextpad=0.6, labelspacing=0.55,
               borderpad=0.0,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})

    os.makedirs("paper/fig", exist_ok=True)
    fig.savefig("paper/fig/fig_calibration.pdf")
    print("wrote paper/fig/fig_calibration.pdf")


if __name__ == "__main__":
    main()
