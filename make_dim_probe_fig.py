"""Figure: validation mean rank against embedding dimension.

Reads data/dim_probe_pheno.csv, the 24-configuration sweep at each of four
dimensions on the phenotype graph (fold 0, seed 0), and writes
paper/fig/fig_dim_probe.pdf.

The figure carries a two-part claim: the ordering of dimensions is not
monotonic, and the differences between dimensions are smaller than the noise.
Each panel therefore shows every configuration rather than a summary, marks the
median per dimension, and draws a bar whose height is the seed-to-seed standard
deviation so the reader can compare the spread of the medians against it.

    python make_dim_probe_fig.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DIMS = [100, 200, 400, 800]
LR_COLORS = {"0.0001": "#1b6ca8", "0.001": "#e08214", "0.01": "#9970ab"}
SEED_SD = {"raw_val_mr": 313.0, "cal_val_mr": 76.0}
PANELS = [("raw_val_mr", "uncalibrated"), ("cal_val_mr", "calibrated")]


def load(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def main():
    rows = load("data/dim_probe_pheno.csv")
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for ax, (column, title) in zip(axes, PANELS):
        medians = []
        for x, dim in enumerate(DIMS):
            vals = [(float(r[column]), r["lr"]) for r in rows if int(r["dim"]) == dim]
            jitter = rng.uniform(-0.16, 0.16, len(vals))
            for (value, lr), dx in zip(vals, jitter):
                ax.plot(x + dx, value, "o", ms=3.4, alpha=0.75,
                        color=LR_COLORS.get(lr, "grey"),
                        markeredgewidth=0, zorder=2)
            median = float(np.median([v for v, _ in vals]))
            medians.append(median)
            ax.hlines(median, x - 0.3, x + 0.3, color="black", lw=1.8, zorder=3)

        ax.plot(range(len(DIMS)), medians, color="black", lw=1.0,
                ls="--", alpha=0.55, zorder=1)

        sd = SEED_SD[column]
        lo, hi = ax.get_ylim()
        span = hi - lo
        ax.set_ylim(lo - 0.06 * span, hi + 0.16 * span)
        bar_x = len(DIMS) - 0.42
        bar_top = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.errorbar(bar_x, bar_top - sd / 2, yerr=sd / 2, color="crimson",
                    capsize=3, lw=1.4, zorder=4)
        ax.text(bar_x - 0.12, bar_top - sd / 2, f"seed sd\n{sd:.0f}", color="crimson",
                fontsize=7, ha="right", va="center")

        ax.set_xticks(range(len(DIMS)))
        ax.set_xticklabels(DIMS)
        ax.set_xlabel("embedding dimension")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("validation mean rank")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=4.5,
                          label=f"lr {lr}") for lr, c in LR_COLORS.items()]
    handles.append(plt.Line2D([], [], color="black", lw=1.8, label="median"))
    axes[1].legend(handles=handles, fontsize=7, frameon=False,
                   loc="upper left", ncol=2)

    fig.tight_layout()
    os.makedirs("paper/fig", exist_ok=True)
    fig.savefig("paper/fig/fig_dim_probe.pdf", bbox_inches="tight")
    print("wrote paper/fig/fig_dim_probe.pdf")

    for column, title in PANELS:
        line = [f"{title:>13}:"]
        for dim in DIMS:
            vals = [float(r[column]) for r in rows if int(r["dim"]) == dim]
            line.append(f"d{dim} med {np.median(vals):7.1f} min {min(vals):7.1f}")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
