"""Rank-CDF figures for the paper, from the pooled per-instance ranks.

Data are the empirical CDF percentages P(rank <= x) computed by
rank_cdf_median.py over the ten folds (all 6,571 instances), on the grid
GRID, split by phenotype overlap. Emits two PDFs into paper/fig/:
  fig_rankcdf_two.pdf     two panels (zero overlap | overlap>0)
  fig_rankcdf_single.pdf  one panel (all instances)
Regenerate after rerunning rank_cdf_median.py if the ranks change.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.6})

GRID = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4399]
RANDOM = [x / 4399 * 100 for x in GRID]

CDF = {
 "all": {
  "LinkGDA-pf": [1.6,4.4,6.5,10.3,16.0,27.2,38.4,49.8,66.2,79.1,91.4,100],
  "LinkGDA-p":  [0.8,2.5,3.6,6.4,10.6,18.7,28.4,39.3,56.9,72.9,87.7,100],
  "INDIGENA":   [1.8,4.8,6.9,11.6,18.2,29.8,40.3,49.6,61.9,70.7,81.3,100],
  "Resnik-BMA": [1.8,5.0,6.9,10.7,15.5,24.0,31.7,39.9,51.1,60.8,71.9,100]},
 "zero": {
  "LinkGDA-pf": [1.1,4.1,6.4,9.9,15.1,26.8,38.9,51.1,69.8,83.4,94.5,100],
  "LinkGDA-p":  [0.4,1.6,2.5,4.4,8.1,15.6,25.6,36.8,56.8,75.2,90.7,100],
  "INDIGENA":   [1.3,3.3,4.9,8.7,13.9,23.5,32.5,40.9,53.0,62.4,74.9,100],
  "Resnik-BMA": [1.6,4.2,5.7,9.2,13.1,20.6,27.0,34.3,44.1,53.4,65.5,100]},
 "some": {
  "LinkGDA-pf": [2.2,4.7,6.6,10.9,17.2,27.6,37.7,48.2,61.7,73.6,87.4,100],
  "LinkGDA-p":  [1.3,3.6,5.0,8.9,13.8,22.8,31.9,42.5,57.0,69.9,83.8,100],
  "INDIGENA":   [2.6,6.7,9.4,15.3,23.7,37.8,50.2,60.7,73.1,81.2,89.5,100],
  "Resnik-BMA": [2.2,5.9,8.4,12.6,18.5,28.4,37.6,47.0,60.1,70.1,80.1,100]},
}

STYLE = {  # label: (color, linestyle, linewidth)
 "LinkGDA-pf": ("#1f77b4", "-", 1.6),
 "LinkGDA-p":  ("#1f77b4", "--", 1.4),
 "INDIGENA":   ("#d62728", "-", 1.6),
 "Resnik-BMA": ("#555555", "-", 1.3),
}
ORDER = ["LinkGDA-pf", "LinkGDA-p", "INDIGENA", "Resnik-BMA"]


def draw(ax, stratum, legend=True):
    for name in ORDER:
        c, ls, lw = STYLE[name]
        ax.plot(GRID, CDF[stratum][name], color=c, linestyle=ls, linewidth=lw, label=name)
    ax.plot(GRID, RANDOM, color="black", linestyle=":", linewidth=0.9, label="random")
    ax.set_xscale("log"); ax.set_xlim(1, 4399); ax.set_ylim(0, 100)
    ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.set_xlabel(r"rank $x$")
    ax.grid(True, which="both", linewidth=0.3, color="0.85")
    ax.tick_params(labelsize=8)
    if legend:
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)


# two panels
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
draw(axes[0], "zero", legend=True); axes[0].set_title("Zero overlap (55.9%)", fontsize=9)
axes[0].set_ylabel(r"instances with rank $\leq x$ (%)")
draw(axes[1], "some", legend=False); axes[1].set_title(r"Overlap $>0$ (44.1%)", fontsize=9)
fig.tight_layout(); fig.savefig("paper/fig/fig_rankcdf_two.pdf"); plt.close(fig)

# single panel
fig, ax = plt.subplots(figsize=(3.6, 3.0))
draw(ax, "all", legend=True)
ax.set_ylabel(r"instances with rank $\leq x$ (%)")
fig.tight_layout(); fig.savefig("paper/fig/fig_rankcdf_single.pdf"); plt.close(fig)
print("wrote paper/fig/fig_rankcdf_two.pdf and fig_rankcdf_single.pdf")
