"""Diagnose whether early stopping explains the seed variance on the excluded benchmark.

The stopper (pykeen_utils.ValidationStopper) evaluates every 20 epochs, keeps the
checkpoint with the lowest validation mean rank, and stops after `tolerance`
consecutive evaluations without improvement. kge_transd.py reloads that best
checkpoint before testing, so a short run is not the same as an undertrained model:
each run is tested at its own validation optimum.

That shifts the question from "did the run stop too early" to "does validation mean
rank identify a good model at all". Three checks:

  1. Validation noise. How much does validation mean rank move between consecutive
     evaluations, relative to the improvement the stopper is trying to detect? If the
     step-to-step swing is comparable to the total improvement, the tolerance counter
     is reacting mostly to noise.
  2. Selection skill. Across seeds, does a better validation mean rank predict a
     better test mean rank? If the correlation is weak, model selection is close to
     random and the test variance follows from it.
  3. Stop time. Does the epoch at which a run stopped predict its test mean rank?
     A strong relationship would mean the stopping rule, not initialization, drives
     the spread.

Usage, from the excluded-benchmark working directory:
    python diagnose_early_stopping.py --logs logs --results data/results
"""
import glob
import os
import re

import click as ck
import numpy as np

VARIANTS = {"f": "func", "fs": "func_expr"}
CONFIG = "dim_100_bs_16384_lr_0.001"
VAL_LINE = re.compile(r"Epoch (\d+), Val \w+ MR: ([\d.]+), Best Val MR: ([\d.]+), Tolerance left: (-?\d+)")


def validation_curve(log_path):
    """(epoch, val_mr, best_so_far, tolerance_left) for each evaluation in a run."""
    rows = []
    with open(log_path, errors="ignore") as handle:
        for line in handle:
            m = VAL_LINE.search(line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)),
                             float(m.group(3)), int(m.group(4))))
    return rows


def test_mean_rank(results_dir, variant_source, seed, aggregation="bma"):
    pattern = (f"{results_dir}/kge_results_transd_fold_0_seed_{seed}_{CONFIG}_"
               f"{variant_source}_proj_owl2vecstar_gda_use_graph_True_by_graph_{aggregation}.tsv")
    hits = glob.glob(pattern)
    if not hits:
        return None
    ranks = []
    with open(hits[0]) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            idx = int(parts[2])
            scores = np.asarray(parts[3:], dtype=float)
            ranks.append(1 + int(np.count_nonzero(scores > scores[idx])))
    return float(np.mean(ranks))


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@ck.command()
@ck.option("--logs", default="logs")
@ck.option("--results", default="data/results")
@ck.option("--seeds", default=10)
def main(logs, results, seeds):
    for variant, source in VARIANTS.items():
        print(f"{'=' * 72}\nLinkGDA-{variant}\n{'=' * 72}")
        best_vals, test_mrs, stop_epochs, best_epochs, swings = [], [], [], [], []

        print(f"{'seed':>4} {'evals':>6} {'best_val_MR':>12} {'best_ep':>8} "
              f"{'stop_ep':>8} {'test_MR':>9} {'med|Δval|':>10}")
        for seed in range(seeds):
            path = os.path.join(logs, f"train_{variant}_seed{seed}.log")
            if not os.path.exists(path):
                continue
            curve = validation_curve(path)
            if not curve:
                print(f"{seed:>4}   no validation lines parsed")
                continue
            epochs = [r[0] for r in curve]
            vals = [r[1] for r in curve]
            best_val = min(vals)
            best_ep = epochs[int(np.argmin(vals))]
            deltas = np.abs(np.diff(vals)) if len(vals) > 1 else np.array([np.nan])
            med_swing = float(np.median(deltas))
            tmr = test_mean_rank(results, source, seed)

            best_vals.append(best_val); stop_epochs.append(epochs[-1])
            best_epochs.append(best_ep); swings.append(med_swing)
            test_mrs.append(tmr if tmr is not None else np.nan)

            print(f"{seed:>4} {len(curve):>6} {best_val:>12.2f} {best_ep:>8} "
                  f"{epochs[-1]:>8} {tmr if tmr is None else f'{tmr:>9.2f}'} {med_swing:>10.2f}")

        if len(best_vals) < 3:
            print("  too few runs to correlate\n")
            continue

        total_improvement = float(np.mean([v for v in best_vals]))
        print(f"\n  validation noise : median |change| between consecutive evaluations "
              f"= {np.median(swings):.2f} rank positions")
        print(f"  selection skill  : corr(best val MR, test MR) = {pearson(best_vals, test_mrs):+.3f}"
              f"   (want strongly positive)")
        print(f"  stop-time effect : corr(stop epoch, test MR)  = {pearson(stop_epochs, test_mrs):+.3f}")
        print(f"  best-epoch effect: corr(best epoch, test MR)  = {pearson(best_epochs, test_mrs):+.3f}")
        print(f"  test MR spread   : {np.nanmin(test_mrs):.1f} to {np.nanmax(test_mrs):.1f}"
              f"  (sd {np.nanstd(test_mrs, ddof=1):.1f})")
        print(f"  val  MR spread   : {min(best_vals):.1f} to {max(best_vals):.1f}"
              f"  (sd {np.std(best_vals, ddof=1):.1f})\n")


if __name__ == "__main__":
    main()
