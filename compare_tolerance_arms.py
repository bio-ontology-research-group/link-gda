"""Compare the two early-stopping arms of the excluded-gene benchmark.

Both arms use the same fixed train/validation split (--val_seed 0), so a seed varies
only initialization and negative sampling, and the two arms differ only in patience.
Training is deterministic given the seed, so a seed whose best validation mean rank was
already reached before the shorter arm stopped will reload the same checkpoint and
produce an identical test mean rank. Those seeds are reported as unchanged rather than
treated as failures: they are the cases where extra patience found nothing better.

Reports, per variant: each arm's mean and spread across seeds, the paired per-seed
difference, and a paired t-test over seeds. The paired test is the right one here
because the arms share seeds, so each pair differs only in patience.

Usage, from the benchmark root holding tol5/ and tol15/:
    python compare_tolerance_arms.py --base . --arms 5,15
"""
import os

import click as ck
import numpy as np
from scipy import stats

CONFIG = "dim_100_bs_16384_lr_0.001"
VARIANTS = {"f": "func", "fs": "func_expr"}


def ranks(path):
    out = []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            idx = int(parts[2])
            scores = np.asarray(parts[3:], dtype=float)
            out.append(1 + int(np.count_nonzero(scores > scores[idx])))
    return np.asarray(out, dtype=float)


def arm_mean_ranks(base, arm, source, seeds, aggregation):
    """Per-seed test mean rank for one arm, or None where the run is missing."""
    values = {}
    for seed in range(seeds):
        path = (f"{base}/tol{arm}/data/results/kge_results_transd_fold_0_seed_{seed}_"
                f"{CONFIG}_{source}_proj_owl2vecstar_gda_use_graph_True_by_graph_"
                f"{aggregation}.tsv")
        values[seed] = float(ranks(path).mean()) if os.path.exists(path) else None
    return values


@ck.command()
@ck.option("--base", default=".", help="Directory holding tol5/ and tol15/")
@ck.option("--arms", default="5,15", help="Comma-separated tolerance values")
@ck.option("--seeds", default=10)
@ck.option("--aggregation", default="bma", type=ck.Choice(["bma", "bmm"]))
def main(base, arms, seeds, aggregation):
    lo, hi = [a.strip() for a in arms.split(",")]

    for variant, source in VARIANTS.items():
        a = arm_mean_ranks(base, lo, source, seeds, aggregation)
        b = arm_mean_ranks(base, hi, source, seeds, aggregation)
        paired = [(s, a[s], b[s]) for s in range(seeds)
                  if a.get(s) is not None and b.get(s) is not None]

        print(f"{'=' * 66}\nLinkGDA-{variant}   ({len(paired)}/{seeds} seeds in both arms)\n{'=' * 66}")
        missing = [s for s in range(seeds) if a.get(s) is None or b.get(s) is None]
        if missing:
            print(f"  incomplete seeds: {missing}")
        if not paired:
            print("  nothing to compare yet\n")
            continue

        print(f"{'seed':>5} {f'tol{lo} MR':>10} {f'tol{hi} MR':>10} {'change':>10}")
        for s, x, y in paired:
            note = "   (same checkpoint)" if abs(x - y) < 1e-9 else ""
            print(f"{s:>5} {x:>10.2f} {y:>10.2f} {y - x:>+10.2f}{note}")

        xs = np.array([x for _, x, _ in paired])
        ys = np.array([y for _, _, y in paired])
        print(f"\n  tol{lo:<3} mean {xs.mean():8.2f} +/- {xs.std(ddof=1):7.2f}"
              f"   [min {xs.min():.1f}, max {xs.max():.1f}]")
        print(f"  tol{hi:<3} mean {ys.mean():8.2f} +/- {ys.std(ddof=1):7.2f}"
              f"   [min {ys.min():.1f}, max {ys.max():.1f}]")
        print(f"  mean paired change {(ys - xs).mean():+.2f}"
              f"   improved {int((ys < xs).sum())}/{len(paired)},"
              f" unchanged {int((ys == xs).sum())}")

        if len(paired) >= 3 and np.any(ys != xs):
            t, p = stats.ttest_rel(ys, xs)
            print(f"  paired t over seeds (two-sided): t={t:.3f}  p={p:.4g}")
        else:
            print("  paired t: not run (too few differing seeds)")
        print()


if __name__ == "__main__":
    main()
