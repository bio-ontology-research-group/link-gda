"""Overlap-stratified metrics for any prediction file, from precomputed labels.

A companion to leakage_overlap_perfold.py for the case where the prediction files and
the fold inputs live on different hosts: the labels come from make_overlap_labels.py as
a small TSV, so this script needs nothing but the score files and numpy. Several label
sets can be applied in one pass, which matters when the score files are gigabytes and
the host is a shared login node -- the definitions then cost one read, not one each.

Both settings come out of one read as well. Parsing dominates the cost, and the
uncalibrated and calibrated rankings differ only by an arithmetic step on the matrix
already in memory, so computing them separately would double the I/O for nothing.

Conventions are those of leakage_overlap_perfold.py and rq1_table.py, deliberately:
leave-one-out per-gene z-scoring for the calibrated setting, deterministic average-rank
tie handling, metrics pooled within a fold and then averaged over folds with a sample
standard deviation, and AUC as the closed form (N - MR) / (N - 1). A stratified number
that used any other convention could not be placed beside the published rows.

Only argparse and numpy are used, so the script runs under any environment that can
import numpy.
"""
import argparse
import os

import numpy as np

HITS_K = (1, 3, 10, 100)
COLS = ["MR", "MRR"] + [f"H@{k}" for k in HITS_K] + ["AUC"]


def load_labels(path):
    labels = {}
    with open(path) as handle:
        header = handle.readline()
        if not header.startswith("fold"):
            raise SystemExit(f"{path}: expected a fold/disease/gene/overlap header")
        for line in handle:
            fold, disease, gene, overlap = line.rstrip("\n").split("\t")
            labels[(int(fold), disease, gene)] = float(overlap)
    return labels


def calibrate(scores):
    n = scores.shape[0]
    if n < 3:
        return scores
    mean = (scores.sum(axis=0, keepdims=True) - scores) / (n - 1)
    var = ((scores ** 2).sum(axis=0, keepdims=True) - scores ** 2) / (n - 1) - mean ** 2
    return (scores - mean) / (np.sqrt(np.clip(var, 0, None)) + 1e-12)


def _ranks(matrix, positions):
    out = []
    for i, pos in enumerate(positions):
        true = matrix[i, pos]
        greater = int(np.count_nonzero(matrix[i] > true))
        equal = int(np.count_nonzero(matrix[i] == true))
        out.append(greater + (equal + 1) / 2.0)
    return np.array(out)


def fold_ranks(path):
    """(keys, {setting: ranks}, pool) for one fold's file, keys as (disease, gene)."""
    keys, rows, positions = [], [], []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            keys.append((parts[1], parts[0]))
            positions.append(int(parts[2]))
            rows.append(np.fromiter((float(x) for x in parts[3:]), dtype=np.float64))
    matrix = np.vstack(rows)
    return (keys,
            {"raw": _ranks(matrix, positions),
             "calibrated": _ranks(calibrate(matrix), positions)},
            matrix.shape[1])


def summarise(per_fold):
    return {c: (float(np.mean([r[c] for r in per_fold])),
                float(np.std([r[c] for r in per_fold], ddof=1)) if len(per_fold) > 1 else 0.0)
            for c in COLS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", action="append", required=True,
                    help="name=path, repeatable; each is one overlap definition")
    ap.add_argument("--method", action="append", required=True,
                    help="name=template, repeatable; template contains {fold}")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--pool", type=int, default=4399, help="candidate pool size for the AUC identity")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    label_sets = {}
    for item in args.labels:
        name, path = item.split("=", 1)
        label_sets[name] = load_labels(path)
        zero = sum(1 for v in label_sets[name].values() if v == 0)
        total = len(label_sets[name])
        print(f"labels {name}: {total} instances, zero overlap {zero} ({100 * zero / total:.1f}%)")

    strata = [("zero overlap", lambda o: o == 0),
              ("some overlap", lambda o: o > 0),
              ("all", lambda o: True)]
    settings = ("raw", "calibrated")
    lines = ["\t".join(["method", "definition", "setting", "stratum", "folds", "pool"]
                       + [f"{c}_{s}" for c in COLS for s in ("mean", "sd")])]

    for item in args.method:
        name, template = item.split("=", 1)
        collected = {(d, t): {s: [] for s, _ in strata} for d in label_sets for t in settings}
        missing = []
        pool = args.pool
        for fold in range(args.folds):
            path = template.format(fold=fold)
            if not os.path.exists(path):
                missing.append(fold)
                continue
            keys, ranks_by_setting, pool = fold_ranks(path)
            for definition, labels in label_sets.items():
                overlaps = np.array([labels.get((fold, d, g), np.nan) for d, g in keys])
                for label, test in strata:
                    mask = np.array([False if np.isnan(o) else bool(test(o)) for o in overlaps])
                    if not mask.any():
                        continue
                    for setting in settings:
                        values = ranks_by_setting[setting][mask]
                        row = {"MR": values.mean(), "MRR": float((1.0 / values).mean()),
                               "AUC": (pool - values.mean()) / (pool - 1)}
                        for k in HITS_K:
                            row[f"H@{k}"] = float((values <= k).mean())
                        collected[(definition, setting)][label].append(row)
        if missing:
            print(f"{name}: MISSING folds {missing}")
        if all(not collected[k][s] for k in collected for s, _ in strata):
            print(f"{name}: no prediction files found, skipped")
            continue
        for definition in label_sets:
            for setting in settings:
                for label, _ in strata:
                    per_fold = collected[(definition, setting)][label]
                    if not per_fold:
                        continue
                    stats = summarise(per_fold)
                    lines.append("\t".join([name, definition, setting, label, str(len(per_fold)), str(pool)]
                                           + [f"{v:.6f}" for c in COLS for v in stats[c]]))
        print(f"{name}: done ({args.folds - len(missing)} folds)")

    with open(args.out, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines) - 1} rows)")


if __name__ == "__main__":
    main()
